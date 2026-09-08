"""Regression tests for seven fixes (each test reproduces the original failure).

1. torch_training_meta_info profiling probe must not train the model.
2. ShufflePipeline must yield every item exactly once.
3. SQLiteDB.insert_many column order / transaction() rollback.
4. save_image transform writes RGB PNGs with correct colour order.
5. PyTorchPipeline.to_framework_dataset applies pre/post hooks; TF refuses them.
6. DataLoader helpers are module-level and picklable (spawn workers).
7. Epoch losses are sample-weighted, also after an early break.

Torch / OpenCV dependent tests are skipped when the library is missing.
"""

import os
import pickle
import random
import sqlite3
import tempfile
import unittest

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from xflow.data.pipeline import PyTorchPipeline, TensorFlowPipeline
from xflow.data.provider import DataProvider
from xflow.data.transform import (
    PyTorchTransformDataset,
    ShufflePipeline,
    _MemoryListDataset,
    _seed_worker,
)
from xflow.data.transform import save_image as save_image_transform
from xflow.trainers.trainer import _num_samples
from xflow.utils.sql import SQLiteDB


# ------------------------------------------------------------------ helpers
class _ListProvider(DataProvider):
    """Minimal picklable provider over a Python list."""

    def __init__(self, items):
        self._items = list(items)

    def __call__(self):
        return list(self._items)

    def __len__(self):
        return len(self._items)

    def subsample(self, *args, **kwargs):
        return self


def _double(x):
    return x * 2


def _pre_hook(item, idx):
    return item + idx


def _post_hook(item, idx):
    return (item, idx)


# ------------------------------------------------------------ 1. profiling
@unittest.skipUnless(torch is not None, "torch not installed")
class ProfilingCallbackIsMeasurementOnly(unittest.TestCase):
    def _model(self):
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, 3, padding=1),
            torch.nn.BatchNorm2d(2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(2 * 8 * 8, 1),
        )
        model[0].weight.requires_grad_(False)  # a frozen layer
        model.train()
        return model

    def test_probe_leaves_weights_flags_mode_and_optimizer_untouched(self):
        from xflow.trainers.callback import make_torch_training_meta_info

        model = self._model()
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        flags = [p.requires_grad for p in model.parameters()]

        with tempfile.TemporaryDirectory() as d:
            cb = make_torch_training_meta_info(save_dir=d, input_shape=(2, 1, 8, 8))
            cb.on_train_begin(model=model, epochs=1, total_batches=1)
            cb.on_batch_end(batch=0, model=model, optimizer=opt)

        after = model.state_dict()
        for k in before:  # weights AND BatchNorm running stats
            self.assertTrue(torch.equal(before[k], after[k]), k)
        self.assertEqual([p.requires_grad for p in model.parameters()], flags)
        self.assertTrue(model.training)
        self.assertEqual(len(opt.state), 0)  # optimizer.step() never ran
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(cb._did_profile_once)

    def test_probe_never_runs_on_the_live_model(self):
        # A forward pass alone mutates some modules: Embedding(max_norm)
        # renormalises looked-up rows in place, and a recursive .train()
        # restore would clobber a selectively frozen BatchNorm. Both stay
        # intact only if the probe runs on a copy.
        from xflow.trainers.callback import make_torch_training_meta_info

        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Embedding(10, 4, max_norm=0.5),
            torch.nn.Flatten(),
            torch.nn.Linear(12, 4),
            torch.nn.BatchNorm1d(4),
            torch.nn.Linear(4, 1),
        )
        model.train()
        model[3].eval()  # frozen BatchNorm inside a training model
        before = {k: v.clone() for k, v in model.state_dict().items()}
        idx = torch.tensor([[0, 1, 2], [3, 4, 5]])

        with tempfile.TemporaryDirectory() as d:
            cb = make_torch_training_meta_info(save_dir=d, example_input=idx)
            cb.on_batch_end(batch=0, model=model, optimizer=None)

        after = model.state_dict()
        for k in before:  # embedding rows 0-5 would have been renormalised
            self.assertTrue(torch.equal(before[k], after[k]), k)
        self.assertTrue(model.training)
        self.assertFalse(model[3].training)
        self.assertTrue(cb._did_profile_once)

    def test_failed_probe_is_attempted_only_once(self):
        from xflow.trainers.callback import make_torch_training_meta_info

        model = torch.nn.Linear(4, 1)
        with tempfile.TemporaryDirectory() as d:
            cb = make_torch_training_meta_info(
                save_dir=d, input_shape=(2, 3), method="profiler"  # wrong shape
            )
            cb.on_batch_end(batch=0, model=model, optimizer=None)
            self.assertTrue(cb._did_profile_once)
            self.assertIsNone(cb.per_step_flops)


# -------------------------------------------------------------- 2. shuffle
class ShuffleBufferYieldsEachItemOnce(unittest.TestCase):
    def test_dataset_larger_than_buffer(self):
        random.seed(0)
        base = PyTorchPipeline(_ListProvider(range(100)), [])
        out = list(ShufflePipeline(base, buffer_size=7))
        self.assertEqual(sorted(out), list(range(100)))  # no dups, no drops
        self.assertNotEqual(out, list(range(100)))  # and actually shuffled

    def test_buffer_larger_than_dataset(self):
        base = PyTorchPipeline(_ListProvider(range(5)), [])
        self.assertEqual(sorted(ShufflePipeline(base, buffer_size=100)), list(range(5)))


# ------------------------------------------------------------------ 3. sql
class SQLiteWritesAndTransactions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "t.db")
        self.db = SQLiteDB(self.path)
        self.db.create_table("t", {"a": "INTEGER", "b": "TEXT"})

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def _rows(self, sql):
        conn = sqlite3.connect(
            self.path
        )  # independent connection: committed state only
        try:
            return conn.execute(sql).fetchall()
        finally:
            conn.close()

    def test_insert_many_uses_one_fixed_column_order(self):
        self.db.insert_many("t", [{"a": 1, "b": "x"}, {"b": "y", "a": 2}])
        self.assertEqual(
            self._rows("SELECT a, b FROM t ORDER BY a"), [(1, "x"), (2, "y")]
        )

    def test_insert_many_rejects_mismatched_columns(self):
        with self.assertRaises(ValueError):
            self.db.insert_many("t", [{"a": 1, "b": "x"}, {"a": 2}])

    def test_transaction_rolls_back_every_write_in_the_block(self):
        with self.assertRaises(RuntimeError):
            with self.db.transaction():
                self.db.insert("t", {"a": 1, "b": "x"})
                self.db.insert("t", {"a": 2, "b": "y"})
                raise RuntimeError("boom")
        self.assertEqual(self._rows("SELECT COUNT(*) FROM t"), [(0,)])

    def test_transaction_commits_at_outer_boundary_only(self):
        with self.db.transaction():
            self.db.insert("t", {"a": 1, "b": "x"})
            self.assertEqual(self._rows("SELECT COUNT(*) FROM t"), [(0,)])  # not yet
        self.assertEqual(self._rows("SELECT COUNT(*) FROM t"), [(1,)])

    def test_writes_outside_transaction_still_commit(self):
        self.db.insert("t", {"a": 1, "b": "x"})
        self.assertEqual(self._rows("SELECT COUNT(*) FROM t"), [(1,)])

    def test_schema_changes_are_rolled_back_too(self):
        # sqlite3 only auto-BEGINs before DML; without an explicit BEGIN a
        # CREATE TABLE at the top of the block would be committed immediately.
        with self.assertRaises(RuntimeError):
            with self.db.transaction():
                self.db.create_table("u", {"a": "INTEGER"})
                self.db.insert("u", {"a": 1})
                raise RuntimeError("boom")
        names = self._rows("SELECT name FROM sqlite_master WHERE type='table'")
        self.assertNotIn(("u",), names)

    def test_caught_nested_failure_does_not_survive(self):
        with self.db.transaction():
            self.db.insert("t", {"a": 1, "b": "x"})
            try:
                with self.db.transaction():  # savepoint
                    self.db.insert("t", {"a": 2, "b": "y"})
                    raise RuntimeError("inner")
            except RuntimeError:
                pass
            self.db.insert("t", {"a": 3, "b": "z"})
        self.assertEqual(self._rows("SELECT a FROM t ORDER BY a"), [(1,), (3,)])

    def test_nested_success_commits_with_outer(self):
        with self.db.transaction():
            with self.db.transaction():
                self.db.insert("t", {"a": 1, "b": "x"})
            self.assertEqual(self._rows("SELECT COUNT(*) FROM t"), [(0,)])
        self.assertEqual(self._rows("SELECT COUNT(*) FROM t"), [(1,)])


# --------------------------------------------------------------- 4. colour
@unittest.skipUnless(cv2 is not None, "OpenCV not installed")
class PngColourOrder(unittest.TestCase):
    def test_rgb_png_round_trips_through_pillow(self):
        from PIL import Image

        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        rgb[..., 0], rgb[..., 1], rgb[..., 2] = 200, 30, 10
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "img.png")
            save_image_transform(rgb, directory=path)
            back = np.asarray(Image.open(path).convert("RGB"))
        np.testing.assert_array_equal(back, rgb)

    def test_grayscale_png_unchanged(self):
        from PIL import Image

        gray = np.arange(20, dtype=np.uint8).reshape(4, 5)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "img.png")
            save_image_transform(gray, directory=path)
            back = np.asarray(Image.open(path))
        np.testing.assert_array_equal(back, gray)


# ---------------------------------------------------------------- 5. hooks
class HooksInFrameworkConversion(unittest.TestCase):
    def test_torch_dataset_matches_python_iteration(self):
        p = PyTorchPipeline(
            _ListProvider([10, 20, 30]),
            [_double],
            pre_transform_hook=_pre_hook,
            post_transform_hook=_post_hook,
        )
        ds = p.to_framework_dataset()
        self.assertEqual([ds[i] for i in range(len(ds))], list(p))
        self.assertEqual(ds[1], ((20 + 1) * 2, 1))

    def test_tensorflow_conversion_refuses_hooks(self):
        p = TensorFlowPipeline(
            _ListProvider([1]), [_double], pre_transform_hook=_pre_hook
        )
        with self.assertRaises(NotImplementedError):
            p.to_framework_dataset()


# ----------------------------------------------------------- 6. picklable
class PicklableDataLoaderHelpers(unittest.TestCase):
    def test_datasets_and_worker_init_fn_pickle(self):
        p = PyTorchPipeline(
            _ListProvider([1, 2, 3]), [_double], pre_transform_hook=_pre_hook
        )
        ds = p.to_framework_dataset()
        self.assertIsInstance(ds, PyTorchTransformDataset)
        clone = pickle.loads(pickle.dumps(ds))
        self.assertEqual([clone[i] for i in range(len(clone))], list(p))
        self.assertEqual(
            pickle.loads(pickle.dumps(_MemoryListDataset([1, 2, 3])))[2], 3
        )
        self.assertIs(pickle.loads(pickle.dumps(_seed_worker)), _seed_worker)


# ------------------------------------------------------------ 7. averages
class SampleWeightedEpochAverage(unittest.TestCase):
    def test_num_samples_inference(self):
        self.assertEqual(_num_samples((np.zeros((3, 2)), np.zeros(3))), 3)
        self.assertEqual(_num_samples((["a.png", "b.png"], np.zeros(2))), 2)
        self.assertEqual(_num_samples(1.5), 1)  # unknown -> plain mean of batch means

    @unittest.skipUnless(torch is not None, "torch not installed")
    def test_uneven_last_batch_is_weighted_by_its_size(self):
        from torch.utils.data import DataLoader, TensorDataset

        from xflow.trainers.trainer import TorchTrainer

        class _Trainer(TorchTrainer):
            def train_step(self, batch):
                x, _ = batch
                return {"loss": float(x.shape[0])}  # batch mean := batch size

            def val_step(self, batch):
                x, _ = batch
                return {"val_loss": float(x.shape[0])}

        loader = DataLoader(
            TensorDataset(torch.zeros(5, 1), torch.zeros(5, 1)), batch_size=2
        )
        model = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as d:
            trainer = _Trainer(
                model=model,
                data_pipeline=loader,
                output_dir=d,
                optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                criterion=torch.nn.MSELoss(),
                device=torch.device("cpu"),
            )
            history = trainer.fit(epochs=1, train_loader=loader, val_loader=loader)
        # batches of 2, 2, 1 -> (2*2 + 2*2 + 1*1) / 5 = 1.8, not (2 + 2 + 1) / 3
        self.assertAlmostEqual(history["train_loss"][0], 1.8)
        self.assertAlmostEqual(history["val_loss"][0], 1.8)


if __name__ == "__main__":
    unittest.main()
