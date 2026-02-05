"""Accelerator physics-specific pipeline utilities for specialized data processing workflows."""

from typing import Any, Callable, Iterator, List, Optional, Tuple

import numpy as np

from ...data.pipeline import BasePipeline


# samplers
def uniform_k(min_k: int = 1, max_k: int = 4):
    """Uniform over [min_k, max_k]."""

    def _sampler(rng: np.random.Generator, n: int) -> int:
        hi = min(max_k, n)
        lo = min(min_k, hi)
        return int(rng.integers(lo, hi + 1))

    return _sampler


def poisson_k(lam: float = 2.0, min_k: int = 1, max_k: int | None = None):
    """Poisson(λ) clipped to [min_k, max_k or n]."""

    def _sampler(rng: np.random.Generator, n: int) -> int:
        k = int(rng.poisson(lam))
        if max_k is None:
            hi = n
        else:
            hi = min(max_k, n)
        k = max(min_k, min(k, hi))
        return k

    return _sampler


# Physics-specific pipeline implementations will extend BasePipeline:
# class PhysicsPipeline(BasePipeline):


class StackMixPipeline(BasePipeline):
    """
    Synthetic stacking pipeline:
      - For each output sample: draw k ~ sampler(n_items), sample k distinct items,
        extract image arrays, sum, clip, yield.
      - Yields *single* stacked images; use your existing `.batch(batch_size)` for batching.

    Args:
        data_provider: yields raw items (np arrays, or rows/records)
        sampler: Callable[[np.random.Generator, int], int] -> k
                 (given RNG and dataset size, returns k ∈ [1, n])
        extract_fn: maps a raw item -> np.ndarray image (H,W[,C]); default: identity
        seed: RNG seed for reproducibility; default: None
        dtype: force output dtype; default: inferred from first sample
        max_outputs_per_epoch: how many stacked samples to emit per epoch (default: len(dataset))
                               (An “epoch” here is just one full pass of len(dataset) outputs.)
    """

    def __init__(
        self,
        data_provider,
        sampler: Callable[[np.random.Generator, int], int],
        extract_fn: Optional[Callable[[Any], np.ndarray]] = None,
        *,
        seed: Optional[int] = None,
        dtype: Optional[np.dtype] = None,
        max_outputs_per_epoch: Optional[int] = None,
        **base_kwargs,
    ):
        # no transforms here; keep it minimal and fast
        super().__init__(data_provider, transforms=None, **base_kwargs)
        self.sampler = sampler
        self.extract_fn = extract_fn or (lambda x: x)
        self.rng = np.random.default_rng(seed)
        self._cache = None  # lazy materialization of images
        self._dtype = dtype
        self._max_out = max_outputs_per_epoch

    def _load_cache(self):
        if self._cache is not None:
            return
        # materialize all items and extract images once
        imgs = []
        for raw in self.data_provider():
            arr = self.extract_fn(raw)
            if not isinstance(arr, np.ndarray):
                arr = np.asarray(arr)
            imgs.append(arr)
        if not imgs:
            raise ValueError("StackMixPipeline: empty dataset from data_provider().")
        # basic shape check
        first_shape = imgs[0].shape
        for i, a in enumerate(imgs):
            if a.shape != first_shape:
                raise ValueError(
                    f"All images must share the same shape. "
                    f"Got {a.shape} at index {i}, expected {first_shape}."
                )
        # remember dtype & clip max
        if self._dtype is None:
            self._dtype = imgs[0].dtype
        # decide clip bounds from dtype
        if np.issubdtype(self._dtype, np.integer):
            self._clip_min, self._clip_max = (
                0,
                np.iinfo(self._dtype).max,
            )  # e.g., 255 for uint8
        else:
            # assume normalized floats
            self._clip_min, self._clip_max = 0.0, 1.0
        self._cache = imgs

    def __len__(self) -> int:
        # define an epoch as len(dataset) synthetic samples unless overridden
        self._load_cache()
        return self._max_out or len(self._cache)

    def __iter__(self) -> Iterator[np.ndarray]:
        self._load_cache()
        n = len(self._cache)
        out_count = self._max_out or n

        for _ in range(out_count):
            # 1) draw k
            k = int(self.sampler(self.rng, n))
            if k < 1:
                k = 1
            if k > n:
                k = n
            # 2) choose k distinct indices
            idx = self.rng.choice(n, size=k, replace=False)
            # 3) sum and clip
            s = None
            for i in idx:
                a = self._cache[i].astype(np.float32, copy=False)
                s = a if s is None else (s + a)
            s = np.clip(s, self._clip_min, self._clip_max)

            # 4) cast to target dtype (if integer, round)
            if np.issubdtype(self._dtype, np.integer):
                yield s.round().astype(self._dtype, copy=False)
            else:
                yield s.astype(self._dtype, copy=False)


class LinearBasisPipeline(BasePipeline):
    """
    Generates samples as linear combinations of basis items (shared coefficients across components).

    If basis yields (x, y, z) tuples, output is (Σc_i*x_i, Σc_i*y_i, Σc_i*z_i).
    If basis yields single arrays, output is Σc_i*arr_i.

    Args:
        data_provider: Yields basis items (array or tuple of arrays).
        combo_generator: (rng, n) -> [(idx, coef), ...] defines the linear combination.
        extract_fn: Raw item -> array or tuple; default: identity.
        seed: RNG seed.
        num_samples: Samples per epoch; default: len(basis).
    """

    def __init__(
        self,
        data_provider,
        combo_generator: Callable[[np.random.Generator, int], List[Tuple[int, float]]],
        extract_fn: Optional[Callable[[Any], Any]] = None,
        *,
        seed: Optional[int] = None,
        num_samples: Optional[int] = None,
        clip_range: Tuple[float, float] = (0.0, 1.0),
        **base_kwargs,
    ):
        super().__init__(data_provider, transforms=None, **base_kwargs)
        self.combo_generator = combo_generator
        self.extract_fn = extract_fn or (lambda x: x)
        self.rng = np.random.default_rng(seed)
        self._basis: Optional[List[Any]] = None
        self._is_tuple: bool = False
        self._num_samples = num_samples
        self._clip_range = clip_range

    def _load_basis(self):
        if self._basis is not None:
            return
        self._basis = [self.extract_fn(raw) for raw in self.data_provider()]
        if not self._basis:
            raise ValueError("Empty basis set")
        self._is_tuple = isinstance(self._basis[0], (tuple, list))

    def _apply_combo(self, terms: List[Tuple[int, float]]) -> Any:
        """Apply same linear combination to all components."""
        lo, hi = self._clip_range

        if self._is_tuple:
            n_comp = len(self._basis[0])
            result = [None] * n_comp
            for idx, coef in terms:
                for c in range(n_comp):
                    arr = self._basis[idx][c].astype(np.float32) * coef
                    result[c] = arr if result[c] is None else (result[c] + arr)
            return tuple(np.clip(r, lo, hi) for r in result)
        else:
            result = None
            for idx, coef in terms:
                arr = self._basis[idx].astype(np.float32) * coef
                result = arr if result is None else (result + arr)
            return np.clip(result, lo, hi)

    def __len__(self) -> int:
        self._load_basis()
        return self._num_samples or len(self._basis)

    def __iter__(self) -> Iterator[Any]:
        self._load_basis()
        for _ in range(len(self)):
            terms = self.combo_generator(self.rng, len(self._basis))
            yield self._apply_combo(terms)

    def to_framework_dataset(self) -> Any:
        raise NotImplementedError("Use iteration or .to_numpy()")


# Combo generators
def random_sum(k_sampler=None, coef_sampler=None):
    """Random sum with optional coefficient scaling."""
    k_sampler = k_sampler or uniform_k(1, 3)
    coef_sampler = coef_sampler or (lambda rng: 1.0)

    def _combo(rng: np.random.Generator, n: int) -> List[Tuple[int, float]]:
        k = k_sampler(rng, n)
        indices = rng.choice(n, size=k, replace=False)
        return [(int(i), coef_sampler(rng)) for i in indices]

    return _combo


def identity_combo():
    """Single basis item with coef=1 (pass-through real samples)."""

    def _combo(rng: np.random.Generator, n: int) -> List[Tuple[int, float]]:
        return [(int(rng.integers(0, n)), 1.0)]

    return _combo
