"""Accelerator physics-specific pipeline utilities for specialized data processing workflows."""
import numpy as np
from typing import Callable, Any, Optional, Iterator
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
        self._cache = None           # lazy materialization of images
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
                raise ValueError(f"All images must share the same shape. "
                                 f"Got {a.shape} at index {i}, expected {first_shape}.")
        # remember dtype & clip max
        if self._dtype is None:
            self._dtype = imgs[0].dtype
        # decide clip bounds from dtype
        if np.issubdtype(self._dtype, np.integer):
            self._clip_min, self._clip_max = 0, np.iinfo(self._dtype).max  # e.g., 255 for uint8
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
                

