"""Accelerator physics-specific pipeline utilities for specialized data processing workflows."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from ...data.pipeline import BasePipeline


def _to_numpy(arr) -> np.ndarray:
    """Convert tensor or array to numpy."""
    if hasattr(arr, "numpy"):
        return arr.detach().cpu().numpy()
    if hasattr(arr, "np"):
        return arr.numpy()
    return np.asarray(arr)


# ============================================================
# Contracts
# ============================================================


@dataclass
class BasisMetadata:
    """Metadata about the cached basis set."""

    ids: List[Any]  # Original IDs from data_provider
    count: int
    shapes: Dict[str, Tuple[int, ...]] = field(
        default_factory=dict
    )  # e.g., {"image": (H,W), "label": (N,)}
    extra: Dict[str, Any] = field(default_factory=dict)  # User-defined metadata


@dataclass
class CombinationRecord:
    """Records which basis items were combined (for debugging)."""

    indices: List[int]  # Basis indices used
    coefficients: List[float]  # Coefficients applied


class BasisAccessor:
    """Interface for combinators to request basis items by index."""

    def __init__(self, basis: List[Any], ids: List[Any]):
        self._basis = basis
        self._ids = ids

    def __getitem__(self, idx: int) -> Any:
        """Get basis item by index."""
        return self._basis[idx]

    def get_by_id(self, item_id: Any) -> Any:
        """Get basis item by original ID."""
        idx = self._ids.index(item_id)
        return self._basis[idx]

    def __len__(self) -> int:
        return len(self._basis)


class Combinator(ABC):
    """
    Abstract base for defining how to combine basis items into output samples.

    Receives:
        - accessor: BasisAccessor to fetch basis items by index or ID
        - rng: numpy random generator

    After each __call__, `last_record` holds the indices/coefficients used.
    """

    def __init__(self):
        self._last_record: Optional[CombinationRecord] = None

    @property
    def last_record(self) -> Optional[CombinationRecord]:
        """Get the last combination's indices and coefficients."""
        return self._last_record

    @abstractmethod
    def __call__(self, accessor: BasisAccessor, rng: np.random.Generator) -> Any:
        """Produce one output sample from the basis."""
        pass


# ============================================================
# Pipeline
# ============================================================


class CachedBasisPipeline(BasePipeline):
    """
    Pipeline that:
      1. Loads items from data_provider (each item has an ID)
      2. Applies transforms and caches results as basis
      3. Uses a Combinator to yield output samples

    Args:
        data_provider: Yields (id, raw_item) tuples or just raw_items (auto-indexed).
        combinator: Combinator instance defining how to produce output samples.
        transforms: Applied to each raw item during caching.
        seed: RNG seed.
        num_samples: Samples per epoch; default: len(basis).
        id_extractor: Optional function to extract ID from raw item.
                      If None and data_provider yields tuples, first element is ID.
        eager: If True, load basis immediately during __init__.
    """

    def __init__(
        self,
        data_provider,
        combinator: Combinator,
        *,
        transforms: Optional[List[Callable]] = None,
        seed: Optional[int] = None,
        num_samples: Optional[int] = None,
        id_extractor: Optional[Callable[[Any], Any]] = None,
        eager: bool = False,
        **base_kwargs,
    ):
        super().__init__(data_provider, transforms=transforms, **base_kwargs)
        self.combinator = combinator
        self.rng = np.random.default_rng(seed)
        self._num_samples = num_samples
        self._id_extractor = id_extractor

        # Cached state
        self._basis: Optional[List[Any]] = None
        self._ids: Optional[List[Any]] = None
        self._metadata: Optional[BasisMetadata] = None

        if eager:
            self._load_basis()

    def _load_basis(self):
        """Load, transform, and cache all basis items."""
        if self._basis is not None:
            return

        basis = []
        ids = []
        raw_items = list(self.data_provider())

        pbar = tqdm(raw_items, desc="Caching basis", leave=False)
        for i, raw in enumerate(pbar):
            # Extract ID
            if self._id_extractor:
                item_id = self._id_extractor(raw)
                item = raw
            elif isinstance(raw, tuple) and len(raw) == 2:
                item_id, item = raw
            else:
                item_id = i
                item = raw

            try:
                # Apply transforms
                for fn in self.transforms:
                    item = fn(item)

                if item is not None:
                    item = (
                        _to_numpy(item)
                        if not isinstance(item, tuple)
                        else tuple(_to_numpy(x) for x in item)
                    )
                    basis.append(item)
                    ids.append(item_id)
                else:
                    self.error_count += 1
            except Exception as e:
                self.error_count += 1
                self.logger.warning(f"Transform failed for ID={item_id}: {e}")
                if not self.skip_errors:
                    raise
        pbar.close()

        if not basis:
            raise ValueError("Empty basis after transforms.")

        self._basis = basis
        self._ids = ids
        self._metadata = BasisMetadata(
            ids=ids,
            count=len(basis),
            shapes=self._infer_shapes(basis[0]),
        )
        self.in_memory_sample_count = len(basis)

    def _infer_shapes(self, sample: Any) -> Dict[str, Tuple[int, ...]]:
        """Infer shapes from a sample."""
        if isinstance(sample, tuple):
            return {f"component_{i}": s.shape for i, s in enumerate(sample)}
        return {"data": sample.shape}

    @property
    def metadata(self) -> BasisMetadata:
        """Access metadata (loads basis if needed)."""
        self._load_basis()
        return self._metadata

    @property
    def accessor(self) -> BasisAccessor:
        """Access basis items (loads basis if needed)."""
        self._load_basis()
        return BasisAccessor(self._basis, self._ids)

    @property
    def basis(self) -> List[Any]:
        """Direct read-only access to cached basis list."""
        self._load_basis()
        return self._basis

    def get_basis(self, idx: int) -> Any:
        """Get cached basis item by index."""
        self._load_basis()
        return self._basis[idx]

    def get_basis_by_id(self, item_id: Any) -> Any:
        """Get cached basis item by original ID."""
        self._load_basis()
        idx = self._ids.index(item_id)
        return self._basis[idx]

    def __len__(self) -> int:
        self._load_basis()
        return self._num_samples or len(self._basis)

    def __iter__(self) -> Iterator[Any]:
        self._load_basis()
        accessor = BasisAccessor(self._basis, self._ids)
        for _ in range(len(self)):
            yield self.combinator(accessor, self.rng)

    def to_framework_dataset(self, framework: str = "pytorch", **kwargs):
        """Convert to framework-specific dataset."""
        self._load_basis()

        if framework.lower() == "pytorch":
            import torch
            from torch.utils.data import Dataset

            class _CombinatorDataset(Dataset):
                def __init__(inner_self, pipeline: CachedBasisPipeline):
                    inner_self.pipeline = pipeline
                    inner_self.accessor = pipeline.accessor

                def __len__(inner_self):
                    return len(inner_self.pipeline)

                def __getitem__(inner_self, idx):
                    sample = inner_self.pipeline.combinator(
                        inner_self.accessor, inner_self.pipeline.rng
                    )
                    # Convert to tensors
                    if isinstance(sample, tuple):
                        return tuple(torch.from_numpy(s) for s in sample)
                    return torch.from_numpy(sample)

            return _CombinatorDataset(self)
        else:
            raise ValueError(f"Unsupported framework: {framework}")


# ============================================================
# Built-in Combinators
# ============================================================


class LinearCombinator(Combinator):
    """
    Linear combination: Σ c_i * basis[i]

    Args:
        k_sampler: (rng, n) -> number of items to combine
        coef_sampler: (rng) -> coefficient for each item
        clip_range: Output clipping bounds
    """

    def __init__(
        self,
        k_sampler: Callable[[np.random.Generator, int], int] = None,
        coef_sampler: Callable[[np.random.Generator], float] = None,
        clip_range: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__()
        self.k_sampler = k_sampler or (lambda rng, n: min(2, n))
        self.coef_sampler = coef_sampler or (lambda rng: 1.0)
        self.clip_range = clip_range

    def __call__(self, accessor: BasisAccessor, rng: np.random.Generator) -> Any:
        n = len(accessor)
        k = self.k_sampler(rng, n)
        indices = rng.choice(n, size=k, replace=False).tolist()
        coefficients = []

        lo, hi = self.clip_range
        sample = accessor[0]
        is_tuple = isinstance(sample, tuple)

        if is_tuple:
            result = [None] * len(sample)
            for idx in indices:
                coef = self.coef_sampler(rng)
                coefficients.append(coef)
                for c, arr in enumerate(accessor[idx]):
                    scaled = arr.astype(np.float32) * coef
                    result[c] = scaled if result[c] is None else (result[c] + scaled)
            output = tuple(np.clip(r, lo, hi) for r in result)
        else:
            result = None
            for idx in indices:
                coef = self.coef_sampler(rng)
                coefficients.append(coef)
                scaled = accessor[idx].astype(np.float32) * coef
                result = scaled if result is None else (result + scaled)
            output = np.clip(result, lo, hi)

        self._last_record = CombinationRecord(
            indices=indices, coefficients=coefficients
        )
        return output


class IdentityCombinator(Combinator):
    """Pass through single basis items unchanged."""

    def __call__(self, accessor: BasisAccessor, rng: np.random.Generator) -> Any:
        idx = int(rng.integers(0, len(accessor)))
        self._last_record = CombinationRecord(indices=[idx], coefficients=[1.0])
        return accessor[idx]


# ============================================================
# K-samplers (reusable)
# ============================================================


def uniform_k(min_k: int = 1, max_k: int = 4):
    """Uniform over [min_k, max_k]."""

    def _sampler(rng: np.random.Generator, n: int) -> int:
        hi = min(max_k, n)
        lo = min(min_k, hi)
        return int(rng.integers(lo, hi + 1))

    return _sampler


def poisson_k(lam: float = 2.0, min_k: int = 1, max_k: int = None):
    """Poisson(λ) clipped to [min_k, max_k or n]."""

    def _sampler(rng: np.random.Generator, n: int) -> int:
        k = int(rng.poisson(lam))
        hi = n if max_k is None else min(max_k, n)
        return max(min_k, min(k, hi))

    return _sampler
