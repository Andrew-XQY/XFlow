"""Accelerator physics-specific pipeline utilities for specialized data processing workflows."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

from ...data.pipeline import BasePipeline, Transform


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

    def to_framework_dataset(
        self,
        framework: str = "pytorch",
        dataset_ops: Optional[List[Dict]] = None,
        **kwargs,
    ):
        """Convert to framework-specific dataset."""
        self._load_basis()

        if framework.lower() in ("pytorch", "torch"):
            import torch
            from torch.utils.data import Dataset

            from ...data.transform import apply_dataset_operations_from_config

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

            dataset = _CombinatorDataset(self)
            if dataset_ops:
                dataset = apply_dataset_operations_from_config(dataset, dataset_ops)
            return dataset
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


class PatternDecompositionCombinator(Combinator):
    """
    Decomposes a target pattern into basis coefficients and reconstructs.

    For CachedBasisPipeline with (left, right) image pairs:
      - Uses RIGHT images from basis for decomposition
      - Applies coefficients to BOTH left and right for output

    Args:
        pattern_provider: Callable that returns a pattern (2D np.ndarray) each call.
                          Signature: (rng: np.random.Generator) -> np.ndarray
        decomposition_method: How to compute coefficients:
                              - 'projection': Dot product (fastest, assumes orthonormal basis)
                              - 'nnls': Non-negative least squares
                              - 'lstsq': Standard least squares
                              - 'omp': Orthogonal Matching Pursuit (sparse)
        max_basis_items: Max number of basis items to use (None = all).
        regularization: Regularization strength for 'lstsq' method.
        normalize_pattern: Whether to normalize pattern before decomposition.
        clip_coefficients: Clip coefficients to this range (None = no clipping).
        clip_output: Clip output to this range.
    """

    def __init__(
        self,
        pattern_provider,
        decomposition_method: str = "projection",
        max_basis_items: Optional[int] = None,
        regularization: float = 0.0,
        normalize_pattern: bool = False,
        clip_coefficients: Optional[Tuple[float, float]] = None,
        clip_output: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__()
        # Allow pattern_provider to be either a callable or a generator object
        if hasattr(pattern_provider, "__next__"):
            self._pattern_stream = pattern_provider
            self.pattern_provider = lambda rng: next(self._pattern_stream)
        else:
            self.pattern_provider = pattern_provider
        self.decomposition_method = decomposition_method
        self.max_basis_items = max_basis_items
        self.regularization = regularization
        self.normalize_pattern = normalize_pattern
        self.clip_coefficients = clip_coefficients
        self.clip_output = clip_output

        # Cache for basis matrix (built on first call)
        self._basis_matrix: Optional[np.ndarray] = None
        self._basis_shape: Optional[Tuple[int, ...]] = None

    def _build_basis_matrix(self, accessor: BasisAccessor) -> np.ndarray:
        """
        Build matrix where each column is a flattened RIGHT image from basis.
        Shape: (num_pixels, num_basis_items)
        """
        n = len(accessor)
        if self.max_basis_items:
            n = min(n, self.max_basis_items)

        # Get first sample to determine shape
        sample = accessor[0]
        if not isinstance(sample, tuple) or len(sample) < 2:
            raise ValueError(
                "PatternDecompositionCombinator expects (left, right) tuple basis items"
            )

        right_img = sample[1]  # Use RIGHT image
        right_img = np.squeeze(right_img)
        self._basis_shape = right_img.shape
        num_pixels = right_img.size

        # Build matrix: each column is a flattened basis right image
        basis_matrix = np.zeros((num_pixels, n), dtype=np.float32)
        for i in range(n):
            right = accessor[i][1]  # RIGHT image
            right = np.squeeze(right)
            basis_matrix[:, i] = right.flatten().astype(np.float32)

        return basis_matrix

    def _decompose(
        self, pattern: np.ndarray, basis_matrix: np.ndarray
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Decompose pattern into basis coefficients.

        Returns:
            coefficients: Array of coefficients
            indices: Indices of basis items used
        """
        # Flatten and optionally normalize pattern
        target = pattern.flatten().astype(np.float32)
        if self.normalize_pattern:
            norm = np.linalg.norm(target)
            if norm > 1e-8:
                target = target / norm

        n_basis = basis_matrix.shape[1]
        indices = list(range(n_basis))

        if self.decomposition_method == "projection":
            # Dot product projection (assumes orthonormal basis)
            # coefficient_i = <pattern, basis_i>
            coefficients = np.dot(basis_matrix.T, target)

        elif self.decomposition_method == "nnls":
            # Non-negative least squares
            from scipy.optimize import nnls

            coefficients, _ = nnls(basis_matrix, target)

        elif self.decomposition_method == "lstsq":
            # Standard least squares (can have negative coeffs)
            if self.regularization > 0:
                # Ridge regression
                A = basis_matrix.T @ basis_matrix
                A += self.regularization * np.eye(n_basis)
                b = basis_matrix.T @ target
                coefficients = np.linalg.solve(A, b)
            else:
                coefficients, _, _, _ = np.linalg.lstsq(
                    basis_matrix, target, rcond=None
                )

        elif self.decomposition_method == "omp":
            # Orthogonal Matching Pursuit (sparse)
            from sklearn.linear_model import OrthogonalMatchingPursuit

            n_nonzero = self.max_basis_items or max(1, n_basis // 4)
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
            omp.fit(basis_matrix, target)
            coefficients = omp.coef_

        else:
            raise ValueError(
                f"Unknown decomposition method: {self.decomposition_method}"
            )

        # Clip coefficients if specified
        if self.clip_coefficients:
            lo, hi = self.clip_coefficients
            coefficients = np.clip(coefficients, lo, hi)

        return coefficients, indices

    def __call__(self, accessor: BasisAccessor, rng: np.random.Generator) -> Any:
        # Build basis matrix on first call (or if accessor changed)
        if self._basis_matrix is None:
            self._basis_matrix = self._build_basis_matrix(accessor)

        # Get pattern from provider
        pattern = self.pattern_provider(rng)
        pattern = np.squeeze(pattern)

        # Resize pattern if needed to match basis shape
        if pattern.shape != self._basis_shape:
            from scipy.ndimage import zoom

            zoom_factors = tuple(
                b / p for b, p in zip(self._basis_shape, pattern.shape)
            )
            pattern = zoom(pattern, zoom_factors, order=1)

        # Decompose pattern into coefficients
        coefficients, indices = self._decompose(pattern, self._basis_matrix)

        # Build output by linear combination
        sample = accessor[0]
        left_result = np.zeros_like(sample[0], dtype=np.float32)
        right_result = np.zeros_like(sample[1], dtype=np.float32)

        used_indices = []
        used_coefficients = []

        for idx, coef in zip(indices, coefficients):
            if abs(coef) > 1e-8:  # Skip near-zero coefficients
                left, right = accessor[idx]
                left_result += coef * left.astype(np.float32)
                right_result += coef * right.astype(np.float32)
                used_indices.append(idx)
                used_coefficients.append(float(coef))

        # Clip output
        lo, hi = self.clip_output
        left_result = np.clip(left_result, lo, hi)
        right_result = np.clip(right_result, lo, hi)

        self._last_record = CombinationRecord(
            indices=used_indices, coefficients=used_coefficients
        )

        return (left_result, right_result)

    def reset_cache(self):
        """Reset cached basis matrix (call if basis changes)."""
        self._basis_matrix = None
        self._basis_shape = None


# SlidingBlockPixelateCombinator
def _squeeze_2d(x: Any) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"Expected 2D pattern, got shape {x.shape}")
    return x


class IndexCombinator(Combinator):
    """
    Weighted sum of basis items using coefficients from a 2D map.

    output = sum(coeff[i] * basis[i])

    Coefficients are read in row-major (flattened) order.
    """

    def __init__(
        self,
        pattern_provider,
        *,
        skip_zero: bool = True,
        eps: float = 0.0,
        post_transforms: Optional[
            List[Union[Callable[[np.ndarray], np.ndarray], Transform]]
        ] = None,
    ):
        super().__init__()
        if hasattr(pattern_provider, "__next__"):
            self._pattern_stream = pattern_provider
            self.pattern_provider = lambda rng: next(self._pattern_stream)
        else:
            self.pattern_provider = pattern_provider

        self.skip_zero = skip_zero
        self.eps = eps
        self.post_transforms = [
            (
                fn
                if isinstance(fn, Transform)
                else Transform(fn, getattr(fn, "__name__", "unknown"))
            )
            for fn in (post_transforms or [])
        ]
        self.last_coeff_map: Optional[np.ndarray] = None

    def __call__(self, accessor: BasisAccessor, rng: np.random.Generator) -> Any:
        # Get coefficient map and flatten to 1D
        coeff_map = _squeeze_2d(self.pattern_provider(rng)).astype(
            np.float32, copy=False
        )
        self.last_coeff_map = coeff_map
        coeffs = coeff_map.ravel()

        if len(accessor) < coeffs.size:
            raise ValueError(
                f"Need {coeffs.size} basis items for map shape {coeff_map.shape}, "
                f"but only {len(accessor)} available."
            )

        # Figure out which coefficients are active
        if self.skip_zero:
            mask = np.abs(coeffs) > self.eps if self.eps > 0 else coeffs != 0
            active = np.flatnonzero(mask)
        else:
            active = np.arange(coeffs.size)

        sample0 = accessor[0]
        is_multi = isinstance(sample0, (tuple, list))

        used_idx = []
        used_coef = []

        if is_multi:
            # Basis items are tuples/lists with a fixed number of elements.
            # Scale each element by the same coefficient, then sum element-wise.
            n_elements = len(sample0)
            out = [
                np.zeros_like(np.asarray(sample0[j]), dtype=np.float32)
                for j in range(n_elements)
            ]

            for i in active:
                c = float(coeffs[i])
                item = accessor[int(i)]
                for j in range(n_elements):
                    out[j] += c * np.asarray(item[j], dtype=np.float32)
                used_idx.append(int(i))
                used_coef.append(c)
            out = tuple(out)
        else:
            # Single array basis items
            out = np.zeros_like(np.asarray(sample0), dtype=np.float32)

            for i in active:
                c = float(coeffs[i])
                out += c * np.asarray(accessor[int(i)], dtype=np.float32)
                used_idx.append(int(i))
                used_coef.append(c)

        for fn in self.post_transforms:
            out = fn(out)

        if is_multi:
            if not isinstance(out, (tuple, list)):
                raise ValueError(
                    "post_transforms must return tuple/list for multi-component samples"
                )
            out = tuple(np.asarray(component, dtype=np.float32) for component in out)
        else:
            out = np.asarray(out, dtype=np.float32)

        self._last_record = CombinationRecord(indices=used_idx, coefficients=used_coef)
        return out


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
