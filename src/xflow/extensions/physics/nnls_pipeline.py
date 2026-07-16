"""NNLS coefficient-map combinator for cached accelerator-physics bases."""

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

from ...data.pipeline import Transform
from .pipeline import (
    BasisAccessor,
    CombinationRecord,
    Combinator,
    IntensityScaleSpec,
    _resolve_intensity_scale,
    _squeeze_2d,
)


def solve_regularized_nnls(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    regularization: float = 0.0,
    smoothness_operator: Optional[np.ndarray] = None,
    maxiter: Optional[int] = None,
) -> np.ndarray:
    """Solve ``min(a >= 0) ||A a - x||^2 + lambda ||L a||^2``.

    If regularization is positive and no operator is supplied, ``L`` is the
    first-difference operator in coefficient order.
    """
    from scipy.optimize import nnls

    A = np.asarray(matrix, dtype=np.float64)
    x = np.asarray(target, dtype=np.float64).reshape(-1)
    lam = float(regularization)

    if A.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {A.shape}.")
    if x.size != A.shape[0]:
        raise ValueError(
            f"target length must match matrix rows: got {x.size} vs {A.shape[0]}."
        )
    if A.shape[1] == 0:
        raise ValueError("matrix must contain at least one coefficient column.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(x)):
        raise ValueError("matrix and target must contain only finite values.")
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError(f"regularization must be finite and >= 0, got {lam}.")

    rhs = x
    solve_matrix = A
    if lam > 0.0:
        if smoothness_operator is None:
            L = np.diff(np.eye(A.shape[1], dtype=np.float64), axis=0)
        else:
            L = np.asarray(smoothness_operator, dtype=np.float64)
        if L.ndim != 2 or L.shape[1] != A.shape[1]:
            raise ValueError(
                "smoothness_operator must have one column per coefficient: "
                f"got {L.shape}, expected (*, {A.shape[1]})."
            )
        if not np.all(np.isfinite(L)):
            raise ValueError("smoothness_operator must contain only finite values.")
        if L.shape[0] > 0:
            solve_matrix = np.vstack((A, np.sqrt(lam) * L))
            rhs = np.concatenate((x, np.zeros(L.shape[0], dtype=np.float64)))

    kwargs = {} if maxiter is None else {"maxiter": int(maxiter)}
    coefficients, _ = nnls(solve_matrix, rhs, **kwargs)
    return coefficients.astype(np.float32, copy=False)


class NNLSCoefficientMapCombinator(Combinator):
    """Fit a coefficient map to one basis component, then combine all components.

    The pattern provider supplies ``x``. Columns of ``A`` are the cached basis
    images selected by ``solve_component``, resized to the pattern shape. The
    fitted non-negative coefficients are applied unchanged to every component
    of each cached basis item, preserving paired input/target conventions.

    ``CachedBasisPipeline`` may inject normalized basis positions through
    ``set_basis_positions``. When regularization is enabled, those positions
    define a nearest-neighbor difference operator; without positions, the
    solver's coefficient-order first differences are used.
    """

    def __init__(
        self,
        pattern_provider,
        *,
        solve_component: int = 1,
        regularization: float = 0.0,
        smoothness_neighbors: int = 4,
        coefficient_threshold: float = 1e-8,
        max_coefficient: Optional[float] = None,
        maxiter: Optional[int] = None,
        intensity_scale: IntensityScaleSpec = 1.0,
        clip_output: Tuple[float, float] = (0.0, 1.0),
        transforms: Optional[
            List[Union[Callable[[np.ndarray], np.ndarray], Transform]]
        ] = None,
    ):
        super().__init__()
        if hasattr(pattern_provider, "__next__"):
            self._pattern_stream = pattern_provider
            self.pattern_provider = lambda rng: next(self._pattern_stream)
        else:
            self.pattern_provider = pattern_provider

        if float(regularization) < 0.0:
            raise ValueError("regularization must be >= 0.")
        if int(smoothness_neighbors) < 1:
            raise ValueError("smoothness_neighbors must be >= 1.")
        if float(coefficient_threshold) < 0.0:
            raise ValueError("coefficient_threshold must be >= 0.")
        if max_coefficient is not None and float(max_coefficient) <= 0.0:
            raise ValueError("max_coefficient must be > 0 when provided.")

        self.solve_component = int(solve_component)
        self.regularization = float(regularization)
        self.smoothness_neighbors = int(smoothness_neighbors)
        self.coefficient_threshold = float(coefficient_threshold)
        self.max_coefficient = (
            None if max_coefficient is None else float(max_coefficient)
        )
        self.maxiter = None if maxiter is None else int(maxiter)
        self.intensity_scale = intensity_scale
        self.clip_output = clip_output
        self.transforms = [
            (
                fn
                if isinstance(fn, Transform)
                else Transform(fn, getattr(fn, "__name__", "unknown"))
            )
            for fn in (transforms or [])
        ]

        self.last_coeff_map: Optional[np.ndarray] = None
        self.last_coefficients: Optional[np.ndarray] = None
        self.last_intensity_scale: float = 1.0
        self._basis_positions: Optional[np.ndarray] = None
        self._smoothness_operator: Optional[np.ndarray] = None
        self._solve_matrix: Optional[np.ndarray] = None
        self._solve_shape: Optional[Tuple[int, int]] = None
        self._basis_matrix: Optional[np.ndarray] = None
        self._basis_is_multi = False
        self._basis_shapes: List[Tuple[int, ...]] = []
        self._basis_sizes: List[int] = []

    def set_basis_positions(self, basis_positions: np.ndarray) -> None:
        """Accept the position-injection API used by ``CachedBasisPipeline``."""
        positions = np.asarray(basis_positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(
                f"basis_positions must have shape (N, 2), got {positions.shape}."
            )
        if not np.all(np.isfinite(positions)):
            raise ValueError("basis_positions contains non-finite values.")
        if np.any(positions < 0.0) or np.any(positions > 1.0):
            raise ValueError("basis_positions must be normalized to [0, 1].")
        self._basis_positions = positions
        self._smoothness_operator = None

    @staticmethod
    def _resize_2d(image: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        arr = _squeeze_2d(image).astype(np.float32, copy=False)
        if arr.shape == shape:
            return arr
        from scipy.ndimage import zoom

        resized = zoom(
            arr,
            (shape[0] / arr.shape[0], shape[1] / arr.shape[1]),
            order=1,
        )
        if resized.shape != shape:
            raise RuntimeError(
                f"Failed to resize basis image from {arr.shape} to {shape}; "
                f"got {resized.shape}."
            )
        return resized.astype(np.float32, copy=False)

    def _build_basis_matrices(
        self, accessor: BasisAccessor, solve_shape: Tuple[int, int]
    ) -> None:
        sample0 = accessor[0]
        is_multi = isinstance(sample0, (tuple, list))
        components0 = list(sample0) if is_multi else [sample0]
        if not 0 <= self.solve_component < len(components0):
            raise ValueError(
                f"solve_component={self.solve_component} is invalid for "
                f"{len(components0)} basis component(s)."
            )

        shapes = [np.asarray(component).shape for component in components0]
        sizes = [int(np.prod(shape)) for shape in shapes]
        solve_matrix = np.empty(
            (int(np.prod(solve_shape)), len(accessor)), dtype=np.float32
        )
        basis_matrix = np.empty((len(accessor), int(sum(sizes))), dtype=np.float32)

        for index in range(len(accessor)):
            item = accessor[index]
            components = list(item) if is_multi else [item]
            if len(components) != len(components0):
                raise ValueError(
                    "All cached basis items must have matching components."
                )
            solve_matrix[:, index] = self._resize_2d(
                components[self.solve_component], solve_shape
            ).ravel()
            offset = 0
            for component, shape, size in zip(components, shapes, sizes):
                arr = np.asarray(component, dtype=np.float32)
                if arr.shape != shape:
                    raise ValueError("All cached basis component shapes must match.")
                basis_matrix[index, offset : offset + size] = arr.ravel()
                offset += size

        self._solve_matrix = solve_matrix
        self._solve_shape = solve_shape
        self._basis_matrix = basis_matrix
        self._basis_is_multi = is_multi
        self._basis_shapes = shapes
        self._basis_sizes = sizes

    def _get_smoothness_operator(self, n_basis: int) -> Optional[np.ndarray]:
        if self.regularization == 0.0 or self._basis_positions is None:
            return None
        if len(self._basis_positions) != n_basis:
            raise ValueError(
                "basis_positions length must match cached basis length: "
                f"got {len(self._basis_positions)} vs {n_basis}."
            )
        if self._smoothness_operator is not None:
            return self._smoothness_operator
        if n_basis < 2:
            self._smoothness_operator = np.empty((0, n_basis), dtype=np.float32)
            return self._smoothness_operator

        k = min(self.smoothness_neighbors, n_basis - 1)
        positions = self._basis_positions.astype(np.float64, copy=False)
        distances = np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(distances, np.inf)
        neighbors = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        edges = sorted(
            {
                (min(i, int(j)), max(i, int(j)))
                for i, row in enumerate(neighbors)
                for j in row
            }
        )
        operator = np.zeros((len(edges), n_basis), dtype=np.float32)
        for row, (left, right) in enumerate(edges):
            operator[row, left] = 1.0
            operator[row, right] = -1.0
        self._smoothness_operator = operator
        return operator

    def _weights_for_one(
        self, accessor: BasisAccessor, rng: np.random.Generator
    ) -> np.ndarray:
        coeff_map = _squeeze_2d(self.pattern_provider(rng)).astype(
            np.float32, copy=False
        )
        if not np.all(np.isfinite(coeff_map)):
            raise ValueError("pattern_provider returned non-finite values.")
        self.last_coeff_map = coeff_map
        scale = _resolve_intensity_scale(self.intensity_scale, coeff_map, rng)
        self.last_intensity_scale = scale

        if self._solve_matrix is None or self._solve_shape != coeff_map.shape:
            self._build_basis_matrices(accessor, coeff_map.shape)
        coefficients = solve_regularized_nnls(
            self._solve_matrix,
            coeff_map.ravel() * scale,
            regularization=self.regularization,
            smoothness_operator=self._get_smoothness_operator(len(accessor)),
            maxiter=self.maxiter,
        )
        if self.max_coefficient is not None:
            np.minimum(coefficients, self.max_coefficient, out=coefficients)
        coefficients[coefficients <= self.coefficient_threshold] = 0.0
        self.last_coefficients = coefficients
        return coefficients

    def _finalize_flat(self, flat: np.ndarray) -> Any:
        components = []
        offset = 0
        for shape, size in zip(self._basis_shapes, self._basis_sizes):
            components.append(flat[offset : offset + size].reshape(shape))
            offset += size
        output = tuple(components) if self._basis_is_multi else components[0]
        for transform in self.transforms:
            output = transform(output)

        lo, hi = self.clip_output
        if self._basis_is_multi:
            if not isinstance(output, (tuple, list)):
                raise ValueError(
                    "transforms must return tuple/list for multi-component samples"
                )
            return tuple(
                np.clip(np.asarray(component, dtype=np.float32), lo, hi)
                for component in output
            )
        return np.clip(np.asarray(output, dtype=np.float32), lo, hi)

    def __call__(self, accessor: BasisAccessor, rng: np.random.Generator) -> Any:
        coefficients = self._weights_for_one(accessor, rng)
        flat = coefficients @ self._basis_matrix
        used = np.flatnonzero(coefficients)
        self._last_record = CombinationRecord(
            indices=[int(index) for index in used],
            coefficients=[float(coefficients[index]) for index in used],
        )
        return self._finalize_flat(flat)

    def generate_batch(
        self, accessor: BasisAccessor, rng: np.random.Generator, batch_size: int
    ) -> List[Any]:
        """Solve samples independently and reconstruct them with one matrix product."""
        weights = np.empty((batch_size, len(accessor)), dtype=np.float32)
        for row in range(batch_size):
            weights[row] = self._weights_for_one(accessor, rng)
        flats = weights @ self._basis_matrix
        outputs = [self._finalize_flat(flat) for flat in flats]
        used = np.flatnonzero(weights[-1])
        self._last_record = CombinationRecord(
            indices=[int(index) for index in used],
            coefficients=[float(weights[-1, index]) for index in used],
        )
        return outputs

    def reset_cache(self) -> None:
        """Drop cached matrices after a basis change."""
        self._solve_matrix = None
        self._solve_shape = None
        self._basis_matrix = None
        self._smoothness_operator = None
