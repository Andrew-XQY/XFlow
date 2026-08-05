# Minimal core for DynamicPatterns and StaticGaussianDistribution

from abc import ABC, abstractmethod
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from scipy.stats import beta

PatternSource = Union[Iterable[Any], Callable[[], Iterable[Any]]]


def _sample_center_offset(
    width: int,
    height: int,
    center_radius_range: Tuple[float, float],
    center_angle_range: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """Sample a rendered center offset in image coordinates."""
    r_lo, r_hi = (
        float(center_radius_range[0]),
        float(center_radius_range[1]),
    )
    if not (0.0 <= r_lo <= r_hi):
        raise ValueError(
            "center_radius_range must satisfy 0 <= lo <= hi, "
            f"got {center_radius_range!r}"
        )

    side = float(min(width, height))
    # Sample radius area-uniformly within the requested annulus.
    rho = side * np.sqrt(np.random.uniform(r_lo**2, r_hi**2))
    if center_angle_range is None:
        # Keep the existing full-circle draw and image-coordinate convention.
        phi = np.random.uniform(0.0, 2.0 * np.pi)
        return float(rho * np.cos(phi)), float(rho * np.sin(phi))

    angle_lo, angle_hi = (
        float(center_angle_range[0]),
        float(center_angle_range[1]),
    )
    if not (np.isfinite(angle_lo) and np.isfinite(angle_hi) and angle_lo <= angle_hi):
        raise ValueError(
            "center_angle_range must contain finite degrees and satisfy "
            f"lo <= hi, got {center_angle_range!r}"
        )
    angle = np.deg2rad(np.random.uniform(angle_lo, angle_hi))
    # Angles use Cartesian semantics even though image y increases downward:
    # 0 degrees points right and 90 degrees points up.
    return float(rho * np.cos(angle)), float(-rho * np.sin(angle))


def _to_2d_float32(x: Any) -> np.ndarray:
    """Convert an image-like input to a finite 2D float32 array."""
    arr = np.asarray(x)
    arr = np.squeeze(arr)

    if arr.ndim == 3:
        if arr.shape[-1] in (1, 3, 4):
            # Channel-last layout: (H, W, C)
            arr = arr[..., 0] if arr.shape[-1] == 1 else arr[..., :3].mean(axis=-1)
        elif arr.shape[0] in (1, 3, 4):
            # Channel-first layout: (C, H, W)
            arr = arr[0] if arr.shape[0] == 1 else arr[:3].mean(axis=0)

    if arr.ndim != 2:
        raise ValueError(f"Pattern must be 2D after transforms, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Pattern contains non-finite values.")

    return arr.astype(np.float32, copy=False)


def image_pattern_stream(
    source: PatternSource,
    *,
    transforms: Optional[List[Callable[[Any], Any]]] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
    cache_transformed: bool = False,
) -> Generator[np.ndarray, None, None]:
    """Yield an infinite stream of 2D float32 patterns from image inputs.

    This generator is intentionally lightweight and decoupled from pipeline classes.
    It accepts either:
      - an iterable of image-like items, or
      - a callable returning such an iterable (for provider-style sources).

    Each item may be an image directly, or a ``(id, image)`` tuple where only the
    image component is used.

    Args:
        source: Iterable source or callable returning an iterable source.
        transforms: Optional per-item transform callables applied in order.
        shuffle: If True, reshuffle item order each full pass.
        seed: Optional RNG seed for deterministic shuffling.
        cache_transformed: If True, apply transforms once and reuse the resulting
            patterns on every pass.

    Yields:
        2D ``np.float32`` arrays suitable for pattern-provider contracts.
    """
    fns = list(transforms or [])
    items = list(source() if callable(source) else source)

    if not items:
        raise ValueError("image_pattern_stream source is empty.")

    rng = np.random.default_rng(seed)
    order = np.arange(len(items), dtype=np.int64)

    def process(raw: Any) -> np.ndarray:
        x = raw[1] if isinstance(raw, tuple) and len(raw) == 2 else raw
        for fn in fns:
            x = fn(x)
        return _to_2d_float32(x)

    patterns = [process(raw) for raw in items] if cache_transformed else None

    while True:
        if shuffle and len(order) > 1:
            rng.shuffle(order)

        for i in order:
            idx = int(i)
            yield patterns[idx] if patterns is not None else process(items[idx])


def weighted_stream(
    sources: Sequence[PatternSource],
    *,
    probabilities: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
) -> Generator[Any, None, None]:
    """Yield items by sampling from multiple sources using source probabilities.

    This utility is intentionally source-agnostic: each source can be an iterable
    or a callable returning an iterable. It is suitable for mixing any stream-like
    outputs that share a downstream contract.

    Args:
        sources: Sequence of source iterables or source callables.
        probabilities: Optional non-negative sampling weights per source.
            If None, sources are sampled uniformly.
        seed: Optional RNG seed for deterministic source selection.

    Yields:
        Items from selected source iterators.

    Raises:
        ValueError: If sources/weights are invalid.
        RuntimeError: If a selected source is exhausted.
        TypeError: If a source is not iterable.
    """
    if not sources:
        raise ValueError("weighted_stream requires at least one source.")

    n_sources = len(sources)

    if probabilities is None:
        probs = np.full(n_sources, 1.0 / n_sources, dtype=np.float64)
    else:
        if len(probabilities) != n_sources:
            raise ValueError(
                "probabilities length must match number of sources: "
                f"got {len(probabilities)} vs {n_sources}."
            )

        probs = np.asarray(probabilities, dtype=np.float64)
        if np.any(~np.isfinite(probs)):
            raise ValueError("probabilities must be finite numbers.")
        if np.any(probs < 0.0):
            raise ValueError("probabilities must be >= 0.")

        total = float(np.sum(probs))
        if total <= 0.0:
            raise ValueError("probabilities must sum to a value > 0.")
        probs = probs / total

    iterators = []
    for idx, src in enumerate(sources):
        resolved = src() if callable(src) else src
        if not hasattr(resolved, "__iter__"):
            raise TypeError(
                f"Source at index {idx} is not iterable: {type(resolved)!r}."
            )
        iterators.append(iter(resolved))

    rng = np.random.default_rng(seed)

    while True:
        src_idx = int(rng.choice(n_sources, p=probs))
        try:
            yield next(iterators[src_idx])
        except StopIteration as e:
            raise RuntimeError(
                f"Source at index {src_idx} is exhausted. "
                "Ensure each source is infinite or restartable for weighted_stream."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Source at index {src_idx} failed while producing an item: {e}"
            ) from e


class DynamicPatterns:
    """
    Minimal dynamic 2D canvas that holds multiple Distribution objects.
    """

    def __init__(
        self,
        height: int = 128,
        width: int = 128,
        postprocess_fns: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,
    ) -> None:
        self._height = self._validate_and_convert(height)
        self._width = self._validate_and_convert(width)

        self.canvas: np.ndarray = np.zeros((self._height, self._width), dtype=float)
        self._distributions: List[Distribution] = []
        self._postprocess_fns: List[Callable[[np.ndarray], np.ndarray]] = []

        # Clip ceiling for the final canvas after summing distributions.
        self.max_pixel_value: float = 255.0
        self.threshold: Optional[float] = None
        self.set_postprocess_fns(postprocess_fns)

    def __repr__(self) -> str:
        return f"DynamicPatterns(canvas_shape={self.canvas.shape}, max_pixel_value={self.max_pixel_value})"

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, value: int) -> None:
        self._height = self._validate_and_convert(value)
        self.clear_canvas()

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, value: int) -> None:
        self._width = self._validate_and_convert(value)
        self.clear_canvas()

    def set_max_pixel_value(self, value: float) -> None:
        """
        Your application calls this.
        """
        try:
            v = float(value)
        except Exception as e:
            raise ValueError(f"max_pixel_value must be a number, got {value!r}") from e
        if v <= 0:
            raise ValueError("max_pixel_value must be > 0")
        self.max_pixel_value = v

    def set_postprocess_fns(
        self, fns: Optional[List[Callable[[np.ndarray], np.ndarray]]]
    ) -> None:
        """Set output postprocessing hooks with contract: ndarray -> ndarray."""
        if fns is None:
            self._postprocess_fns = []
            return
        if not isinstance(fns, list):
            raise TypeError("postprocess_fns must be a list of callables or None")
        for fn in fns:
            if not callable(fn):
                raise TypeError("Each postprocess function must be callable")
        self._postprocess_fns = list(fns)

    def add_postprocess_fn(self, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        """Append one output postprocessing hook with contract: ndarray -> ndarray."""
        if not callable(fn):
            raise TypeError("postprocess function must be callable")
        self._postprocess_fns.append(fn)

    def clear_postprocess_fns(self) -> None:
        """Remove all output postprocessing hooks."""
        self._postprocess_fns = []

    def _validate_and_convert(self, value: int) -> int:
        if not isinstance(value, int):
            try:
                value = int(value)
            except Exception as e:
                raise ValueError("Value must be convertible to an integer.") from e
        # Keep the original behavior (clip into [0,4096])
        if value < 0:
            value = 0
        if value > 4096:
            value = 4096
        return value

    def clear_canvas(self) -> None:
        self.canvas = np.zeros((self._height, self._width), dtype=float)

    def set_threshold(self, threshold: Optional[float]) -> None:
        if threshold is None:
            self.threshold = None
            return
        try:
            threshold = float(threshold)
        except Exception as e:
            raise ValueError(
                f"threshold must be a number or None, got {threshold!r}"
            ) from e
        self.threshold = threshold

    def thresholding(self) -> None:
        if self.threshold is None:
            return
        self.canvas[self.canvas < self.threshold] = 0.0

    def is_blank(self) -> bool:
        return np.all(self.canvas == 0)

    def append(self, distribution: "Distribution") -> None:
        self._distributions.append(distribution)

    def apply_distribution(self) -> None:
        """
        Sum patterns onto the canvas and clip to max_pixel_value.
        """
        for dst in self._distributions:
            self.canvas += dst.pattern
        self.canvas = np.clip(self.canvas, 0.0, self.max_pixel_value)
        self.thresholding()

    def _validate_component_params(
        self,
        component_params: Optional[Sequence[Mapping[str, Any]]],
    ) -> Optional[List[dict]]:
        """Validate and copy per-distribution keyword overrides."""
        if component_params is None:
            return None
        if isinstance(component_params, (str, bytes)) or not isinstance(
            component_params, SequenceABC
        ):
            raise TypeError("component_params must be a sequence of mappings or None")
        if len(component_params) != len(self._distributions):
            raise ValueError(
                "component_params length must match number of distributions: "
                f"got {len(component_params)} vs {len(self._distributions)}"
            )

        reserved = {"rendered_center_offset", "shared_center"}
        validated = []
        for idx, params in enumerate(component_params):
            if not isinstance(params, MappingABC):
                raise TypeError(
                    f"component_params[{idx}] must be a mapping, "
                    f"got {type(params)!r}"
                )
            conflicts = reserved.intersection(params)
            if conflicts:
                names = ", ".join(sorted(conflicts))
                raise ValueError(
                    f"component_params[{idx}] cannot set reserved key(s): {names}"
                )
            validated.append(dict(params))
        return validated

    def update(
        self,
        *args,
        component_params: Optional[Sequence[Mapping[str, Any]]] = None,
        **kwargs,
    ) -> None:
        """
        Update all distributions (recompute patterns), then apply them.

        ``component_params`` optionally supplies one mapping per distribution.
        Each mapping overrides the common keyword arguments for that component.
        """
        self.clear_canvas()
        component_params = self._validate_component_params(component_params)
        shared_center = kwargs.pop("shared_center", False)
        if not isinstance(shared_center, (bool, np.bool_)):
            raise TypeError("shared_center must be a boolean")
        if shared_center:
            if kwargs.get("center_radius_range") is None:
                raise ValueError("shared_center=True requires center_radius_range")
            if not all(
                isinstance(dst, StaticGaussianDistribution)
                for dst in self._distributions
            ):
                raise TypeError(
                    "shared_center is supported only for StaticGaussianDistribution"
                )
            kwargs["rendered_center_offset"] = _sample_center_offset(
                self._width,
                self._height,
                kwargs["center_radius_range"],
                kwargs.get("center_angle_range"),
            )
        if component_params is None:
            # Keep the legacy path unchanged, including its RNG draw order.
            for dst in self._distributions:
                dst.update(*args, **kwargs)
        else:
            for dst, overrides in zip(self._distributions, component_params):
                component_kwargs = dict(kwargs)
                component_kwargs.update(overrides)
                # The shared offset is internal coordination state and must win
                # over every user-supplied per-component override.
                if shared_center:
                    component_kwargs["rendered_center_offset"] = kwargs[
                        "rendered_center_offset"
                    ]
                dst.update(*args, **component_kwargs)
        self.apply_distribution()

    def fast_update(
        self,
        *args,
        component_params: Optional[Sequence[Mapping[str, Any]]] = None,
        **kwargs,
    ) -> None:
        """
        Update distribution parameters without generating new patterns.
        """
        self.clear_canvas()
        component_params = self._validate_component_params(component_params)
        shared_center = kwargs.pop("shared_center", False)
        if not isinstance(shared_center, (bool, np.bool_)):
            raise TypeError("shared_center must be a boolean")
        if shared_center:
            if kwargs.get("center_radius_range") is None:
                raise ValueError("shared_center=True requires center_radius_range")
            if not all(
                isinstance(dst, StaticGaussianDistribution)
                for dst in self._distributions
            ):
                raise TypeError(
                    "shared_center is supported only for StaticGaussianDistribution"
                )
            kwargs["rendered_center_offset"] = _sample_center_offset(
                self._width,
                self._height,
                kwargs["center_radius_range"],
                kwargs.get("center_angle_range"),
            )
        if component_params is None:
            # Keep the legacy path unchanged, including its RNG draw order.
            for dst in self._distributions:
                dst.fast_update(*args, **kwargs)
        else:
            for dst, overrides in zip(self._distributions, component_params):
                component_kwargs = dict(kwargs)
                component_kwargs.update(overrides)
                if shared_center:
                    component_kwargs["rendered_center_offset"] = kwargs[
                        "rendered_center_offset"
                    ]
                dst.fast_update(*args, **component_kwargs)

    def get_image(self) -> np.ndarray:
        if not self._postprocess_fns:
            return self.canvas

        out = self.canvas.copy()
        for fn in self._postprocess_fns:
            out = np.asarray(fn(out))
        return out

    def num_of_distributions(self) -> int:
        return sum(0 if dst.is_empty() else 1 for dst in self._distributions)

    def get_metadata(self) -> dict:
        return {
            "simulation_resolution": (self._height, self._width),
            "num_of_distributions": self.num_of_distributions(),
            "types": list({dst._type for dst in self._distributions}),
        }

    def get_distributions_metadata(self) -> List[dict]:
        return [dst.get_metadata() for dst in self._distributions]

    def pattern_stream(
        self,
        *,
        component_params: Optional[Sequence[Mapping[str, Any]]] = None,
        **update_kwargs,
    ) -> Generator[np.ndarray, None, None]:
        """
        Infinite generator. Call next(stream) to get frames.
        """
        while True:
            self.update(component_params=component_params, **update_kwargs)
            yield self.get_image()


class Distribution(ABC):
    """
    Minimal abstract base class.
    """

    def __init__(self, canvas: DynamicPatterns):
        self._type: str = "Distribution"
        self._height = canvas.height
        self._width = canvas.width
        self._pattern = np.zeros((self._height, self._width), dtype=float)

    @property
    def pattern(self) -> np.ndarray:
        return self._pattern

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def fast_update(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def pattern_generation(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_metadata(self) -> dict:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass


class StaticGaussianDistribution(Distribution):
    """
    Your "static gaussian" with randomised params per update.
    Uses numpy only (np.random.beta) for the beta branch.
    """

    def __init__(self, canvas: DynamicPatterns) -> None:
        super().__init__(canvas)
        self._type = "Static_Gaussian"

        self.std_x: float = 0.0
        self.std_y: float = 0.0
        self.intensity: float = 0.0
        self.rotation: float = 0.0
        self.dx: float = 0.0
        self.dy: float = 0.0

    def _beta_scaled(
        self, mode: float, loc: float = 0.01, scale: float = 0.99
    ) -> float:
        # Replicates the intent of your original scipy.stats.beta.rvs(..., loc, scale)
        mode = float(np.clip(mode, 1e-6, 1.0 - 1e-6))
        decay_factor_a = 5.0
        decay_factor_b = 15.0
        a = max(1e-3, decay_factor_a * 2.0 * mode)
        b = max(1e-3, decay_factor_b * 2.0 * (1.0 - mode))
        return loc + scale * np.random.beta(a, b)

    def update_params(
        self,
        std_1: float = 0.15,
        std_2: float = 0.12,
        max_intensity: float = 10,
        fade_rate: float = 0.5,
        area_boost_scale: float = 0.25,
        max_peak_intensity: Optional[float] = None,
        distribution: str = "",
        intensity_range: Optional[Tuple[float, float]] = None,
        center_radius_range: Optional[Tuple[float, float]] = None,
        aspect_range: Optional[Tuple[float, float]] = None,
        orientation_range: Optional[Tuple[float, float]] = None,
        center_angle_range: Optional[Tuple[float, float]] = None,
        rendered_center_offset: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Randomize blob parameters.

        Optional config-driven priors (each None -> exact legacy behavior):
            intensity_range: (lo, hi), lo > 0. Intensity ~ U(lo, hi). Replaces
                the fade_rate formula, whose lower bound is always <= 0 (so a
                fraction of faint/empty blobs was unavoidable).
            center_radius_range: (r_lo, r_hi) in relative units (fraction of the
                smaller canvas side). Blob center placed area-uniformly in the
                annulus r_lo <= r <= r_hi around the canvas center. Replaces the
                legacy hardcoded dx,dy ~ U(0, side/2.25).
            aspect_range: (lo, hi) multiplier for the second-axis std
                (legacy hardcoded U(0.5, 2.0)).
            orientation_range: (lo, hi) rendered major-axis orientation in
                degrees, measured from the horizontal image axis. The sampled
                angle is compensated for whichever internal Gaussian axis is
                wider. The legacy path samples the internal rotation uniformly
                over 0..360 degrees.
            center_angle_range: (lo, hi) center-displacement angle in degrees,
                where 0 points right and 90 points up in the rendered image.
                Negative and narrow ranges are supported. Requires
                center_radius_range. None keeps the full-circle legacy draw.
            rendered_center_offset: Internal rendered ``(x, y)`` pixel offset
                used by DynamicPatterns to coordinate a shared frame center.
        """
        # std sampling (in relative units)
        if distribution == "beta":
            self.std_x = self._beta_scaled(std_1)
            self.std_y = self._beta_scaled(std_2)
        else:
            aspect_lo, aspect_hi = (
                aspect_range if aspect_range is not None else (0.5, 2.0)
            )
            min_std = float(min(std_1, std_2))
            max_std = float(max(std_1, std_2))
            std = np.random.uniform(low=min_std, high=max_std)
            if np.random.uniform(-1.0, 1.0) < 0:
                self.std_x = std
                self.std_y = self.std_x * np.random.uniform(aspect_lo, aspect_hi)
            else:
                self.std_y = std
                self.std_x = self.std_y * np.random.uniform(aspect_lo, aspect_hi)

        # scale std to pixel units
        self.std_x *= float(self._width)
        self.std_y *= float(self._height)
        self.std_x = max(self.std_x, 1e-6)
        self.std_y = max(self.std_y, 1e-6)

        # intensity sampling
        if intensity_range is not None:
            lo, hi = float(intensity_range[0]), float(intensity_range[1])
            if not (0.0 < lo <= hi):
                raise ValueError(
                    f"intensity_range must satisfy 0 < lo <= hi, got {intensity_range!r}"
                )
            self.intensity = float(np.random.uniform(lo, hi))
        else:
            # legacy formula (kept consistent; guard fade_rate == 1)
            if fade_rate == 1:
                min_intensity = -float(max_intensity)
            else:
                min_intensity = (
                    float(fade_rate) * float(max_intensity) / (float(fade_rate) - 1.0)
                )
            self.intensity = float(
                np.random.uniform(min_intensity, float(max_intensity))
            )

        if self.intensity > 0:
            # Smaller spots can have higher peaks for the same charge, but keep
            # the inverse-area boost tunable so tiny Gaussians do not dominate.
            boost_scale = max(0.0, float(area_boost_scale))
            if boost_scale > 0.0:
                area_scaling = (self._width * self._height) / (self.std_x * self.std_y)
                self.intensity += float(
                    np.random.uniform(0.0, area_scaling * boost_scale)
                )
            if max_peak_intensity is not None:
                peak_cap = float(max_peak_intensity)
                if peak_cap <= 0:
                    raise ValueError("max_peak_intensity must be > 0 when set")
                self.intensity = min(self.intensity, peak_cap)

            if orientation_range is None:
                # Preserve the exact legacy random-orientation path.
                self.rotation = np.deg2rad(np.random.uniform(0.0, 360.0))
            else:
                angle_lo, angle_hi = (
                    float(orientation_range[0]),
                    float(orientation_range[1]),
                )
                if not (
                    np.isfinite(angle_lo)
                    and np.isfinite(angle_hi)
                    and angle_lo <= angle_hi
                ):
                    raise ValueError(
                        "orientation_range must contain finite degrees and satisfy "
                        f"lo <= hi, got {orientation_range!r}"
                    )
                major_axis_angle = np.deg2rad(np.random.uniform(angle_lo, angle_hi))
                # pattern_generation applies R(theta) to the coordinates. Its
                # rendered x-axis therefore lies at -theta; its y-axis lies at
                # pi/2-theta. Compensate for the random std_x/std_y assignment
                # so this option always controls the *major* rendered axis.
                if self.std_x >= self.std_y:
                    rotation = -major_axis_angle
                else:
                    rotation = np.pi / 2.0 - major_axis_angle
                self.rotation = float(rotation % (2.0 * np.pi))

            if rendered_center_offset is not None:
                ox, oy = (
                    float(rendered_center_offset[0]),
                    float(rendered_center_offset[1]),
                )
                if not (np.isfinite(ox) and np.isfinite(oy)):
                    raise ValueError(
                        "rendered_center_offset must contain finite pixel offsets"
                    )
            elif center_radius_range is not None:
                ox, oy = _sample_center_offset(
                    self._width,
                    self._height,
                    center_radius_range,
                    center_angle_range,
                )
            else:
                if center_angle_range is not None:
                    raise ValueError("center_angle_range requires center_radius_range")
                # legacy hardcoded behavior
                self.dx = float(np.random.uniform(0.0, self._width / 2.25))
                self.dy = float(np.random.uniform(0.0, self._height / 2.25))
                return

            # pattern_generation places the peak at [x; y] = R^{-1}(-[dx; dy])
            # with R = [[c, -s], [s, c]]; to land it at (ox, oy):
            #   [dx; dy] = -R [ox; oy]   (verified <1px placement error)
            c, s = np.cos(self.rotation), np.sin(self.rotation)
            self.dx = float(-(c * ox - s * oy))
            self.dy = float(-(s * ox + c * oy))

    def pattern_generation(self) -> np.ndarray:
        if self.intensity <= 0:
            return np.zeros((self._height, self._width), dtype=float)

        x = np.linspace(0, self._width - 1, self._width, dtype=float)
        y = np.linspace(0, self._height - 1, self._height, dtype=float)
        x, y = np.meshgrid(x, y)

        # shift origin to center, then rotate, then translate (same as your code)
        x -= self._width / 2.0
        y -= self._height / 2.0

        cos_t = np.cos(self.rotation)
        sin_t = np.sin(self.rotation)
        x_new = x * cos_t - y * sin_t
        y_new = x * sin_t + y * cos_t

        x_new += self.dx
        y_new += self.dy

        dist = np.exp(
            -((x_new**2) / (2.0 * self.std_x**2) + (y_new**2) / (2.0 * self.std_y**2))
        )
        dist *= self.intensity
        return dist

    def update(self, *args, **kwargs) -> None:
        self.update_params(*args, **kwargs)
        self._pattern = self.pattern_generation()

    def fast_update(self, *args, **kwargs) -> None:
        self.update_params(*args, **kwargs)

    def is_empty(self) -> bool:
        return self.intensity <= 0

    def get_metadata(self) -> dict:
        return {
            "is_empty": self.is_empty(),
            "intensity": float(self.intensity),
            "std_x": float(self.std_x) / float(self._width),
            "std_y": float(self.std_y) / float(self._height),
            "rotation": float(self.rotation),
            "x": float(self.dx),
            "y": float(self.dy),
            "type": self._type,
        }
