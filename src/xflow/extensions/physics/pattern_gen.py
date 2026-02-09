# Minimal core for DynamicPatterns and StaticGaussianDistribution

from abc import ABC, abstractmethod
from typing import Callable, Generator, List

import numpy as np
from scipy.stats import beta


class DynamicPatterns:
    """
    Minimal dynamic 2D canvas that holds multiple Distribution objects.
    """

    def __init__(self, height: int = 128, width: int = 128) -> None:
        self._height = self._validate_and_convert(height)
        self._width = self._validate_and_convert(width)

        self.canvas: np.ndarray = np.zeros((self._height, self._width), dtype=float)
        self._distributions: List[Distribution] = []

        # Clip ceiling for the final canvas after summing distributions.
        self.max_pixel_value: float = 255.0

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

    def thresholding(self, threshold: float = 5) -> None:
        self.canvas[self.canvas < threshold] = 0.0

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

    def update(self, *args, **kwargs) -> None:
        """
        Update all distributions (recompute patterns), then apply them.
        """
        self.clear_canvas()
        for dst in self._distributions:
            dst.update(*args, **kwargs)
        self.apply_distribution()

    def fast_update(self, *args, **kwargs) -> None:
        """
        Update distribution parameters without generating new patterns.
        """
        self.clear_canvas()
        for dst in self._distributions:
            dst.fast_update(*args, **kwargs)

    def get_image(self) -> np.ndarray:
        return self.canvas

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

    def pattern_stream(self, **update_kwargs) -> Generator[np.ndarray, None, None]:
        """
        Infinite generator. Call next(stream) to get frames.
        """
        while True:
            self.update(**update_kwargs)
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
        distribution: str = "",
    ) -> None:
        # std sampling (in relative units)
        if distribution == "beta":
            self.std_x = self._beta_scaled(std_1)
            self.std_y = self._beta_scaled(std_2)
        else:
            min_std = float(min(std_1, std_2))
            max_std = float(max(std_1, std_2))
            std = np.random.uniform(low=min_std, high=max_std)
            if np.random.uniform(-1.0, 1.0) < 0:
                self.std_x = std
                self.std_y = self.std_x * np.random.uniform(0.5, 2.0)
            else:
                self.std_y = std
                self.std_x = self.std_y * np.random.uniform(0.5, 2.0)

        # scale std to pixel units
        self.std_x *= float(self._width)
        self.std_y *= float(self._height)
        self.std_x = max(self.std_x, 1e-6)
        self.std_y = max(self.std_y, 1e-6)

        # intensity sampling (kept consistent with your formula; guard fade_rate == 1)
        if fade_rate == 1:
            min_intensity = -float(max_intensity)
        else:
            min_intensity = (
                float(fade_rate) * float(max_intensity) / (float(fade_rate) - 1.0)
            )

        self.intensity = float(np.random.uniform(min_intensity, float(max_intensity)))

        if self.intensity > 0:
            area_scaling = (self._width * self._height) / (self.std_x * self.std_y)
            self.intensity += float(np.random.uniform(0.0, area_scaling / 4.0))

            self.rotation = np.deg2rad(np.random.uniform(0.0, 360.0))
            self.dx = float(np.random.uniform(0.0, self._width / 2.25))
            self.dy = float(np.random.uniform(0.0, self._height / 2.25))

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
