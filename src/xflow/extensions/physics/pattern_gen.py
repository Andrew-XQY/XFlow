# Minimal core for DynamicPatterns and StaticGaussianDistribution

from typing import Callable, Generator, List

import numpy as np
from scipy.stats import beta


class DynamicPatterns:
    def __init__(self, height: int = 128, width: int = 128) -> None:
        self._height = self._validate_and_convert(height)
        self._width = self._validate_and_convert(width)
        self.clear_canvas()
        self._distributions = []
        self.max_pixel_value = 255

    def _validate_and_convert(self, value: int) -> int:
        if not isinstance(value, int):
            value = int(value)
        if not (0 <= value <= 4096):
            value = 4096 if value > 4096 else 0
        return value

    def clear_canvas(self) -> None:
        self.canvas = np.zeros((self._height, self._width))

    def update(self, *args, **kwargs) -> None:
        self.clear_canvas()
        for dst in self._distributions:
            dst.update(*args, **kwargs)
        self.apply_distribution()

    def apply_distribution(self) -> None:
        for dst in self._distributions:
            self.canvas += dst.pattern
            self.canvas = np.clip(self.canvas, 0, self.max_pixel_value)

    def get_image(self) -> np.ndarray:
        return self.canvas

    def pattern_stream(self, **update_kwargs) -> Generator[np.ndarray, None, None]:
        while True:
            self.update(**update_kwargs)
            yield self.get_image()


class StaticGaussianDistribution:
    def __init__(self, canvas: DynamicPatterns) -> None:
        self._type = "Static_Gaussian"
        self._height = canvas._height
        self._width = canvas._width
        self.std_x = 0
        self.std_y = 0
        self.intensity = 0
        self.rotation = 0
        self.dx = 0
        self.dy = 0
        self._pattern = np.zeros((self._height, self._width))

    @property
    def pattern(self) -> np.ndarray:
        return self._pattern

    def update_params(
        self,
        std_1: float = 0.15,
        std_2: float = 0.12,
        max_intensity: int = 10,
        fade_rate: float = 0.5,
        distribution: str = "",
    ) -> None:
        if distribution == "beta":
            c_x = std_1
            c_y = std_2
            loc = 0.01
            scale = 1 - loc
            decay_factor_a = 5
            decay_factor_b = 15
            a_x = decay_factor_a * 2 * c_x
            b_x = decay_factor_b * 2 * (1 - c_x)
            a_y = decay_factor_a * 2 * c_y
            b_y = decay_factor_b * 2 * (1 - c_y)
            self.std_x = beta.rvs(a_x, b_x, loc=loc, scale=scale)
            self.std_y = beta.rvs(a_y, b_y, loc=loc, scale=scale)
        else:
            min_std = min(std_1, std_2)
            max_std = max(std_1, std_2)
            temp = np.random.uniform(low=-1, high=1)
            std = np.random.uniform(low=min_std, high=max_std)
            if temp < 0:
                self.std_x = std
                self.std_y = self.std_x * np.random.uniform(0.5, 2)
            else:
                self.std_y = std
                self.std_x = self.std_y * np.random.uniform(0.5, 2)
        self.std_x *= self._width
        self.std_y *= self._height
        min_intensity = fade_rate * max_intensity / (fade_rate - 1)
        self.intensity = np.random.uniform(min_intensity, max_intensity)
        if self.intensity > 0:
            area_scaling = (self._width * self._height) / (self.std_x * self.std_y)
            self.intensity += np.random.uniform(0, area_scaling / 4)
            angle_degrees = np.random.uniform(0, 360)
            self.rotation = np.deg2rad(angle_degrees)
            self.dx = np.random.uniform(0, self._width // 2.25)
            self.dy = np.random.uniform(0, self._height // 2.25)

    def pattern_generation(self) -> np.ndarray:
        if self.intensity <= 0:
            return np.zeros((self._height, self._width))
        x = np.linspace(0, self._width - 1, self._width)
        y = np.linspace(0, self._height - 1, self._height)
        x, y = np.meshgrid(x, y)
        x -= self._width / 2
        y -= self._height / 2
        x_new = x * np.cos(self.rotation) - y * np.sin(self.rotation)
        y_new = x * np.sin(self.rotation) + y * np.cos(self.rotation)
        x_new += self.dx
        y_new += self.dy
        dist = np.exp(
            -((x_new) ** 2 / (2 * self.std_x**2) + (y_new) ** 2 / (2 * self.std_y**2))
        )
        dist *= self.intensity
        return dist

    def update(self, *args, **kwargs) -> None:
        self.update_params(*args, **kwargs)
        self._pattern = self.pattern_generation()
