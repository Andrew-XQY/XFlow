import unittest

import numpy as np

from xflow.extensions.physics.pattern_gen import (
    DynamicPatterns,
    StaticGaussianDistribution,
)


def _second_moments(image):
    image = np.asarray(image, dtype=float)
    ys, xs = np.indices(image.shape)
    total = image.sum()
    cx = (xs * image).sum() / total
    cy = (ys * image).sum() / total
    var_x = (((xs - cx) ** 2) * image).sum() / total
    var_y = (((ys - cy) ** 2) * image).sum() / total
    cov_xy = (((xs - cx) * (ys - cy)) * image).sum() / total
    return var_x, var_y, cov_xy


def _rendered_center_offset(gaussian):
    """Recover the rendered image-space center offset from internal params."""
    c = np.cos(gaussian.rotation)
    s = np.sin(gaussian.rotation)
    ox = -(c * gaussian.dx + s * gaussian.dy)
    oy = s * gaussian.dx - c * gaussian.dy
    return np.asarray([ox, oy])


class StaticGaussianOrientationTest(unittest.TestCase):
    def _generate(self, orientation):
        canvas = DynamicPatterns(128, 128)
        gaussian = StaticGaussianDistribution(canvas)
        gaussian.update(
            std_1=0.08,
            std_2=0.08,
            intensity_range=(1.0, 1.0),
            area_boost_scale=0.0,
            center_radius_range=(0.0, 0.0),
            aspect_range=(2.0, 2.0),
            orientation_range=(orientation, orientation),
        )
        return gaussian.pattern

    def test_zero_degrees_aligns_major_axis_horizontally(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                np.random.seed(seed)
                var_x, var_y, cov_xy = _second_moments(self._generate(0.0))
                self.assertGreater(var_x, 3.5 * var_y)
                self.assertAlmostEqual(cov_xy, 0.0, delta=1e-8)

    def test_ninety_degrees_aligns_major_axis_vertically(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                np.random.seed(seed)
                var_x, var_y, cov_xy = _second_moments(self._generate(90.0))
                self.assertGreater(var_y, 3.5 * var_x)
                self.assertAlmostEqual(cov_xy, 0.0, delta=1e-8)

    def test_absent_orientation_range_keeps_legacy_random_rotation(self):
        np.random.seed(42)
        canvas = DynamicPatterns(32, 32)
        gaussian = StaticGaussianDistribution(canvas)
        gaussian.update_params(
            std_1=0.08,
            std_2=0.08,
            intensity_range=(1.0, 1.0),
            area_boost_scale=0.0,
            center_radius_range=(0.0, 0.0),
            aspect_range=(2.0, 2.0),
        )
        # The fifth legacy uniform draw is the raw 0..360-degree rotation.
        self.assertAlmostEqual(gaussian.rotation, 0.980294029274052, places=14)


class StaticGaussianCenterTest(unittest.TestCase):
    def _generate(self, center_angle_range):
        canvas = DynamicPatterns(128, 128)
        gaussian = StaticGaussianDistribution(canvas)
        gaussian.update(
            std_1=0.08,
            std_2=0.08,
            intensity_range=(1.0, 1.0),
            area_boost_scale=0.0,
            center_radius_range=(0.25, 0.25),
            center_angle_range=center_angle_range,
            aspect_range=(2.0, 2.0),
            orientation_range=(23.0, 23.0),
        )
        return gaussian

    def test_center_angle_zero_points_right(self):
        np.random.seed(1)
        ox, oy = _rendered_center_offset(self._generate((0.0, 0.0)))
        self.assertAlmostEqual(ox, 32.0, places=12)
        self.assertAlmostEqual(oy, 0.0, places=12)

    def test_center_angle_ninety_points_up(self):
        np.random.seed(1)
        ox, oy = _rendered_center_offset(self._generate((90.0, 90.0)))
        self.assertAlmostEqual(ox, 0.0, places=12)
        self.assertAlmostEqual(oy, -32.0, places=12)

    def test_negative_narrow_center_angle_range(self):
        for seed in range(16):
            with self.subTest(seed=seed):
                np.random.seed(seed)
                ox, oy = _rendered_center_offset(self._generate((-10.0, 10.0)))
                angle = float(np.rad2deg(np.arctan2(-oy, ox)))
                self.assertGreaterEqual(angle, -10.0)
                self.assertLessEqual(angle, 10.0)

    def test_absent_center_angle_keeps_existing_full_circle_draw(self):
        np.random.seed(1234)
        canvas = DynamicPatterns(128, 128)
        gaussian = StaticGaussianDistribution(canvas)
        gaussian.update_params(
            std_1=0.08,
            std_2=0.08,
            intensity_range=(1.0, 1.0),
            area_boost_scale=0.0,
            center_radius_range=(0.2, 0.3),
            aspect_range=(2.0, 2.0),
            orientation_range=(0.0, 0.0),
        )
        self.assertAlmostEqual(gaussian.dx, 4.906239751264972, places=14)
        self.assertAlmostEqual(gaussian.dy, -29.233485487547455, places=14)

    def test_center_angle_requires_radial_center_prior(self):
        np.random.seed(1)
        gaussian = StaticGaussianDistribution(DynamicPatterns(32, 32))
        with self.assertRaisesRegex(ValueError, "requires center_radius_range"):
            gaussian.update_params(
                intensity_range=(1.0, 1.0),
                center_angle_range=(-10.0, 10.0),
            )


class DynamicPatternsSharedCenterTest(unittest.TestCase):
    @staticmethod
    def _canvas():
        canvas = DynamicPatterns(128, 128)
        for _ in range(3):
            canvas.append(StaticGaussianDistribution(canvas))
        return canvas

    @staticmethod
    def _update_kwargs():
        return {
            "std_1": 0.04,
            "std_2": 0.08,
            "intensity_range": (1.0, 1.0),
            "area_boost_scale": 0.0,
            "center_radius_range": (0.2, 0.3),
            "aspect_range": (2.0, 3.0),
            "orientation_range": (-5.0, 5.0),
        }

    def test_shared_center_places_every_gaussian_at_one_frame_center(self):
        np.random.seed(7)
        canvas = self._canvas()
        canvas.update(shared_center=True, **self._update_kwargs())
        offsets = np.stack(
            [_rendered_center_offset(dst) for dst in canvas._distributions]
        )
        np.testing.assert_allclose(
            offsets,
            np.repeat(offsets[:1], len(offsets), axis=0),
            atol=1e-12,
            rtol=0.0,
        )
        radius = np.linalg.norm(offsets[0]) / 128.0
        self.assertGreaterEqual(radius, 0.2)
        self.assertLessEqual(radius, 0.3)

    def test_explicit_false_preserves_independent_legacy_randomness(self):
        kwargs = self._update_kwargs()
        np.random.seed(99)
        absent = self._canvas()
        absent.update(**kwargs)
        absent_image = absent.canvas.copy()
        absent_metadata = absent.get_distributions_metadata()

        np.random.seed(99)
        explicit_false = self._canvas()
        explicit_false.update(shared_center=False, **kwargs)
        np.testing.assert_array_equal(explicit_false.canvas, absent_image)
        self.assertEqual(explicit_false.get_distributions_metadata(), absent_metadata)

    def test_shared_center_requires_radial_center_prior(self):
        canvas = self._canvas()
        with self.assertRaisesRegex(ValueError, "requires center_radius_range"):
            canvas.update(shared_center=True)


class DynamicPatternsComponentParamsTest(unittest.TestCase):
    @staticmethod
    def _canvas(count=2):
        canvas = DynamicPatterns(64, 64)
        for _ in range(count):
            canvas.append(StaticGaussianDistribution(canvas))
        return canvas

    @staticmethod
    def _update_kwargs():
        return {
            "std_1": 0.04,
            "std_2": 0.04,
            "intensity_range": (0.5, 0.5),
            "area_boost_scale": 0.0,
            "center_radius_range": (0.2, 0.3),
            "aspect_range": (1.0, 1.0),
            "orientation_range": (0.0, 0.0),
        }

    def test_none_preserves_absent_legacy_rng_and_output(self):
        kwargs = self._update_kwargs()
        np.random.seed(314)
        absent = self._canvas()
        absent.update(**kwargs)

        np.random.seed(314)
        explicit_none = self._canvas()
        explicit_none.update(component_params=None, **kwargs)

        np.testing.assert_array_equal(explicit_none.canvas, absent.canvas)
        self.assertEqual(
            explicit_none.get_distributions_metadata(),
            absent.get_distributions_metadata(),
        )

    def test_fast_update_merges_each_override_over_common_kwargs(self):
        canvas = self._canvas()
        canvas.fast_update(
            component_params=[
                {"intensity_range": (0.25, 0.25)},
                {
                    "intensity_range": (0.75, 0.75),
                    "std_1": 0.1,
                    "std_2": 0.1,
                },
            ],
            **self._update_kwargs(),
        )

        first, second = canvas._distributions
        self.assertEqual(first.intensity, 0.25)
        self.assertEqual(second.intensity, 0.75)
        self.assertAlmostEqual(first.std_x, 0.04 * canvas.width)
        self.assertAlmostEqual(second.std_x, 0.1 * canvas.width)

    def test_pattern_stream_accepts_component_params(self):
        canvas = self._canvas()
        stream = canvas.pattern_stream(
            component_params=[
                {"intensity_range": (0.2, 0.2)},
                {"intensity_range": (0.8, 0.8)},
            ],
            **self._update_kwargs(),
        )
        image = next(stream)

        self.assertEqual(image.shape, (64, 64))
        self.assertEqual(canvas._distributions[0].intensity, 0.2)
        self.assertEqual(canvas._distributions[1].intensity, 0.8)

    def test_shared_center_is_injected_after_component_overrides(self):
        np.random.seed(19)
        canvas = self._canvas(3)
        canvas.update(
            shared_center=True,
            component_params=[
                {"center_radius_range": (0.0, 0.0)},
                {"center_angle_range": (90.0, 90.0)},
                {"orientation_range": (45.0, 45.0)},
            ],
            **self._update_kwargs(),
        )
        offsets = np.stack(
            [_rendered_center_offset(dst) for dst in canvas._distributions]
        )
        np.testing.assert_allclose(
            offsets,
            np.repeat(offsets[:1], len(offsets), axis=0),
            atol=1e-12,
            rtol=0.0,
        )
        radius = np.linalg.norm(offsets[0]) / canvas.width
        self.assertGreaterEqual(radius, 0.2)
        self.assertLessEqual(radius, 0.3)

    def test_component_params_validation(self):
        canvas = self._canvas()
        with self.assertRaisesRegex(TypeError, "sequence of mappings"):
            canvas.update(component_params={"std_1": 0.1})
        with self.assertRaisesRegex(ValueError, "length must match"):
            canvas.update(component_params=[{}])
        with self.assertRaisesRegex(TypeError, r"component_params\[1\].*mapping"):
            canvas.update(component_params=[{}, None])

    def test_component_params_cannot_override_coordination_keys(self):
        canvas = self._canvas()
        for key, value in (
            ("rendered_center_offset", (1.0, 2.0)),
            ("shared_center", False),
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "reserved key"):
                    canvas.update(component_params=[{key: value}, {}])


if __name__ == "__main__":
    unittest.main()
