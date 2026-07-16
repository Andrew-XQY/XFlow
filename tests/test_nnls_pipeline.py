import numpy as np

from xflow.extensions.physics.nnls_pipeline import (
    NNLSCoefficientMapCombinator,
    solve_regularized_nnls,
)
from xflow.extensions.physics.pipeline import BasisAccessor


def test_plain_nnls_recovers_nonnegative_coefficients():
    matrix = np.eye(3, dtype=np.float32)
    target = np.array([0.25, -1.0, 0.75], dtype=np.float32)

    coefficients = solve_regularized_nnls(matrix, target)

    np.testing.assert_allclose(coefficients, [0.25, 0.0, 0.75])


def test_regularization_reduces_first_difference_norm():
    matrix = np.eye(3, dtype=np.float32)
    target = np.array([1.0, 0.0, 1.0], dtype=np.float32)

    plain = solve_regularized_nnls(matrix, target)
    smooth = solve_regularized_nnls(matrix, target, regularization=1.0)

    assert np.linalg.norm(np.diff(smooth)) < np.linalg.norm(np.diff(plain))


def test_combinator_preserves_paired_basis_convention():
    target_basis = np.eye(4, dtype=np.float32).reshape(4, 2, 2)
    basis = [(2.0 * image[None], image[None]) for image in target_basis]
    accessor = BasisAccessor(basis, list(range(4)))
    pattern = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    combinator = NNLSCoefficientMapCombinator(
        pattern_provider=lambda rng: pattern,
        regularization=0.0,
        clip_output=(0.0, 10.0),
    )

    left, right = combinator(accessor, np.random.default_rng(42))

    np.testing.assert_allclose(right[0], pattern, atol=1e-6)
    np.testing.assert_allclose(left[0], 2.0 * pattern, atol=1e-6)
    assert combinator.last_record.indices == [0, 1, 2, 3]
