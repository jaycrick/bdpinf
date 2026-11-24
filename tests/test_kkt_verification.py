"""Test the KKT verification of constrained optima on small known problems."""

import numpy as np

from bdpinf.inference.constrained import verify_kkt

_TOL = 1e-6


def _toy(point: np.ndarray) -> dict[str, np.ndarray]:
    """Minimise ||x - (2, 0)||^2 / 2 subject to c(x) = 1 - x0 >= 0.

    The optimum is (1, 0) with multiplier 1 on the constraint.
    """
    return {
        "grad": point - np.array([2.0, 0.0]),
        "hess": np.eye(2),
        "c": np.array([1.0 - point[0]]),
        "jac": np.array([[-1.0, 0.0]]),
        "hess_c": np.zeros((1, 2, 2)),
    }


class TestVerifyKKT:
    """Test verify_kkt on a 2-D toy constrained problem."""

    def test_known_optimum_verified(self) -> None:
        report = verify_kkt(**_toy(np.array([1.0, 0.0])), grad_tol=_TOL)
        assert report.verified
        assert report.kind == "constrained"
        assert report.residual < 1e-12
        assert np.isclose(report.multipliers[0], 1.0)
        assert report.active == (0,)
        assert report.min_reduced_eig is not None
        assert np.isclose(report.min_reduced_eig, 1.0)

    def test_wrong_sign_multiplier_rejected(self) -> None:
        # Minimise ||x + (2, 0)||^2 / 2 with the same constraint x0 <= 1,
        # evaluated at x0 = 1: the gradient points the wrong way
        # (the constraint would need a negative multiplier).
        point = np.array([1.0, 0.0])
        problem = _toy(point)
        problem["grad"] = point + np.array([2.0, 0.0])
        report = verify_kkt(**problem, grad_tol=_TOL)
        assert not report.verified
        assert report.residual > 1.0
        assert np.all(report.multipliers >= 0)

    def test_non_stationary_point_rejected(self) -> None:
        report = verify_kkt(**_toy(np.array([1.0, 0.5])), grad_tol=_TOL)
        assert not report.verified
        assert report.residual > 0.4

    def test_complementarity_violation_rejected(self) -> None:
        # Constraint is barely "active" by tolerance but the multiplier is huge.
        problem = _toy(np.array([1.0 - 5e-7, 0.0]))
        problem["grad"] = np.array([-1.0e3, 0.0])
        report = verify_kkt(**problem, grad_tol=1e9, slack_tol=1e-6)
        assert report.complementarity > 1e-6
        assert not report.verified

    def test_negative_curvature_rejected(self) -> None:
        problem = _toy(np.array([1.0, 0.0]))
        problem["hess"] = np.diag([1.0, -1.0])
        report = verify_kkt(**problem, grad_tol=_TOL)
        assert not report.verified
        assert report.min_reduced_eig is not None
        assert report.min_reduced_eig < 0

    def test_non_finite_likelihood_rejected(self) -> None:
        report = verify_kkt(**_toy(np.array([1.0, 0.0])), grad_tol=_TOL, finite=False)
        assert not report.verified

    def test_interior_point_uses_ordinary_test(self) -> None:
        point = np.array([0.0, 0.0])
        problem = _toy(point)
        problem["grad"] = np.zeros(2)
        problem["c"] = np.array([1.0])
        interior = verify_kkt(**problem, grad_tol=_TOL)
        assert interior.kind == "interior"
        assert interior.verified
        problem["hess"] = np.diag([1.0, -1.0])
        assert not verify_kkt(**problem, grad_tol=_TOL).verified

    def test_active_transform_bound_counts_as_active(self) -> None:
        # Minimise x0 + x1^2 / 2 with x0 >= 0 as a transform bound:
        # at the bound the multiplier is the gradient 1, and the flat direction
        # of the bound coordinate is removed from the tangent space.
        grad = np.array([1.0, 0.0])
        report = verify_kkt(
            grad,
            np.diag([0.0, 1.0]),
            np.array([5.0]),
            np.array([[0.0, 1.0]]),
            None,
            grad_tol=_TOL,
            bound_rows=np.array([[1.0, 0.0]]),
        )
        assert report.verified
        assert report.bound_active == (0,)

    def test_slackness_is_measured_from_the_held_margin(self) -> None:
        # A large multiplier times a tiny held margin is the price of strict
        # feasibility, not a complementarity violation.
        margin = 2e-9
        problem = _toy(np.array([1.0 - margin, 0.0]))
        problem["grad"] = np.array([-900.0, 0.0])
        strict = verify_kkt(**problem, grad_tol=1e-6)
        assert strict.complementarity > 1e-6
        assert not strict.verified
        held = verify_kkt(**problem, grad_tol=1e-6, margin=margin)
        assert held.complementarity <= 1e-6
        assert held.verified

    def test_non_finite_gradient_is_unverified_not_an_error(self) -> None:
        problem = _toy(np.array([1.0, 0.0]))
        problem["grad"] = np.array([np.nan, 0.0])
        report = verify_kkt(**problem, grad_tol=_TOL)
        assert not report.verified
        assert report.kind == "non-finite"
