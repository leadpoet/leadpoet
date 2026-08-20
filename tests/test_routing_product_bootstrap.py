from __future__ import annotations

from types import SimpleNamespace

from gateway.research_lab.routing_product_bootstrap import (
    RoutingProductBootstrapError,
    ReviewedRoutingBootstrapDependencies,
    install_reviewed_routing_product_at_startup,
)


def test_missing_release_dependencies_fail_closed_before_store_or_queue() -> None:
    app = SimpleNamespace(state=SimpleNamespace())
    result = install_reviewed_routing_product_at_startup(app)
    assert result is None
    assert app.state.routing_experiment_api_service is None
    assert app.state.reviewed_routing_product_composition is None


def test_untyped_release_dependencies_are_rejected_before_construction() -> None:
    app = SimpleNamespace(state=SimpleNamespace())
    try:
        install_reviewed_routing_product_at_startup(
            app,
            dependencies=object(),  # type: ignore[arg-type]
        )
    except Exception as exc:  # startup helper converts this to fail closed
        raise AssertionError("startup helper must fail closed") from exc
    assert app.state.routing_experiment_api_service is None


def test_dependency_bundle_is_explicit_and_not_an_import_path() -> None:
    assert ReviewedRoutingBootstrapDependencies.__dataclass_fields__.keys() == {
        "inputs",
        "reviewed_runner_factory",
        "billing_rollup_factory",
        "execution_envelope_factory",
        "store_factory",
    }
    assert RoutingProductBootstrapError.__mro__[1].__name__ == (
        "RoutingProductCompositionError"
    )


def test_startup_uses_fixed_loader_and_ignores_app_state(monkeypatch) -> None:
    app = SimpleNamespace(
        state=SimpleNamespace(
            reviewed_routing_bootstrap_dependencies=object(),
        )
    )
    called = []

    def _loader():
        called.append(True)
        return None

    monkeypatch.setattr(
        "gateway.research_lab.routing_product_bootstrap.load_reviewed_routing_release_dependencies",
        _loader,
    )
    assert install_reviewed_routing_product_at_startup(app) is None
    assert called == [True]
