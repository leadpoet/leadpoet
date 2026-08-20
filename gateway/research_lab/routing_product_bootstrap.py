"""Static bootstrap for the reviewed Research Lab routing product.

This module is the only process-start composition seam for the routing API and
queue consumer.  It does not import a model from a path, read a provider URL,
or construct credentials from environment values.  A deployment must pass a
release-owned, already-verified dependency bundle.  Missing or substituted
dependencies leave the API and consumer disabled before a store or queue is
constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from gateway.research_lab.routing_execution_envelope import (
    RoutingExperimentExecutionEnvelopeV2,
)
from gateway.research_lab.routing_product_composition import (
    ReviewedRoutingProductComposition,
    ReviewedRoutingReleaseInputs,
    RoutingProductCompositionError,
    bootstrap_reviewed_routing_product,
    install_reviewed_routing_product,
)
from research_lab.routing_experiments import RoutingExperimentV2Spec


class RoutingProductBootstrapError(RoutingProductCompositionError):
    """The static release bootstrap cannot safely install routing."""


@dataclass(frozen=True)
class ReviewedRoutingBootstrapDependencies:
    """Release-owned objects required by the static product bootstrap.

    The object must be constructed by deployment code linked into the release.
    This dataclass is deliberately not populated from request data or an
    environment import path.
    """

    inputs: ReviewedRoutingReleaseInputs
    reviewed_runner_factory: Callable[[RoutingExperimentV2Spec], Any]
    billing_rollup_factory: Callable[[RoutingExperimentV2Spec], Callable[..., Mapping[str, Any]]]
    execution_envelope_factory: Callable[
        [RoutingExperimentV2Spec], RoutingExperimentExecutionEnvelopeV2
    ]
    store_factory: Callable[[], Any]


def load_reviewed_routing_release_dependencies() -> (
    ReviewedRoutingBootstrapDependencies | None
):
    """Load the dependency bundle linked into the exact product release.

    The open-source checkout has no release-owned model adapter and therefore
    returns ``None``.  A release build may replace this fixed function with
    its statically linked loader.  It must not resolve a module, URL, or
    credential from a request or environment value.
    """

    return None


def build_reviewed_routing_product_from_release(
    dependencies: ReviewedRoutingBootstrapDependencies,
    *,
    environment: Mapping[str, str] | None = None,
) -> ReviewedRoutingProductComposition:
    """Build a product only from a typed, release-owned dependency bundle."""

    if not isinstance(dependencies, ReviewedRoutingBootstrapDependencies):
        raise RoutingProductBootstrapError(
            "reviewed routing release dependencies are unavailable"
        )
    for name in (
        "reviewed_runner_factory",
        "billing_rollup_factory",
        "execution_envelope_factory",
        "store_factory",
    ):
        if not callable(getattr(dependencies, name, None)):
            raise RoutingProductBootstrapError(
                f"reviewed routing release dependency {name} is unavailable"
            )
    try:
        return bootstrap_reviewed_routing_product(
            inputs=dependencies.inputs,
            reviewed_runner_factory=dependencies.reviewed_runner_factory,
            billing_rollup_factory=dependencies.billing_rollup_factory,
            execution_envelope_factory=dependencies.execution_envelope_factory,
            store_factory=dependencies.store_factory,
            environment=environment,
        )
    except RoutingProductCompositionError as exc:
        raise RoutingProductBootstrapError(str(exc)) from exc


def install_reviewed_routing_product_at_startup(
    app: Any,
    *,
    dependencies: ReviewedRoutingBootstrapDependencies | None = None,
    dependency_loader: Callable[
        [], ReviewedRoutingBootstrapDependencies | None
    ] | None = None,
    environment: Mapping[str, str] | None = None,
) -> ReviewedRoutingProductComposition | None:
    """Install the reviewed product before the first request.

    ``None`` is a normal fail-closed state for the current release.  The
    caller must not continue with a partially built composition.  The
    dependency bundle is explicit so no request or environment value can
    select Python code, an endpoint, a credential, or a provider client.
    """

    from gateway.research_lab.routing_experiment_api import (
        install_routing_experiment_api_service,
    )

    # ``dependencies`` and ``dependency_loader`` are test/deployment seams.
    # The production gateway passes neither and therefore always calls the
    # one fixed release loader below; it never reads app.state for objects.
    if dependencies is None and dependency_loader is None:
        dependency_loader = load_reviewed_routing_release_dependencies
    if dependencies is None and dependency_loader is not None:
        try:
            dependencies = dependency_loader()
        except Exception as exc:  # noqa: BLE001 - startup remains fail closed
            dependencies = None
            loader_error = type(exc).__name__
        else:
            loader_error = None
    else:
        loader_error = None

    if dependencies is None:
        install_routing_experiment_api_service(None, app=app)
        if getattr(app, "state", None) is not None:
            app.state.reviewed_routing_product_composition = None
            app.state.reviewed_routing_bootstrap_error = (
                loader_error
                or "reviewed routing release dependencies are unavailable"
            )
        return None

    try:
        composition = build_reviewed_routing_product_from_release(
            dependencies,
            environment=environment,
        )
        install_reviewed_routing_product(composition, app=app)
    except Exception as exc:  # noqa: BLE001 - startup must remain fail closed
        install_routing_experiment_api_service(None, app=app)
        if getattr(app, "state", None) is not None:
            app.state.reviewed_routing_product_composition = None
            app.state.reviewed_routing_bootstrap_error = type(exc).__name__
        return None

    if getattr(app, "state", None) is not None:
        app.state.reviewed_routing_bootstrap_error = None
        app.state.reviewed_routing_product_composition = composition
    return composition


def bootstrap_reviewed_routing_consumer_process(
    *,
    dependencies: ReviewedRoutingBootstrapDependencies | None = None,
    environment: Mapping[str, str] | None = None,
) -> ReviewedRoutingProductComposition:
    """Bootstrap the child process from the same static release seam.

    A separate child cannot inherit the parent app object.  Until deployment
    supplies a release-linked dependency bundle in this process, it fails
    before constructing the Supabase store or claiming a queue lease.
    """

    if dependencies is None:
        dependencies = load_reviewed_routing_release_dependencies()
    if dependencies is None:
        raise RoutingProductBootstrapError(
            "reviewed routing consumer release dependencies are unavailable"
        )
    return build_reviewed_routing_product_from_release(
        dependencies,
        environment=environment,
    )


__all__ = [
    "RoutingProductBootstrapError",
    "ReviewedRoutingBootstrapDependencies",
    "load_reviewed_routing_release_dependencies",
    "build_reviewed_routing_product_from_release",
    "install_reviewed_routing_product_at_startup",
    "bootstrap_reviewed_routing_consumer_process",
]
