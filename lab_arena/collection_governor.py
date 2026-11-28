"""Standing-pool admission using the finalized epoch, approved route and original miner grant."""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from lab_arena import chain
from lab_arena.collection_intake import CollectionIntake
from lab_arena.collection_prices import CatalogError, endpoint_prices
from lab_arena.collection_rewards import emission_valuation


def prepare_epoch(
    graph, cutover, database, directory_lock, *, value=emission_valuation
):
    snapshot = chain.finalized_epoch_snapshot(graph)
    epoch = snapshot.settlement_epoch_id(cutover)
    block = snapshot.last_epoch_block
    anchor = graph.finalized_block_hash(block)
    params = {"p_epoch": epoch, "p_boundary": block, "p_hash": anchor}

    def refresh(valuation):
        return database.rpc(
            "collection_epoch_capacity_v1", dict(params, p_valuation=valuation)
        )

    if refresh(None) is not True:
        # Parallel lanes on this host use a single fresh oracle observation.
        with directory_lock():
            if refresh(None) is not True and refresh(value(graph, block)) is not True:
                raise ValueError("standing epoch has no measured funding capacity")
    return epoch


class HeldRoute(Exception):
    def __init__(self, model, cause):
        super().__init__("collection route in backoff")
        self.model = model
        self.cause = cause


def retry_after_seconds(header):
    if not isinstance(header, str) or len(header) > 128:
        return 0
    try:
        seconds = (
            int(header)
            if header.isascii() and header.isdecimal()
            else int(
                (
                    parsedate_to_datetime(header) - datetime.now(timezone.utc)
                ).total_seconds()
            )
            + 1
        )
    except (ValueError, TypeError, OverflowError):
        return 0
    return min(86400, max(0, seconds))


class GovernedIntake(CollectionIntake):
    def __init__(self, database, credentials, epoch, routes, current_epoch):
        super().__init__(database, credentials, "live")
        self.epoch = epoch
        self.routes = routes
        self.current_epoch = current_epoch

    def _lease(self, token, expected_model, **params):
        if not params["batch"].startswith("pool-"):
            raise ValueError("governor only admits standing collection")
        if expected_model is not None and self.current_epoch() != self.epoch:
            return None  # The next preview must use the new finalized epoch.
        roster = set(params["models"])
        job = self._rpc(
            "governed_claim",
            **params,
            expected_model=expected_model,
            epoch=self.epoch,
            routes={m: self.routes[m] for m in roster},
        )
        if expected_model is None or not job:
            return job
        return dict(job, lease_token=token)

    def prices(self, model):
        try:
            return endpoint_prices(self.routes, models=[model])
        except CatalogError as exc:
            # With no preflight, there is no provider POST or ambiguous lease.
            self._rpc(
                "route_backoff",
                model=model,
                route=self.routes[model],
                wait=retry_after_seconds(exc.retry_after),
            )
            raise HeldRoute(model, exc) from exc

    def respect_retry_after(self, model, header):
        seconds = retry_after_seconds(header)
        if seconds > 0:
            self._rpc(
                "route_backoff",
                model=model,
                route=self.routes[model],
                wait=seconds,
            )
