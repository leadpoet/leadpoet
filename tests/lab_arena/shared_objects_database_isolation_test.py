"""Fixture buckets share durable objects only within one actual database."""
from contextlib import ExitStack, contextmanager

import pytest

from lab_arena.contracts import ArenaContractError
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration
from tests.lab_arena.test_lab_arena_service_round import Harness


@contextmanager
def _database():
    generator = database_with_lab_arena_migration(())
    try:
        yield next(generator)
    finally:
        generator.close()


def test_harness_buckets_isolate_databases_and_preserve_same_database_objects(tmp_path):
    with ExitStack() as stack:
        pg, first = stack.enter_context(_database())
        _, second = stack.enter_context(_database())
        first_connect = lambda: pg.connect(**first)
        second_connect = lambda: pg.connect(**second)
        # The real Harness binds the object root before building the service.
        # No migration or service calls are needed to test bucket identity.
        class BucketHarness(Harness):
            def build_service(self):
                return None

        original = BucketHarness(first_connect, tmp_path / "first", challengers=[], runners=[])
        same_database = BucketHarness(
            lambda: pg.connect(**first), tmp_path / "another-parent" / "another-directory",
            challengers=[], runners=[]
        )
        other_database = BucketHarness(second_connect, tmp_path / "second", challengers=[], runners=[])
        ref = "arena/arena-2026-10-22/benchmark.json"
        original.objects.put(ref, b"first benchmark")
        assert same_database.objects_root == original.objects_root
        assert same_database.objects.get(ref) == b"first benchmark"
        # A conflict inside the same bucket must still fail; isolation must
        # not remove immutable-object validation to hide fixture collisions.
        with pytest.raises(ArenaContractError, match="already holds different bytes"):
            same_database.objects.put(ref, b"different benchmark")
        assert other_database.objects_root != original.objects_root
        other_database.objects.put(ref, b"different benchmark")
        assert original.objects.get(ref) == b"first benchmark"
        assert other_database.objects.get(ref) == b"different benchmark"
