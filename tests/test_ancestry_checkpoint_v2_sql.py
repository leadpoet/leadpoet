from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

import pytest

from gateway.tee.supabase_schema_preflight_v2 import (
    REQUIRED_SUPABASE_V2_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[1]
MIGRATION_NAME = "scripts/136-research-lab-ancestry-checkpoint-sidecars.sql"
SQL = (ROOT / MIGRATION_NAME).read_text(encoding="utf-8")
COMPACT_MIGRATION_NAME = (
    "scripts/143-research-lab-compact-ancestry-checkpoints.sql"
)
COMPACT_SQL = (ROOT / COMPACT_MIGRATION_NAME).read_text(encoding="utf-8")
DISCLOSURE_FAST_PATH_MIGRATION_NAME = (
    "scripts/155-research-lab-ancestry-disclosure-root-fast-path.sql"
)
DISCLOSURE_FAST_PATH_SQL = (
    ROOT / DISCLOSURE_FAST_PATH_MIGRATION_NAME
).read_text(encoding="utf-8")
TABLE = "research_lab_attested_ancestry_checkpoints_v2"
COMPACT_WEIGHT_TABLE = "research_lab_compact_weight_authorities_v2"


def test_ancestry_checkpoint_migration_is_additive_and_ordered() -> None:
    assert SQL.lstrip().startswith(
        "-- Authenticated, bounded V2 receipt ancestry and compact weight authority."
    )
    assert re.search(r"\bBEGIN\s*;", SQL)
    assert re.search(r"\bCOMMIT\s*;\s*$", SQL)
    assert "CREATE TABLE IF NOT EXISTS\npublic.%s" % TABLE in SQL
    assert not re.search(r"^\s*UPDATE\s+", SQL, flags=re.MULTILINE)
    assert not re.search(r"^\s*DELETE\s+FROM\s+", SQL, flags=re.MULTILINE)
    altered_relations = set(
        re.findall(
            r"ALTER\s+TABLE\s+(?:public\.)?([a-z][a-z0-9_]*)",
            SQL,
            flags=re.IGNORECASE,
        )
    )
    altered_relations.discard("public")  # Dynamic ALTER TABLE public.%I loop.
    assert altered_relations == {COMPACT_WEIGHT_TABLE}
    assert "Raw receipts, attempts, host operations, and edges remain append-only" in SQL
    assert "full-graph parent" in SQL


def test_compaction_activation_is_per_root_not_global_lineage() -> None:
    activation = re.search(
        r"CREATE TABLE IF NOT EXISTS\s+public\.research_lab_attested_ancestry_activations_v2\s*\((.*?)\n\);",
        SQL,
        flags=re.DOTALL,
    )
    assert activation is not None
    body = activation.group(1)
    assert re.search(r"activation_root_receipt_hash\s+TEXT PRIMARY KEY", body)
    assert not re.search(r"lineage_id\s+TEXT PRIMARY KEY", body)
    assert "UNIQUE (lineage_id, activation_root_receipt_hash)" in body
    assert "a.activation_root_receipt_hash = parent->>'parent_receipt_hash'" in SQL
    assert "ON CONFLICT (activation_root_receipt_hash) DO NOTHING" in SQL
    assert "activated ancestry lineage rejects" not in SQL


def test_ancestry_checkpoint_migration_binds_durable_receipt_and_boot() -> None:
    assert (
        "REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash)"
        in SQL
    )
    assert (
        "REFERENCES public.research_lab_attested_boot_identities_v2(boot_identity_hash)"
        in SQL
    )
    assert SQL.count("ON DELETE RESTRICT") >= 4
    assert re.search(r"root_receipt_hash\s+TEXT PRIMARY KEY", SQL)
    assert re.search(r"certificate_hash\s+TEXT NOT NULL UNIQUE", SQL)
    assert re.search(r"proof_hash\s+TEXT NOT NULL UNIQUE", SQL)
    assert re.search(r"certificate_sequence\s+BIGINT NOT NULL", SQL)
    assert "CHECK (certificate_sequence >= 0)" in SQL
    assert "UNIQUE (root_receipt_hash, lineage_id)" in SQL


def test_ancestry_checkpoint_hashes_and_embedded_proof_are_cross_bound() -> None:
    for column in ("lineage_id", "certificate_hash", "proof_hash"):
        assert re.search(
            rf"{column}\s+TEXT\s+NOT NULL(?:\s+UNIQUE)?\s+"
            r"CHECK \(" + column + r" ~ '\^sha256:\[0-9a-f\]\{64\}\$'\)",
            SQL,
        )
    required_checks = (
        "certificate_doc->>'schema_version' = schema_version",
        "certificate_doc->>'certificate_hash' = certificate_hash",
        "certificate_doc #>> '{claim,output_root_receipt_hash}' = root_receipt_hash",
        "certificate_doc #>> '{claim,lineage_id}' = lineage_id",
        "(certificate_doc #>> '{claim,certificate_sequence}')::BIGINT = certificate_sequence",
        "certificate_doc #>> '{claim,issuer_boot_identity_hash}' = issuer_boot_identity_hash",
        "proof_doc->>'schema_version' = 'leadpoet.attested_ancestry_compact_proof.v2'",
        "proof_doc->>'proof_hash' = proof_hash",
        "proof_doc #>> '{certificate,schema_version}' = schema_version",
        "proof_doc #>> '{certificate,certificate_hash}' = certificate_hash",
        "proof_doc #>> '{certificate,claim,output_root_receipt_hash}' = root_receipt_hash",
        "proof_doc #>> '{certificate,claim,lineage_id}' = lineage_id",
        "(proof_doc #>> '{certificate,claim,certificate_sequence}')::BIGINT = certificate_sequence",
        "proof_doc #>> '{certificate,claim,issuer_boot_identity_hash}' = issuer_boot_identity_hash",
    )
    for marker in required_checks:
        assert marker in SQL
    assert "leadpoet.attested_ancestry_certificate.v2" in SQL
    assert "jsonb_typeof(certificate_doc) = 'object'" in SQL
    assert "jsonb_typeof(proof_doc) = 'object'" in SQL


def test_ancestry_checkpoint_is_append_only_and_service_role_private() -> None:
    assert "BEFORE UPDATE OR DELETE" in SQL
    assert "prevent_research_lab_attested_v2_mutation()" in SQL
    assert "ENABLE ROW LEVEL SECURITY" in SQL
    assert "FROM PUBLIC, anon, authenticated" in SQL
    assert "'GRANT SELECT ON TABLE public.%I TO service_role'" in SQL
    assert "persist_research_lab_ancestry_checkpoint_v2(JSONB)" in SQL
    assert re.search(
        r"GRANT\s+EXECUTE\s+ON\s+FUNCTION[\s\S]+?"
        r"persist_research_lab_ancestry_checkpoint_v2\(JSONB\)[\s\S]+?"
        r"TO\s+service_role",
        SQL,
        flags=re.IGNORECASE,
    )
    assert "FOR SELECT TO service_role USING (true)" in SQL
    assert "FOR INSERT TO service_role WITH CHECK (true)" in SQL
    assert not re.search(r"GRANT\s+(?:UPDATE|DELETE)", SQL, flags=re.IGNORECASE)
    assert not re.search(
        r"GRANT\s+[^;]+\s+TO\s+(?:PUBLIC|anon|authenticated)",
        SQL,
        flags=re.IGNORECASE,
    )


def test_ancestry_checkpoint_indexes_bound_lineage_sequence_and_issuer() -> None:
    assert "idx_research_lab_ancestry_checkpoint_lineage_v2" in SQL
    assert re.search(
        r"lineage_id,\s*certificate_sequence\s+DESC",
        SQL,
        flags=re.IGNORECASE,
    )
    assert "idx_research_lab_ancestry_checkpoint_issuer_v2" in SQL
    assert re.search(
        r"idx_research_lab_ancestry_checkpoint_issuer_v2[\s\S]+?"
        r"issuer_boot_identity_hash",
        SQL,
    )


def test_gateway_schema_preflight_requires_the_checkpoint_sidecar() -> None:
    matches = [
        (migration, relation, tuple(columns))
        for migration, relation, columns in REQUIRED_SUPABASE_V2_SCHEMA
        if relation == TABLE
    ]
    assert matches == [
        (
            MIGRATION_NAME,
            TABLE,
            (
                "root_receipt_hash",
                "schema_version",
                "lineage_id",
                "certificate_hash",
                "certificate_sequence",
                "issuer_boot_identity_hash",
                "proof_hash",
                "checkpoint_graph_hash",
                "certificate_doc",
                "proof_doc",
                "checkpoint_graph_doc",
            ),
        )
    ]


def test_compact_weight_sidecar_is_bound_indexed_and_preflighted() -> None:
    assert "CREATE TABLE IF NOT EXISTS\npublic.%s" % COMPACT_WEIGHT_TABLE in SQL
    for marker in (
        "UNIQUE (netuid, epoch_id, validator_hotkey, authority_stage)",
        "authority_doc->>'authority_hash' = authority_hash",
        "authority_doc #>> '{compact_submission,compact_submission_hash}' = compact_submission_hash",
        "authority_doc #>> '{publication,publication_receipt_hash}' = publication_receipt_hash",
        "FOREIGN KEY (binding_receipt_hash, lineage_id)",
        "FOREIGN KEY (publication_receipt_hash, lineage_id)",
        "FOREIGN KEY (finalization_receipt_hash, lineage_id)",
        "submission_doc #>> '{validator_ancestry_proof,certificate,claim,lineage_id}' = lineage_id",
        "authority_doc #>> '{publication,ancestry_proof,certificate,claim,lineage_id}' = lineage_id",
        "authority_doc #>> '{finalization,compact_submission,validator_ancestry_proof,certificate,claim,lineage_id}' = lineage_id",
        "idx_research_lab_compact_weight_identity_v2",
    ):
        assert marker in SQL
    matches = [
        (migration, relation, tuple(columns))
        for migration, relation, columns in REQUIRED_SUPABASE_V2_SCHEMA
        if relation == COMPACT_WEIGHT_TABLE
    ]
    assert matches == [
        (
            MIGRATION_NAME,
            COMPACT_WEIGHT_TABLE,
            (
                "bundle_hash",
                "netuid",
                "epoch_id",
                "validator_hotkey",
                "authority_stage",
                "schema_version",
                "lineage_id",
                "authority_hash",
                "compact_submission_hash",
                "publication_receipt_hash",
                "compact_finalization_hash",
                "finalization_receipt_hash",
                "authority_doc",
            ),
        )
    ]


def test_checkpoint_rpc_is_required_before_gateway_shutdown() -> None:
    from gateway.tee.supabase_schema_preflight_v2 import REQUIRED_SUPABASE_V2_RPCS

    assert (
        MIGRATION_NAME,
        "persist_research_lab_ancestry_checkpoint_v2",
    ) in REQUIRED_SUPABASE_V2_RPCS


def test_compact_checkpoint_migration_is_additive_and_preflight_required() -> None:
    from gateway.tee.supabase_schema_preflight_v2 import REQUIRED_SUPABASE_V2_RPCS

    assert COMPACT_SQL.lstrip().startswith(
        "-- Compact operational ancestry checkpoints without duplicating raw sidecars."
    )
    assert re.search(r"\bBEGIN\s*;", COMPACT_SQL)
    assert re.search(r"\bCOMMIT\s*;\s*$", COMPACT_SQL)
    assert not re.search(r"^\s*UPDATE\s+", COMPACT_SQL, flags=re.MULTILINE)
    assert not re.search(
        r"^\s*DELETE\s+FROM\s+", COMPACT_SQL, flags=re.MULTILINE
    )
    assert "VALIDATE CONSTRAINT" not in COMPACT_SQL
    assert "new_row_constraint_enabled" in COMPACT_SQL
    assert "historical_rows_append_only" in COMPACT_SQL
    assert "leadpoet.attested_checkpointed_receipt_graph.v3" in COMPACT_SQL
    assert "leadpoet.attested_checkpointed_receipt_graph.v4" in COMPACT_SQL
    assert "compact checkpoint raw sidecars are incomplete" in COMPACT_SQL
    assert (
        COMPACT_MIGRATION_NAME,
        "research_lab_compact_checkpoint_graph_contract_v1",
    ) in REQUIRED_SUPABASE_V2_RPCS


def test_disclosure_root_fast_path_preserves_exact_fallback_and_is_preflighted() -> None:
    from gateway.tee.supabase_schema_preflight_v2 import REQUIRED_SUPABASE_V2_RPCS

    assert re.search(r"\bBEGIN\s*;", DISCLOSURE_FAST_PATH_SQL)
    assert re.search(r"\bCOMMIT\s*;\s*$", DISCLOSURE_FAST_PATH_SQL)
    root_predicate = (
        "p.root_receipt_hash = parent->>'parent_receipt_hash'"
    )
    # One existing exact-root predicate serves certificate parents; migration
    # 155 adds the second for certificate-disclosure parents.
    assert DISCLOSURE_FAST_PATH_SQL.count(root_predicate) == 2
    assert DISCLOSURE_FAST_PATH_SQL.count(
        "p.lineage_id = lineage"
    ) >= 3
    assert DISCLOSURE_FAST_PATH_SQL.count(
        "p.certificate_sequence = (parent->>'authority_sequence')::BIGINT"
    ) >= 3
    assert DISCLOSURE_FAST_PATH_SQL.count(
        "disclosed->>'receipt_hash' = parent->>'parent_receipt_hash'"
    ) == 2
    assert re.search(
        r"IF NOT EXISTS \([\s\S]+?root_receipt_hash[\s\S]+?\) THEN\s+"
        r"IF NOT EXISTS \(",
        DISCLOSURE_FAST_PATH_SQL,
    )
    assert (
        DISCLOSURE_FAST_PATH_MIGRATION_NAME,
        "research_lab_ancestry_disclosure_lookup_contract_v1",
    ) in REQUIRED_SUPABASE_V2_RPCS


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _named_sha(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _checkpoint(
    *,
    root: str,
    lineage: str,
    certificate_hash: str,
    proof_hash: str,
    graph_hash: str,
    issuer: str,
    sequence: int,
    parent_authorities: list[dict[str, object]],
) -> dict[str, object]:
    claim = {
        "output_root_receipt_hash": root,
        "lineage_id": lineage,
        "certificate_sequence": sequence,
        "issuer_boot_identity_hash": issuer,
        "parent_authorities": parent_authorities,
    }
    certificate = {
        "schema_version": "leadpoet.attested_ancestry_certificate.v2",
        "certificate_hash": certificate_hash,
        "claim": claim,
    }
    proof = {
        "schema_version": "leadpoet.attested_ancestry_compact_proof.v2",
        "proof_hash": proof_hash,
        "certificate": certificate,
        "disclosed_receipts": [],
    }
    graph = {
        "schema_version": "leadpoet.attested_checkpointed_receipt_graph.v3",
        "root_receipt_hash": root,
        "ancestry_lineage_id": lineage,
        "ancestry_proof": proof,
    }
    return {
        "root_receipt_hash": root,
        "schema_version": "leadpoet.attested_ancestry_certificate.v2",
        "lineage_id": lineage,
        "certificate_hash": certificate_hash,
        "certificate_sequence": sequence,
        "issuer_boot_identity_hash": issuer,
        "proof_hash": proof_hash,
        "checkpoint_graph_hash": graph_hash,
        "certificate_doc": certificate,
        "proof_doc": proof,
        "checkpoint_graph_doc": graph,
    }


def test_migration_executes_idempotently_and_enforces_irreversible_frontiers() -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker is required for the PostgreSQL migration contract")
    info = subprocess.run(
        ["docker", "info"], capture_output=True, text=True, timeout=15
    )
    if info.returncode != 0:
        pytest.skip("Docker daemon is unavailable")

    container = "leadpoet-ancestry-sql-%s" % uuid.uuid4().hex[:12]

    def psql(statement: str, *, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                container,
                "psql",
                "-X",
                "-A",
                "-t",
                "-U",
                "postgres",
                "-d",
                "leadpoet",
                "-v",
                "ON_ERROR_STOP=1",
            ],
            input=statement,
            capture_output=True,
            text=True,
            timeout=90,
        )
        if expect_success:
            assert result.returncode == 0, result.stderr
        else:
            assert result.returncode != 0, result.stdout
        return result

    setup_sql = """
CREATE ROLE anon NOLOGIN;
CREATE ROLE authenticated NOLOGIN;
CREATE ROLE service_role NOLOGIN;
CREATE TABLE public.research_lab_attested_execution_receipts_v2 (
    receipt_hash TEXT PRIMARY KEY
);
CREATE TABLE public.research_lab_attested_boot_identities_v2 (
    boot_identity_hash TEXT PRIMARY KEY
);
CREATE TABLE public.research_lab_attested_receipt_transport_v2 (
    receipt_hash TEXT NOT NULL,
    attempt_hash TEXT NOT NULL,
    PRIMARY KEY (receipt_hash, attempt_hash)
);
CREATE TABLE public.research_lab_attested_host_operations_v2 (
    request_hash TEXT PRIMARY KEY,
    receipt_hash TEXT NOT NULL
);
CREATE FUNCTION public.prevent_research_lab_attested_v2_mutation()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'append-only relation';
END;
$$;
"""
    lineage = _sha("1")
    issuer = _sha("2")
    root_a, root_b, root_c, root_d, root_e = (_sha(c) for c in "abcde")
    legacy_a, legacy_b = _sha("e"), _sha("f")
    checkpoint_a = _checkpoint(
        root=root_a,
        lineage=lineage,
        certificate_hash=_sha("3"),
        proof_hash=_sha("4"),
        graph_hash=_sha("5"),
        issuer=issuer,
        sequence=0,
        parent_authorities=[
            {
                "authority_kind": "full_projection",
                "parent_receipt_hash": legacy_a,
            }
        ],
    )
    checkpoint_b = _checkpoint(
        root=root_b,
        lineage=lineage,
        certificate_hash=_sha("6"),
        proof_hash=_sha("7"),
        graph_hash=_sha("8"),
        issuer=issuer,
        sequence=0,
        parent_authorities=[
            {
                "authority_kind": "full_projection",
                "parent_receipt_hash": legacy_b,
            }
        ],
    )
    fallback_receipt = _named_sha("fallback-disclosed-receipt")
    checkpoint_a["proof_doc"]["disclosed_receipts"] = [
        {"receipt_hash": root_a}
    ]
    checkpoint_a["checkpoint_graph_doc"]["ancestry_proof"] = checkpoint_a[
        "proof_doc"
    ]
    checkpoint_b["proof_doc"]["disclosed_receipts"] = [
        {"receipt_hash": fallback_receipt}
    ]
    checkpoint_b["checkpoint_graph_doc"]["ancestry_proof"] = checkpoint_b[
        "proof_doc"
    ]
    checkpoint_c = _checkpoint(
        root=root_c,
        lineage=lineage,
        certificate_hash=_sha("9"),
        proof_hash=_sha("0"),
        graph_hash=_sha("a"),
        issuer=issuer,
        sequence=1,
        parent_authorities=[
            {
                "authority_kind": "full_projection",
                "parent_receipt_hash": root_a,
            }
        ],
    )
    checkpoint_d = _checkpoint(
        root=root_d,
        lineage=lineage,
        certificate_hash=_sha("b"),
        proof_hash=_sha("c"),
        graph_hash=_sha("d"),
        issuer=issuer,
        sequence=1,
        parent_authorities=[
            {
                "authority_kind": "certificate",
                "parent_receipt_hash": root_a,
                "authority_hash": checkpoint_a["certificate_hash"],
                "authority_sequence": 0,
            }
        ],
    )

    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                "--rm",
                "--name",
                container,
                "--env",
                "POSTGRES_PASSWORD=postgres",
                "--env",
                "POSTGRES_DB=leadpoet",
                "postgres:15",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        for _ in range(80):
            startup = subprocess.run(
                ["docker", "logs", container],
                capture_output=True,
                text=True,
                timeout=5,
            )
            ready = subprocess.run(
                [
                    "docker",
                    "exec",
                    container,
                    "pg_isready",
                    "-U",
                    "postgres",
                    "-d",
                    "leadpoet",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if (
                "PostgreSQL init process complete; ready for start up."
                in startup.stdout + startup.stderr
                and ready.returncode == 0
            ):
                break
            time.sleep(0.25)
        else:
            raise AssertionError("PostgreSQL 15 did not become ready")

        psql(setup_sql)
        psql(SQL)
        psql(SQL)  # Operators may safely rerun the migration.
        psql(COMPACT_SQL)
        psql(COMPACT_SQL)  # The compact contract is also idempotent.
        psql(DISCLOSURE_FAST_PATH_SQL)
        psql(DISCLOSURE_FAST_PATH_SQL)  # Function replacement is idempotent.
        disclosure_roots = [
            _named_sha(label)
            for label in (
                "direct-disclosure-child",
                "fallback-disclosure-child",
                "wrong-lineage-child",
                "wrong-sequence-child",
                "wrong-hash-child",
            )
        ]
        receipt_values = ", ".join(
            "('%s')" % value
            for value in (
                root_a,
                root_b,
                root_c,
                root_d,
                root_e,
                fallback_receipt,
                *disclosure_roots,
            )
        )
        psql(
            "INSERT INTO public.research_lab_attested_boot_identities_v2 VALUES "
            "('%s'); INSERT INTO public.research_lab_attested_execution_receipts_v2 "
            "VALUES %s;"
            % (issuer, receipt_values)
        )

        for checkpoint in (checkpoint_a, checkpoint_b):
            result = psql(
                "SELECT public.persist_research_lab_ancestry_checkpoint_v2("
                + "'%s'::jsonb);" % json.dumps(checkpoint).replace("'", "''")
            )
            assert '"root_activated": true' in result.stdout

        direct_disclosure = _checkpoint(
            root=disclosure_roots[0],
            lineage=lineage,
            certificate_hash=_named_sha("direct-disclosure-certificate"),
            proof_hash=_named_sha("direct-disclosure-proof"),
            graph_hash=_named_sha("direct-disclosure-graph"),
            issuer=issuer,
            sequence=1,
            parent_authorities=[
                {
                    "authority_kind": "certificate_disclosure",
                    "parent_receipt_hash": root_a,
                    "authority_sequence": 0,
                }
            ],
        )
        direct = psql(
            "SELECT public.persist_research_lab_ancestry_checkpoint_v2("
            + "'%s'::jsonb);"
            % json.dumps(direct_disclosure).replace("'", "''")
        )
        assert '"root_activated": true' in direct.stdout

        fallback_disclosure = _checkpoint(
            root=disclosure_roots[1],
            lineage=lineage,
            certificate_hash=_named_sha("fallback-disclosure-certificate"),
            proof_hash=_named_sha("fallback-disclosure-proof"),
            graph_hash=_named_sha("fallback-disclosure-graph"),
            issuer=issuer,
            sequence=1,
            parent_authorities=[
                {
                    "authority_kind": "certificate_disclosure",
                    "parent_receipt_hash": fallback_receipt,
                    "authority_sequence": 0,
                }
            ],
        )
        fallback = psql(
            "SELECT public.persist_research_lab_ancestry_checkpoint_v2("
            + "'%s'::jsonb);"
            % json.dumps(fallback_disclosure).replace("'", "''")
        )
        assert '"root_activated": true' in fallback.stdout

        wrong_lineage = _checkpoint(
            root=disclosure_roots[2],
            lineage=_named_sha("wrong-lineage"),
            certificate_hash=_named_sha("wrong-lineage-certificate"),
            proof_hash=_named_sha("wrong-lineage-proof"),
            graph_hash=_named_sha("wrong-lineage-graph"),
            issuer=issuer,
            sequence=1,
            parent_authorities=[
                {
                    "authority_kind": "certificate_disclosure",
                    "parent_receipt_hash": root_a,
                    "authority_sequence": 0,
                }
            ],
        )
        wrong_sequence = _checkpoint(
            root=disclosure_roots[3],
            lineage=lineage,
            certificate_hash=_named_sha("wrong-sequence-certificate"),
            proof_hash=_named_sha("wrong-sequence-proof"),
            graph_hash=_named_sha("wrong-sequence-graph"),
            issuer=issuer,
            sequence=2,
            parent_authorities=[
                {
                    "authority_kind": "certificate_disclosure",
                    "parent_receipt_hash": root_a,
                    "authority_sequence": 1,
                }
            ],
        )
        wrong_hash = _checkpoint(
            root=disclosure_roots[4],
            lineage=lineage,
            certificate_hash=_named_sha("wrong-hash-certificate"),
            proof_hash=_named_sha("wrong-hash-proof"),
            graph_hash=_named_sha("wrong-hash-graph"),
            issuer=issuer,
            sequence=1,
            parent_authorities=[
                {
                    "authority_kind": "certificate_disclosure",
                    "parent_receipt_hash": _named_sha("undisclosed-parent"),
                    "authority_sequence": 0,
                }
            ],
        )
        for invalid_disclosure in (
            wrong_lineage,
            wrong_sequence,
            wrong_hash,
        ):
            rejected_disclosure = psql(
                "SELECT public.persist_research_lab_ancestry_checkpoint_v2("
                + "'%s'::jsonb);"
                % json.dumps(invalid_disclosure).replace("'", "''"),
                expect_success=False,
            )
            assert (
                "checkpoint disclosure parent is not durable"
                in rejected_disclosure.stderr
            )

        disclosure_replay = psql(
            "SELECT public.persist_research_lab_ancestry_checkpoint_v2("
            + "'%s'::jsonb);"
            % json.dumps(direct_disclosure).replace("'", "''")
        )
        assert '"status": "persisted"' in disclosure_replay.stdout

        fast_path_plan = psql(
            "SET enable_seqscan TO off; "
            "EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF) "
            "SELECT 1 "
            "FROM public.research_lab_attested_ancestry_checkpoints_v2 p, "
            "LATERAL pg_catalog.jsonb_array_elements("
            "p.proof_doc->'disclosed_receipts') disclosed "
            "WHERE p.root_receipt_hash = '%s' "
            "AND p.lineage_id = '%s' "
            "AND p.certificate_sequence = 0 "
            "AND p.certificate_sequence < 1 "
            "AND disclosed->>'receipt_hash' = '%s'; "
            "RESET enable_seqscan;"
            % (root_a, lineage, root_a)
        )
        assert "Index Scan using" in fast_path_plan.stdout
        assert "root_receipt_hash" in fast_path_plan.stdout
        assert "Execution Time:" in fast_path_plan.stdout

        rejected = psql(
            "SELECT public.persist_research_lab_ancestry_checkpoint_v2("
            + "'%s'::jsonb);" % json.dumps(checkpoint_c).replace("'", "''"),
            expect_success=False,
        )
        assert "compacted ancestry root rejects full graph parent" in rejected.stderr

        chained = psql(
            "SELECT public.persist_research_lab_ancestry_checkpoint_v2("
            + "'%s'::jsonb);" % json.dumps(checkpoint_d).replace("'", "''")
        )
        assert '"root_activated": true' in chained.stdout

        exact_replay = psql(
            "SELECT public.persist_research_lab_ancestry_checkpoint_v2("
            + "'%s'::jsonb);" % json.dumps(checkpoint_a).replace("'", "''")
        )
        assert '"status": "persisted"' in exact_replay.stdout

        conflicting = dict(checkpoint_a)
        conflicting["checkpoint_graph_hash"] = _sha("e")
        conflict = psql(
            "SELECT public.persist_research_lab_ancestry_checkpoint_v2("
            + "'%s'::jsonb);" % json.dumps(conflicting).replace("'", "''"),
            expect_success=False,
        )
        assert "checkpoint durable readback conflicts" in conflict.stderr

        checkpoint_e = _checkpoint(
            root=root_e,
            lineage=lineage,
            certificate_hash=_sha("e"),
            proof_hash=_sha("f"),
            graph_hash=_sha("0"),
            issuer=issuer,
            sequence=2,
            parent_authorities=[],
        )
        projection = {
            "receipt_count": 1,
            "boot_identity_count": 1,
            "transport_attempt_count": 0,
            "host_operation_count": 0,
        }
        checkpoint_e["certificate_doc"]["claim"][
            "local_delta_projection"
        ] = projection
        checkpoint_e["proof_doc"]["disclosed_receipts"] = [
            {"receipt_hash": root_e}
        ]
        checkpoint_e["proof_doc"]["disclosed_boot_identities"] = [
            {"boot_identity_hash": issuer}
        ]
        checkpoint_e["checkpoint_graph_doc"].update(
            {
                "schema_version": (
                    "leadpoet.attested_checkpointed_receipt_graph.v4"
                ),
                "receipts": checkpoint_e["proof_doc"]["disclosed_receipts"],
                "boot_identities": checkpoint_e["proof_doc"][
                    "disclosed_boot_identities"
                ],
                "transport_attempts": [],
                "host_operations": [],
            }
        )
        compact = psql(
            "SELECT public.persist_research_lab_ancestry_checkpoint_v2("
            + "'%s'::jsonb);" % json.dumps(checkpoint_e).replace("'", "''")
        )
        assert '"root_activated": true' in compact.stdout

        incomplete = json.loads(json.dumps(checkpoint_e))
        incomplete["certificate_doc"]["claim"]["local_delta_projection"][
            "transport_attempt_count"
        ] = 1
        incomplete["proof_doc"]["certificate"] = incomplete[
            "certificate_doc"
        ]
        incomplete["checkpoint_graph_doc"]["ancestry_proof"] = incomplete[
            "proof_doc"
        ]
        rejected_incomplete = psql(
            "SELECT public.persist_research_lab_ancestry_checkpoint_v2("
            + "'%s'::jsonb);" % json.dumps(incomplete).replace("'", "''"),
            expect_success=False,
        )
        assert (
            "compact checkpoint raw sidecars are incomplete"
            in rejected_incomplete.stderr
        )
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container],
            capture_output=True,
            text=True,
            timeout=30,
        )
