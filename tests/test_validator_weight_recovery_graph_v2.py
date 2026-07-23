from validator_tee.enclave import hotkey_authority_v2 as module


def test_recovery_graph_keeps_only_computed_receipt_ancestors():
    receipts = [
        {
            "receipt_hash": "sha256:" + "1" * 64,
            "job_id": "input",
            "purpose": "validator.weight_snapshot.v2",
            "boot_identity_hash": "sha256:" + "a" * 64,
            "parent_receipt_hashes": [],
        },
        {
            "receipt_hash": "sha256:" + "2" * 64,
            "job_id": "compute",
            "purpose": "validator.weights.computed.v2",
            "boot_identity_hash": "sha256:" + "a" * 64,
            "parent_receipt_hashes": ["sha256:" + "1" * 64],
        },
        {
            "receipt_hash": "sha256:" + "3" * 64,
            "job_id": "binding",
            "purpose": "validator.hotkey_signature.v2",
            "boot_identity_hash": "sha256:" + "b" * 64,
            "parent_receipt_hashes": ["sha256:" + "2" * 64],
        },
    ]
    graph = {
        "root_receipt_hash": "sha256:" + "3" * 64,
        "boot_identities": [
            {"boot_identity_hash": "sha256:" + "a" * 64},
            {"boot_identity_hash": "sha256:" + "b" * 64},
        ],
        "receipts": receipts,
        "transport_attempts": [
            {"job_id": receipt["job_id"], "purpose": receipt["purpose"]}
            for receipt in receipts
        ],
        "host_operations": [
            {
                "request": {
                    "job_id": receipt["job_id"],
                    "purpose": receipt["purpose"],
                }
            }
            for receipt in receipts
        ],
    }
    result = module._receipt_ancestor_graph(
        graph,
        root_receipt_hash="sha256:" + "2" * 64,
    )

    assert result["root_receipt_hash"] == "sha256:" + "2" * 64
    assert [item["receipt_hash"] for item in result["receipts"]] == [
        "sha256:" + "1" * 64,
        "sha256:" + "2" * 64,
    ]
    assert result["boot_identities"] == [
        {"boot_identity_hash": "sha256:" + "a" * 64}
    ]
    assert {
        (item["job_id"], item["purpose"])
        for item in result["transport_attempts"]
    } == {
        ("input", "validator.weight_snapshot.v2"),
        ("compute", "validator.weights.computed.v2"),
    }
    assert {
        (item["request"]["job_id"], item["request"]["purpose"])
        for item in result["host_operations"]
    } == {
        ("input", "validator.weight_snapshot.v2"),
        ("compute", "validator.weights.computed.v2"),
    }
