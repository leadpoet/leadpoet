"""Operator archive transport keeps the canonical historical proof checks."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lab_arena.local_weight_signer import HttpsJsonRpcTransport, HostValidatorChainSource
from tests.test_validator_chain_source_v2 import (
    STATEFUL_LAST_EPOCH_BLOCK,
    STATEFUL_SETTLEMENT_EPOCH_ID,
    _stateful_cutover,
    _stateful_rpc,
)
from validator_tee.enclave.chain_source_v2 import ValidatorChainSourceV2Error


@pytest.mark.parametrize(
    ("tampered_block", "message"),
    [(None, None), (0, "genesis differs"),
     (STATEFUL_LAST_EPOCH_BLOCK, "cutover block hash differs")],
)
def test_operator_archive_http_preserves_canonical_anchor_checks(tampered_block, message):
    _, rpc = _stateful_rpc()
    archive_calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            archive_calls.append((request["method"], request["params"]))
            if request["method"] == "chain_getBlockHash" and request["params"] == [tampered_block]:
                result = "0x" + "99" * 32
            else:
                result = rpc(
                    method=request["method"], params=request["params"],
                    request_id=request["id"], job_id="operator-archive-test",
                    purpose="validator.subnet_epoch_snapshot.v2",
                    logical_operation_id="operator-archive-test:%s" % request["id"],
                )["result"]
            body = json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    class LiveTransport:
        def call(self, **kwargs):
            return rpc(**kwargs)["result"]

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01))
        thread.start()
        archive = HttpsJsonRpcTransport(
            "ws://127.0.0.1:%d" % server.server_port, timeout_seconds=4,
        )
        source = HostValidatorChainSource(
            live_transport=LiveTransport(), archive_transport=archive,
            cutover_manifest=_stateful_cutover(), finalization_sleep=lambda _: None,
        )
        try:
            if message:
                with pytest.raises(ValidatorChainSourceV2Error, match=message):
                    source.read_finalized_snapshot(netuid=71, epoch_id=STATEFUL_SETTLEMENT_EPOCH_ID)
            else:
                snapshot = source.read_finalized_snapshot(netuid=71, epoch_id=STATEFUL_SETTLEMENT_EPOCH_ID)
                assert snapshot["epoch_authority"]["settlement_epoch_id"] == STATEFUL_SETTLEMENT_EPOCH_ID
            assert ("chain_getBlockHash", [0]) in archive_calls
            assert ("chain_getBlockHash", [STATEFUL_LAST_EPOCH_BLOCK]) in archive_calls
        finally:
            source.close()
            server.shutdown()
            thread.join(timeout=2)
        assert not thread.is_alive()
