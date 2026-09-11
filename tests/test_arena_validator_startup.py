from types import SimpleNamespace

import pytest

from lab_arena import signing
from lab_arena import validator
from lab_arena.local_weight_signer import LocalWeightSignerError
from leadpoet_canonical import lab_arena_rewards
from leadpoet_canonical.lab_arena_rewards import LabArenaRewardError


VALID_HOTKEY = "5" + "A" * 47


class _Api:
    instances = []
    signing_document = None

    def __init__(self, base_url, *, keypair, network, netuid):
        self.base_url = base_url
        self.keypair = keypair
        self.network = network
        self.netuid = netuid
        self.__class__.instances.append(self)

    def signing_key(self):
        return dict(self.signing_document)


class _Signer:
    extrinsic_period = 8

    def __init__(self):
        self.readiness_epochs = []
        self.closed = False

    def readiness(self, epoch):
        self.readiness_epochs.append(epoch)

    def close(self):
        self.closed = True


class _Chain:
    instances = []

    def __init__(self, config, _client):
        self.config = config
        self.closed = False
        self.__class__.instances.append(self)

    def refresh_metagraph(self):
        return SimpleNamespace(hotkeys=(VALID_HOTKEY,))

    def close(self):
        self.closed = True


class _Snapshot:
    def settlement_epoch_id(self, _cutover):
        return 123


@pytest.fixture
def startup_harness(monkeypatch):
    """Patch only I/O boundaries used by validator.main."""

    arena_signer = signing.LocalSigner.generate()
    signing_document = signing.signing_key_document(arena_signer.public_key_der)
    keypair = SimpleNamespace(ss58_address=VALID_HOTKEY)
    captured = {
        "loops": [], "runner_args": [], "signer_args": [], "signers": [],
        "connect_configs": [], "wallet_args": [],
    }
    chain_module = __import__("lab_arena.chain", fromlist=["chain"])
    signer_module = __import__("lab_arena.local_weight_signer", fromlist=["signer"])

    _Api.instances = []
    _Chain.instances = []
    _Api.signing_document = signing_document

    real_signing_key_from_document = lab_arena_rewards.signing_key_from_document

    def validate_signing_key(document, expected_hash):
        # The published Finney pin is intentionally not duplicated in tests.
        # Still run the real validator against an equivalent test pin, while
        # recording the production pin selected by validator.main.
        if expected_hash == validator.FINNEY_SN71_SIGNING_KEY_HASH:
            captured["default_signing_key_hash"] = expected_hash
            return real_signing_key_from_document(document, signing_document["public_key_hash"])
        return real_signing_key_from_document(document, expected_hash)

    monkeypatch.setattr(validator, "ArenaPublicApi", _Api)
    monkeypatch.setattr(lab_arena_rewards, "signing_key_from_document", validate_signing_key)
    monkeypatch.setattr(
        validator,
        "load_local_hotkey",
        lambda args: captured["wallet_args"].append(args) or keypair,
    )
    monkeypatch.setattr(
        chain_module,
        "connect_substrate",
        lambda config: captured["connect_configs"].append(config) or object(),
    )
    monkeypatch.setattr(chain_module, "ArenaChain", _Chain)
    monkeypatch.setattr(chain_module, "finalized_epoch_snapshot", lambda _chain: _Snapshot())

    def build_signer(**kwargs):
        captured["signer_args"].append(kwargs)
        signer = _Signer()
        captured["signers"].append(signer)
        return signer

    monkeypatch.setattr(
        signer_module,
        "build_local_weight_signer",
        build_signer,
    )

    def loops(**kwargs):
        captured["loops"].append(kwargs)

    monkeypatch.setattr(validator, "run_validator_loops", loops)
    monkeypatch.setattr(validator, "signal", SimpleNamespace(
        SIGINT=2,
        SIGTERM=15,
        signal=lambda *_args: None,
    ))

    return SimpleNamespace(
        signing_document=signing_document,
        signing_key_hash=signing_document["public_key_hash"],
        captured=captured,
        chain_module=chain_module,
        signer_module=signer_module,
    )


def _clear_startup_environment(monkeypatch):
    for name in (
        "LAB_ARENA_NETWORK",
        "LAB_ARENA_NETUID",
        "LAB_ARENA_CHAIN_ENDPOINT",
        "LAB_ARENA_API_BASE_URL",
        "LAB_ARENA_SIGNING_KEY_HASH",
        "LAB_ARENA_BURN_HOTKEY",
        "LAB_ARENA_VALIDATOR_STATE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def test_advertised_finney71_check_only_uses_public_defaults_and_stops_at_readiness(
    monkeypatch, startup_harness, capsys
):
    _clear_startup_environment(monkeypatch)

    assert validator.main([
        "--netuid", "71", "--subtensor.network", "finney",
        "--wallet.name", "YOUR_WALLET", "--wallet.hotkey", "YOUR_HOTKEY",
        "--wallet.path", "/absolute/path/to/YOUR_WALLETS_DIRECTORY",
        "--check-only",
    ]) == 0

    chain = startup_harness.chain_module.ArenaChain.instances[-1]
    api = validator.ArenaPublicApi.instances[-1]
    signer = startup_harness.captured["loops"]
    assert chain.config.endpoint == "wss://entrypoint-finney.opentensor.ai:443"
    assert chain.config.network_name == "finney"
    assert chain.config.netuid == 71
    assert api.base_url == validator.FINNEY_SN71_API_BASE_URL
    assert api.network == "finney" and api.netuid == 71
    assert startup_harness.captured["signer_args"][-1]["chain_config"].endpoint == chain.config.endpoint
    assert startup_harness.captured["signers"][-1].readiness_epochs == [123]
    assert startup_harness.captured["default_signing_key_hash"] == validator.FINNEY_SN71_SIGNING_KEY_HASH
    assert signer == []
    wallet_args = startup_harness.captured["wallet_args"][-1]
    assert (wallet_args.wallet_name, wallet_args.hotkey_name, wallet_args.wallet_path) == (
        "YOUR_WALLET", "YOUR_HOTKEY", "/absolute/path/to/YOUR_WALLETS_DIRECTORY"
    )
    assert "readiness is valid" in capsys.readouterr().out


def test_endpoint_flag_wins_over_environment_and_preserves_finney_identity(
    monkeypatch, startup_harness
):
    _clear_startup_environment(monkeypatch)
    monkeypatch.setenv("LAB_ARENA_CHAIN_ENDPOINT", "wss://env.example:443")

    assert validator.main([
        "--check-only",
        "--subtensor.chain_endpoint",
        "wss://flag.example:443",
    ]) == 0

    chain = startup_harness.chain_module.ArenaChain.instances[-1]
    assert chain.config.endpoint == "wss://flag.example:443"
    assert chain.config.network_name == "finney"
    assert chain.config.netuid == 71
    signer_config = startup_harness.captured["signer_args"][-1]["chain_config"]
    assert signer_config.endpoint == "wss://flag.example:443"
    assert signer_config.network_name == "finney"


def test_environment_endpoint_is_used_when_flag_is_absent(monkeypatch, startup_harness):
    _clear_startup_environment(monkeypatch)
    monkeypatch.setenv("LAB_ARENA_CHAIN_ENDPOINT", "wss://env.example:443")

    assert validator.main(["--check-only"]) == 0

    chain = startup_harness.chain_module.ArenaChain.instances[-1]
    assert chain.config.endpoint == "wss://env.example:443"
    signer_config = startup_harness.captured["signer_args"][-1]["chain_config"]
    assert signer_config.endpoint == "wss://env.example:443"


def test_loopback_ws_endpoint_is_allowed(monkeypatch, startup_harness):
    _clear_startup_environment(monkeypatch)

    assert validator.main([
        "--check-only", "--subtensor.chain_endpoint", "ws://127.0.0.1:9944"
    ]) == 0

    chain = startup_harness.chain_module.ArenaChain.instances[-1]
    assert chain.config.endpoint == "ws://127.0.0.1:9944"


def test_explicit_service_environment_is_preserved(monkeypatch, startup_harness):
    _clear_startup_environment(monkeypatch)
    monkeypatch.setenv("LAB_ARENA_API_BASE_URL", "https://service.example///")
    monkeypatch.setenv("LAB_ARENA_SIGNING_KEY_HASH", startup_harness.signing_key_hash)

    assert validator.main(["--check-only"]) == 0

    api = validator.ArenaPublicApi.instances[-1]
    assert api.base_url == "https://service.example"
    assert api.network == "finney" and api.netuid == 71


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--check-only", "--netuid", "72"], "Arena API URL is required"),
        (["--check-only", "--subtensor.network", "test"], "Arena API URL is required"),
    ],
)
def test_custom_network_or_subnet_requires_explicit_service_settings(
    monkeypatch, startup_harness, argv, message
):
    _clear_startup_environment(monkeypatch)

    with pytest.raises(validator.ArenaValidatorError, match=message):
        validator.main(argv)


def test_custom_gateway_requires_an_explicit_signing_key_pin(monkeypatch, startup_harness):
    _clear_startup_environment(monkeypatch)
    monkeypatch.setenv("LAB_ARENA_API_BASE_URL", "https://service.example")

    with pytest.raises(validator.ArenaValidatorError, match="SIGNING_KEY_HASH"):
        validator.main(["--check-only"])


def test_wrong_signing_key_document_fails_closed(monkeypatch, startup_harness):
    _clear_startup_environment(monkeypatch)
    monkeypatch.setenv("LAB_ARENA_SIGNING_KEY_HASH", startup_harness.signing_key_hash)
    wrong_signer = signing.LocalSigner.generate()
    _Api.signing_document = signing.signing_key_document(wrong_signer.public_key_der)

    with pytest.raises(LabArenaRewardError, match="does not match"):
        validator.main(["--check-only"])


def test_network_url_must_be_passed_as_chain_endpoint(monkeypatch, startup_harness):
    _clear_startup_environment(monkeypatch)

    with pytest.raises(validator.ArenaValidatorError, match="--subtensor.chain_endpoint"):
        validator.main(["--check-only", "--subtensor.network", "wss://node.example"])


def test_unsafe_explicit_endpoint_is_rejected_before_chain_connect(
    monkeypatch, startup_harness
):
    _clear_startup_environment(monkeypatch)

    with pytest.raises(LocalWeightSignerError, match="plaintext chain RPC"):
        validator.main([
            "--check-only",
            "--subtensor.chain_endpoint",
            "ws://203.0.113.10:9944",
        ])
    assert startup_harness.captured["connect_configs"] == []


def test_once_passes_resolved_api_to_runner_factory_and_keeps_state_default(
    monkeypatch, startup_harness
):
    _clear_startup_environment(monkeypatch)
    monkeypatch.setenv("LAB_ARENA_SIGNING_KEY_HASH", startup_harness.signing_key_hash)
    monkeypatch.setattr(
        "lab_arena.wiring.build_runner_from_environment",
        lambda args, *, keypair: startup_harness.captured["runner_args"].append(
            (args, keypair)
        ) or object(),
    )

    # The controlled loop invokes the factory once, which proves that main
    # wires scoring after the weight signer has passed startup validation.
    def run_one_cycle(**kwargs):
        startup_harness.captured["loops"].append(kwargs)
        kwargs["runner_factory"]()

    monkeypatch.setattr(validator, "run_validator_loops", run_one_cycle)
    assert validator.main(["--once"]) == 0

    api = validator.ArenaPublicApi.instances[-1]
    loop = startup_harness.captured["loops"][-1]
    assert api.base_url == validator.FINNEY_SN71_API_BASE_URL
    assert startup_harness.captured["runner_args"]
    runner_args, keypair = startup_harness.captured["runner_args"][-1]
    assert runner_args.api_base_url == validator.FINNEY_SN71_API_BASE_URL
    assert runner_args.subtensor_network == "finney"
    assert runner_args.chain_endpoint == ""
    assert keypair.ss58_address == VALID_HOTKEY
    assert loop["orchestrator"].paths.root.as_posix() == "/var/lib/leadpoet/arena-validator"
