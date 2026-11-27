import pytest
from bittensor_wallet import Keypair

from lab_arena.tool_market.config import ToolMarketConfigError, load_config


def test_resale_permission_is_specific_to_the_vendor_and_account():
    miner = Keypair.create_from_uri("//Alice").ss58_address
    other = Keypair.create_from_uri("//Bob").ss58_address
    env = {"LEADPOET_TOOL_MARKET_MODE": "live", "LEADPOET_TOOL_MARKET_GAMMA_SERVICE": other,
           "LEADPOET_TOOL_MARKET_TERMS_VERSION": "tool-market-terms-v1",
           "LEADPOET_TOOL_MARKET_PUBLIC_BASE_URL": "https://gamma.example"}
    assert not load_config(env).may_supply("zerobounce", miner)
    env["LEADPOET_TOOL_MARKET_APPROVED_SUPPLIERS"] = f"zerobounce:{miner}"
    cfg = load_config(env)
    assert cfg.may_supply("zerobounce", miner)
    assert not cfg.may_supply("harvestapi", miner)
    assert not cfg.may_supply("zerobounce", other)
    for invalid in (f"unknown:{miner}", "zerobounce:", miner):
        with pytest.raises(ToolMarketConfigError):
            load_config(dict(env, LEADPOET_TOOL_MARKET_APPROVED_SUPPLIERS=invalid))
