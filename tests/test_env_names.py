"""The one-release fallback from LEADPOET_* variable names to the new names."""

from leadpoet_canonical.env_names import env_get, env_pop, env_set, legacy_name
from lab_arena.lead_orders import config as lead_orders_config
from lab_arena.tool_market import config as tool_market_config


def test_legacy_name_drops_subnet_prefix():
    assert legacy_name("TOOL_MARKET_MODE") == "LEADPOET_TOOL_MARKET_MODE"
    assert legacy_name("SUBNET71_SENTRY_DSN") == "LEADPOET_SENTRY_DSN"


def test_new_name_wins_and_legacy_is_read():
    assert env_get({"A": "new", "LEADPOET_A": "old"}, "A") == "new"
    assert env_get({"LEADPOET_A": "old"}, "A") == "old"
    assert env_get({"SUBNET71_X": ""}, "SUBNET71_X", "d") == ""
    assert env_get({"LEADPOET_SENTRY_DSN": "old"}, "SUBNET71_SENTRY_DSN") == "old"
    assert env_get({}, "A", "d") == "d"


def test_os_environ_default(monkeypatch):
    monkeypatch.delenv("ENV_NAMES_PROBE", raising=False)
    monkeypatch.setenv("LEADPOET_ENV_NAMES_PROBE", "old")
    assert env_get(None, "ENV_NAMES_PROBE") == "old"


def test_set_and_pop_cover_both_spellings():
    env = {}
    env_set(env, "SUBNET71_SENTRY_API_TOKEN", "t")
    assert env == {"SUBNET71_SENTRY_API_TOKEN": "t", "LEADPOET_SENTRY_API_TOKEN": "t"}
    env_pop(env, "SUBNET71_SENTRY_API_TOKEN")
    assert env == {}


def test_lead_orders_and_tool_market_modes_read_legacy_names():
    # A legacy-only mode is read: an invalid value proves it reached the parser.
    for module, error, name in (
        (lead_orders_config, lead_orders_config.LeadOrdersConfigError, "LEAD_ORDERS_MODE"),
        (tool_market_config, tool_market_config.ToolMarketConfigError, "TOOL_MARKET_MODE"),
    ):
        assert module.load_config({}).mode == "off"
        try:
            module.load_config({legacy_name(name): "bogus"})
        except error:
            pass
        else:
            raise AssertionError("%s legacy name was not read" % name)
        # The new name wins over the legacy one.
        assert module.load_config({name: "off", legacy_name(name): "bogus"}).mode == "off"


def test_sentry_bootstrap_reads_legacy_dsn(monkeypatch):
    from leadpoet_observability import sentry_bootstrap

    assert "LEADPOET_SENTRY_DSN" in sentry_bootstrap.INGEST_ENVIRONMENT_KEYS
    assert "SUBNET71_SENTRY_DSN" in sentry_bootstrap.INGEST_ENVIRONMENT_NAMES
    monkeypatch.delenv(sentry_bootstrap.DSN_ENV, raising=False)
    monkeypatch.setenv("LEADPOET_SENTRY_DSN", "https://k@example.invalid/1")
    assert env_get(None, sentry_bootstrap.DSN_ENV) == "https://k@example.invalid/1"
