"""Regenerate qualification/scoring/country_data.py from geonamescache.

The country pre-check needs the ISO-3166 table (canonical name, alpha-2,
alpha-3, continent code). Vendoring the generated table keeps the scoring
hot path free of any geo-library import and identical across every runtime;
this script is the only place the table comes from — never edit the
generated module by hand.

Usage (any environment with geonamescache installed, e.g. the gateway
venv311):

    python3 scripts/generate_country_data.py

The output file is deterministic (sorted by alpha-2 code) so regeneration
produces a stable diff.
"""

from pathlib import Path

import geonamescache
import us as us_states

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "qualification"
    / "scoring"
    / "country_data.py"
)

HEADER = '''"""ISO-3166 country and US state tables for the country pre-check.

GENERATED FILE - regenerate with scripts/generate_country_data.py; never
edit by hand. COUNTRIES rows are (canonical name, alpha-2, alpha-3,
continent code) as published by the geonamescache database; US_STATES maps
lowercase state names and uppercase USPS abbreviations to the state name,
as published by the us package.
"""

# fmt: off
COUNTRIES: tuple = (
'''

MIDDLE = ''')

US_STATES: dict = {
'''

FOOTER = '''}
# fmt: on
'''


def main() -> None:
    rows = []
    for iso2, info in sorted(geonamescache.GeonamesCache().get_countries().items()):
        name = str(info.get("name") or "").strip()
        iso3 = str(info.get("iso3") or "").strip()
        continent = str(info.get("continentcode") or "").strip()
        if not name:
            continue
        rows.append(f"    ({name!r}, {iso2!r}, {iso3!r}, {continent!r}),\n")
    states = []
    for state in sorted(us_states.states.STATES_AND_TERRITORIES, key=lambda s: s.abbr):
        states.append(f"    {state.name.lower()!r}: {state.name!r},\n")
        states.append(f"    {state.abbr.upper()!r}: {state.name!r},\n")
    OUTPUT.write_text(
        HEADER + "".join(rows) + MIDDLE + "".join(states) + FOOTER,
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT} ({len(rows)} countries, {len(states) // 2} states)")


if __name__ == "__main__":
    main()
