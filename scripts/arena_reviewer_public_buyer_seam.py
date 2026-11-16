"""Apply only PR1249's public-buyer evidence correction to verified frozen source."""
import hashlib

CORRECTION_ID = "public_issuer_acquisition_target_binding_v1"
FINALITY_FILE = "qualification/scoring/lead_scorer.py"
FROZEN_IMAGE_DIGEST = "sha256:2ec6357b4517b0af50cbece24a2d130afd5f8352e548b0a77f0fa02b33ef0c63"
ORIGINAL_SHA256 = "f1874c0f8554a92d955067c7833569eb285ca63b425359df3d0c679c97e6b49c"
REPAIRED_SHA256 = "15275b9a300a24ee6026f89df39597228a6573dfec97ccda2d56c412bd1a4d82"

_OLD_0 = r'''def _stage_quote_supports_observation(observed: str, quote: str) -> bool:
'''
_NEW_0 = r'''def _stage_quote_supports_observation(
    observed: str, quote: str, *, identity_names: Sequence[str] = (),
) -> bool:
'''

_OLD_1 = r'''        reject_historical=True,
    )
'''
_NEW_1 = r'''        reject_historical=True,
    )
    if public and acquired and identity_names:
        # Ignore only a completed acquisition of an explicitly different named
        # target. Keep passive/parent/ambiguous claims and all listing guards.
        acquisition_text = text
        for match in reversed(list(_BOUND_ACQUISITION_SUBJECT_PATTERNS[3].finditer(text))):
            target = match.group("subject")
            acquisition_clause = _compact_company_name(
                re.split(r"[.!?;\n]", text[match.start():], maxsplit=1)[0]
            )
            if (
                not re.match(r"(?:the|its|our|their|this|that|it|us|them)\b", target, re.I)
                and _compact_company_name(target) not in {
                    "company", "issuer", "registrant", "business", "target",
                }
                and _acquired_stage_quote_supports_names((target,), match.group(0))
                and not any(
                    _compact_company_name(name) in acquisition_clause
                    for name in identity_names if _compact_company_name(name)
                )
            ):
                acquisition_text = (
                    acquisition_text[:match.start()] + " " * (match.end() - match.start())
                    + acquisition_text[match.end():]
                )
        acquired = _has_affirmed_stage_proof(
            acquisition_text, _ACQUIRED_STAGE_PROOF_PATTERNS, reject_historical=True,
        )
'''

_OLD_2 = r'''    lexical_support = _stage_quote_supports_observation(normalized_stage, quote)
'''
_NEW_2 = r'''    lexical_support = _stage_quote_supports_observation(
        normalized_stage, quote, identity_names=identity_names,
    )
'''

DELTAS = ((_OLD_0, _NEW_0), (_OLD_1, _NEW_1), (_OLD_2, _NEW_2),)

class PublicBuyerSeamError(ValueError):
    """The source differs from the independently verified grading file."""


def convert_public_buyer_seam(source: bytes) -> bytes:
    digest = hashlib.sha256(source).hexdigest()
    if digest == REPAIRED_SHA256:
        return source
    if digest != ORIGINAL_SHA256:
        raise PublicBuyerSeamError("unrecognized or partial public buyer source")
    text = source.decode("utf-8")
    for old, new in DELTAS:
        if text.count(old) != 1:
            raise PublicBuyerSeamError("public buyer correction anchor differs")
        text = text.replace(old, new, 1)
    result = text.encode("utf-8")
    compile(result, "frozen_public_buyer.py", "exec")
    if hashlib.sha256(result).hexdigest() != REPAIRED_SHA256:
        raise PublicBuyerSeamError("public buyer correction differs from reviewed source")
    return result
