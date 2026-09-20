from __future__ import annotations

import asyncio

import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from lab_arena import company_judgments, scorer_entrypoint
from lab_arena import scoring as arena_scoring
from qualification.scoring import company_verification, lead_scorer
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_MISMATCH,
    COMPANY_FIT_UNAVAILABLE,
    company_fit_match,
    company_fit_unavailable,
)
from qualification.scoring import linkedin_company_size
from qualification.scoring.competition import (
    scorer_breakdown_has_retryable_infrastructure_failure,
)


@pytest.fixture(autouse=True)
def _no_live_structured_profile_key(monkeypatch):
    monkeypatch.delenv("DEEPLINE_API_KEY", raising=False)


def _company(*, linkedin: str = "https://linkedin.com/company/acme") -> CompanyOutput:
    return CompanyOutput(
        company_name="Acme",
        company_website="https://acme.example.com",
        company_linkedin=linkedin,
        industry="Software",
        employee_count="11-50",
        country="United States",
        intent_signals=[
            {
                "description": "Acme launched a product.",
                "source": "news",
                "url": "https://acme.example.com/news",
                "date": "2026-08-01",
                "snippet": "Acme launched a product.",
            }
        ],
    )


def _competition_company() -> dict:
    return {
        "company_name": "Acme",
        "company_website": "https://example.com",
        "company_linkedin": "https://www.linkedin.com/company/acme",
        "industry": "Software",
        "employee_count": "11-50",
        "company_stage": "Public",
        "country": "United States",
        "state": "",
        "fit_summary": "Acme supplies workflow software.",
        "fit_evidence_urls": ["https://example.com/product"],
        "intent_signals": [{
            "matched_icp_signal": 0,
            "description": "Acme launched a product.",
            "date": "2026-08-01",
            "why_now": "The launch is a current buying signal.",
            "url": "https://example.com/news",
            "snippet": "Acme launched a product.",
        }],
    }


def _icp() -> ICPPrompt:
    return ICPPrompt(
        icp_id="current-linkedin-size",
        prompt="test",
        industry="Software",
        sub_industry="SaaS",
        employee_count="11-50",
        company_stage="",
        geography="United States",
        country="United States",
        product_service="Workflow software",
    )


def _verdict(
    *,
    observed_size: object = "1",
    size_matches: object = False,
    employee_url: str = "https://www.linkedin.com/company/acme",
) -> dict:
    return {
        "observed_company_name": "Acme",
        "observed_company_website": "https://acme.example.com/about",
        "observed_company_linkedin": "https://linkedin.com/company/acme",
        "observed_employee_count": observed_size,
        "employee_size_matches": size_matches,
        "employee_size_evidence_url": employee_url,
        "employee_size_evidence_quote": "1 employee on an old profile copy.",
        "observed_industry": "Software",
        "observed_subindustry": "SaaS",
        "industry_matches": True,
        "industry_activity_role": "supplier_operator",
        "industry_evidence_url": "https://acme.example.com/product",
        "industry_evidence_quote": "Acme provides workflow software.",
        "observed_hq_country": "United States",
        "observed_hq_state": "",
        "geography_matches": True,
        "geography_evidence_url": "https://acme.example.com/about",
        "geography_evidence_quote": "Acme is headquartered in the United States.",
        "reason": "Independent sources support the observations.",
    }


def _homepage_anchor():
    return company_fit_match(
        "homepage identity verified",
        details={
            "identity": {
                "decision": COMPANY_FIT_MATCH,
                "evidence_source": "company_homepage",
                "observed_name": "acme",
                "observed_domain": "acme.example.com",
                "observed_linkedin_slug": "acme",
            }
        },
    )


def _public_homepage_anchor():
    return company_fit_match(
        "homepage identity verified",
        details={
            "identity": {
                "decision": COMPANY_FIT_MATCH,
                "evidence_source": "company_homepage",
                "observed_name": "acme",
                "observed_domain": "example.com",
                "observed_linkedin_slug": "acme",
            }
        },
    )


def _install_exa_bodies(monkeypatch, *bodies):
    pending = list(bodies)
    calls = []

    class Response:
        status = 200

        def __init__(self, body):
            self.body = body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return self.body

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return Response(pending.pop(0))

    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(linkedin_company_size.aiohttp, "ClientSession", Session)
    return calls, pending


def _structured_company_payload(**updates):
    element = {
        "name": "Acme",
        "website": "https://www.acme.example.com/about",
        "linkedinUrl": "https://www.linkedin.com/company/acme/",
        "employeeCountRange": {"start": 11, "end": 50},
        "employeeCount": 37,
    }
    element.update(updates)
    return {
        "status": "completed",
        "result": {"data": {"status": 200, "element": element}},
    }


def test_literal_company_size_is_bounded_to_about_section():
    text = """# Acme

## About us

Cloud workflow software.

Company size 11-50 employees

## Employees at Acme

View all 89 employees
Company size 51-200 employees
"""

    assert linkedin_company_size.extract_linkedin_company_size(text) == {
        "employee_count": "11-50",
        "quote": "Company size 11-50 employees",
    }
    assert linkedin_company_size.extract_linkedin_company_size(
        "## About us\nView all 89 employees\n## Employees at Acme"
    ) is None


def test_dutch_company_size_is_canonicalized_inside_translated_about_section():
    text = """Imagine Pediatrics | LinkedIn

403 medewerkers

## Over ons

Imagine Pediatrics provides pediatric care.

Bedrijfsgrootte
501 - 1.000 medewerkers

## Medewerkers van Imagine Pediatrics

Alle 403 medewerkers weergeven
"""

    assert linkedin_company_size.extract_linkedin_company_size(text) == {
        "employee_count": "501-1,000",
        "quote": "Bedrijfsgrootte\n501 - 1.000 medewerkers",
    }


def test_portuguese_company_size_is_bounded_and_canonicalized():
    text = """Estuary | LinkedIn
Visualizar todos os 66 funcionários
## Sobre nós
Estuary provides real-time data integration.
Tamanho da empresa
11-50 funcionários
## Funcionários da Estuary
Ver 66 funcionários na empresa Estuary
"""

    assert linkedin_company_size.extract_linkedin_company_size(text) == {
        "employee_count": "11-50",
        "quote": "Tamanho da empresa\n11-50 funcionários",
    }


@pytest.mark.parametrize(
    "text",
    [
        "## Over ons\n52.304 volgers\nVacatures weergeven\n## Updates",
        "## Over ons\nAlle 403 medewerkers weergeven\n## Updates",
        "## Over ons\nBedrijfsgrootte\n403 medewerkers\n## Updates",
        (
            "## Over ons\nGeen grootteveld.\n"
            "## Medewerkers van Acme\nBedrijfsgrootte\n501 - 1.000 medewerkers"
        ),
    ],
)
def test_translated_parser_rejects_unrelated_or_non_bucket_numbers(text):
    assert linkedin_company_size.extract_linkedin_company_size(text) is None


def test_authentication_wall_is_retryable_fetch_failure(monkeypatch):
    wall = """Cadastre-se | LinkedIn
# Cadastre-se no LinkedIn
E-mail
Senha (+ de 6 caracteres)
## Entrar
E-mail ou telefone
Senha
"""
    body = {
        "statuses": [{"status": "success", "source": "crawled"}],
        "results": [
            {
                "url": "https://linkedin.com/company/acme",
                "text": wall,
            }
        ],
    }
    calls, pending = _install_exa_bodies(monkeypatch, body)

    diagnostic = {}
    assert asyncio.run(
        linkedin_company_size.fetch_current_linkedin_company_size(
            "https://linkedin.com/company/acme",
            diagnostic=diagnostic,
        )
    ) is None
    assert diagnostic == {"failure_reason": "source_blocked"}
    assert len(calls) == 1
    assert pending == []


def test_exact_noncanonical_profile_is_source_blocked(monkeypatch):
    body = {
        "statuses": [{
            "id": "https://linkedin.com/company/acme",
            "status": "error",
            "error": {
                "httpStatusCode": 409,
                "tag": "CRAWL_NON_CANONICAL",
            },
        }],
        "results": [],
    }
    calls, pending = _install_exa_bodies(monkeypatch, body)
    diagnostic = {}

    assert asyncio.run(
        linkedin_company_size.fetch_current_linkedin_company_size(
            "https://linkedin.com/company/acme",
            diagnostic=diagnostic,
        )
    ) is None
    assert diagnostic == {"failure_reason": "source_blocked"}
    assert len(calls) == 1
    assert pending == []


def test_exact_livecrawl_timeout_profile_is_source_blocked(monkeypatch):
    body = {
        "statuses": [{
            "id": "https://linkedin.com/company/acme",
            "status": "error",
            "error": {
                "httpStatusCode": 504,
                "tag": "CRAWL_LIVECRAWL_TIMEOUT",
            },
        }],
        "results": [],
    }
    calls, pending = _install_exa_bodies(monkeypatch, body)
    diagnostic = {}

    assert asyncio.run(
        linkedin_company_size.fetch_current_linkedin_company_size(
            "https://linkedin.com/company/acme",
            diagnostic=diagnostic,
        )
    ) is None
    assert diagnostic == {"failure_reason": "source_blocked"}
    assert len(calls) == 1
    assert pending == []


@pytest.mark.parametrize(
    "statuses",
    [
        [{
            "id": "https://linkedin.com/company/other",
            "status": "error",
            "error": {
                "httpStatusCode": 504,
                "tag": "CRAWL_LIVECRAWL_TIMEOUT",
            },
        }],
        [
            {
                "id": "https://linkedin.com/company/acme",
                "status": "error",
                "error": {
                    "httpStatusCode": 504,
                    "tag": "CRAWL_LIVECRAWL_TIMEOUT",
                },
            },
            {
                "id": "https://linkedin.com/company/other",
                "status": "success",
            },
        ],
        [{
            "id": "https://linkedin.com/company/acme",
            "status": "error",
            "error": {
                "httpStatusCode": 504,
                "tag": "CRAWL_OTHER",
            },
        }],
    ],
)
def test_livecrawl_timeout_requires_one_exact_matching_profile(
    monkeypatch, statuses
):
    calls, pending = _install_exa_bodies(
        monkeypatch, {"statuses": statuses, "results": []}
    )
    diagnostic = {}

    assert asyncio.run(
        linkedin_company_size.fetch_current_linkedin_company_size(
            "https://linkedin.com/company/acme",
            diagnostic=diagnostic,
        )
    ) is None
    assert diagnostic == {"failure_reason": "provider_error"}
    assert len(calls) == 1
    assert pending == []


def test_outer_http_504_remains_provider_error(monkeypatch):
    class Response:
        status = 504

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(linkedin_company_size.aiohttp, "ClientSession", Session)
    diagnostic = {}

    assert asyncio.run(
        linkedin_company_size.fetch_current_linkedin_company_size(
            "https://linkedin.com/company/acme",
            diagnostic=diagnostic,
        )
    ) is None
    assert diagnostic == {"failure_reason": "provider_error"}


@pytest.mark.parametrize(
    "wall",
    [
        """S’inscrire | LinkedIn
# S’inscrire sur LinkedIn
E-mail
Mot de passe (6 caractères ou plus)
""",
        """Registrarse | LinkedIn
# Únete a LinkedIn
Email
Contraseña (más de 6 caracteres)
""",
        """Đăng ký | LinkedIn
# Tham gia LinkedIn
Email
Mật khẩu (6 ký tự trở lên)
""",
        """Iscriviti | LinkedIn
# Iscriviti a LinkedIn
Email o telefono
Password
""",
        """الاشتراك | LinkedIn
# الانضمام إلى LinkedIn
البريد الإلكتروني أو رقم الهاتف
كلمة المرور
""",
        """Zarejestruj się | LinkedIn
# Dołącz do LinkedIn
Adres e-mail lub numer telefonu
Hasło
""",
        """Daftar | LinkedIn
# Sertai LinkedIn
E-mel atau nombor telefon
Kata laluan
""",
        """Daftar | LinkedIn
# Bergabung dengan LinkedIn
Email atau telepon
Kata sandi
""",
        """Anmelden | LinkedIn
# Mitglied bei LinkedIn werden
E-Mail-Adresse/Telefon
Passwort
""",
        """Inschrijven | LinkedIn
# Word lid van LinkedIn
E-mail of telefoonnummer
Wachtwoord
""",
        """ลงทะเบียน | LinkedIn
# เข้าร่วม LinkedIn
อีเมลหรือโทรศัพท์
รหัสผ่าน
""",
        """Εγγραφή | LinkedIn
# Εγγραφείτε στο LinkedIn
Email ή τηλέφωνο
Κωδικός πρόσβασης
""",
    ],
)
def test_observed_localized_authentication_walls_are_retryable(
    monkeypatch,
    wall,
):
    body = {
        "statuses": [{"status": "success", "source": "crawled"}],
        "results": [
            {
                "url": "https://linkedin.com/company/acme",
                "text": wall,
            }
        ],
    }
    calls, pending = _install_exa_bodies(monkeypatch, body)

    assert asyncio.run(
        linkedin_company_size.fetch_current_linkedin_company_size(
            "https://linkedin.com/company/acme"
        )
    ) is None
    assert len(calls) == 1
    assert pending == []


@pytest.mark.parametrize(
    ("wall_title", "username_field", "password_field"),
    [
        ("Iscriviti | LinkedIn", "Email o telefono", "Password"),
        (
            "الاشتراك | LinkedIn",
            "البريد الإلكتروني أو رقم الهاتف",
            "كلمة المرور",
        ),
        (
            "Zarejestruj się | LinkedIn",
            "Adres e-mail lub numer telefonu",
            "Hasło",
        ),
        ("Daftar | LinkedIn", "E-mel atau nombor telefon", "Kata laluan"),
        ("Daftar | LinkedIn", "Email atau telepon", "Kata sandi"),
        ("Anmelden | LinkedIn", "E-Mail-Adresse/Telefon", "Passwort"),
        ("Inschrijven | LinkedIn", "E-mail of telefoonnummer", "Wachtwoord"),
        ("ลงทะเบียน | LinkedIn", "อีเมลหรือโทรศัพท์", "รหัสผ่าน"),
        ("Εγγραφή | LinkedIn", "Email ή τηλέφωνο", "Κωδικός πρόσβασης"),
    ],
)
def test_company_page_with_localized_login_links_keeps_company_size(
    wall_title,
    username_field,
    password_field,
):
    text = f"""Acme | LinkedIn
## About us
Acme provides workflow software.
Company size
51-200 employees
## Updates
Use these links to manage your account:
{wall_title}
{username_field}
{password_field}
"""

    assert not linkedin_company_size._is_linkedin_access_wall(text)
    assert linkedin_company_size.extract_linkedin_company_size(text) == {
        "employee_count": "51-200",
        "quote": "Company size\n51-200 employees",
    }


@pytest.mark.parametrize(
    "blocked_text",
    [
        "Access Denied\nRequest blocked. You don't have permission to access this page.",
        (
            "Security Verification | LinkedIn\nSecurity check\n"
            "Additional verification required. Verify you are human."
        ),
        "Robot Check | LinkedIn\nVerify you are human to continue.",
    ],
)
def test_http_200_blocked_page_is_retryable_fetch_failure(
    monkeypatch,
    blocked_text,
):
    body = {
        "statuses": [{"status": "success", "source": "crawled"}],
        "results": [
            {
                "url": "https://linkedin.com/company/acme",
                "text": blocked_text,
            }
        ],
    }
    calls, pending = _install_exa_bodies(monkeypatch, body)

    diagnostic = {}
    assert asyncio.run(
        linkedin_company_size.fetch_current_linkedin_company_size(
            "https://linkedin.com/company/acme",
            diagnostic=diagnostic,
        )
    ) is None
    assert diagnostic == {"failure_reason": "source_blocked"}
    assert len(calls) == 1
    assert pending == []


def test_public_profile_footer_blocker_mentions_do_not_hide_company_size():
    text = """Acme | LinkedIn
## About us
Acme provides workflow software.
Company size
51-200 employees
## Updates
Our security product can show an Access Denied or Robot Check page.
Users may need to verify they are human.
"""

    assert not linkedin_company_size._is_linkedin_access_wall(text)
    assert linkedin_company_size.extract_linkedin_company_size(text) == {
        "employee_count": "51-200",
        "quote": "Company size\n51-200 employees",
    }


def test_exa_contents_request_is_uncached_and_bound_to_returned_url(monkeypatch):
    calls = []

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return {
                "results": [
                    {
                        "url": "https://linkedin.com/company/acme/",
                        "text": "## About\n\nCompany size:\n\n11-50 employees",
                    }
                ],
                "statuses": [{"status": "success"}],
            }

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return Response()

    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(linkedin_company_size.aiohttp, "ClientSession", Session)

    result = asyncio.run(
        linkedin_company_size.fetch_current_linkedin_company_size(
            "https://www.linkedin.com/company/acme"
        )
    )

    assert result == {
        "employee_count": "11-50",
        "quote": "Company size:\n\n11-50 employees",
        "url": "https://linkedin.com/company/acme/",
    }
    assert len(calls) == 1
    assert calls[0][0] == "https://api.exa.ai/contents"
    assert calls[0][1]["json"] == {
        "ids": ["https://www.linkedin.com/company/acme"],
        "text": {"maxCharacters": 4000},
        "maxAgeHours": 0,
        "livecrawlTimeout": 20000,
    }
    assert calls[0][1]["headers"]["x-api-key"] == "test-exa-key"


def test_invalid_requested_profile_url_never_calls_exa(monkeypatch):
    class UnexpectedSession:
        def __init__(self, **_kwargs):
            raise AssertionError("invalid profile URL must fail before transport")

    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(
        linkedin_company_size.aiohttp,
        "ClientSession",
        UnexpectedSession,
    )

    assert asyncio.run(
        linkedin_company_size.fetch_current_linkedin_company_size(
            "https://linkedin.com/in/acme"
        )
    ) is None


@pytest.mark.parametrize(
    ("mode", "expected_reason"),
    [
        ("http", "provider_error"),
        ("timeout", "provider_error"),
        ("invalid_json", "malformed_response"),
        ("invalid_body", "malformed_response"),
        ("unexpected", "unexpected_verifier_error"),
    ],
)
def test_fetch_failure_diagnostic_is_a_fixed_reason_only(
    monkeypatch, mode, expected_reason
):
    class Response:
        status = 503 if mode == "http" else 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            if mode == "invalid_json":
                raise ValueError("secret malformed payload")
            if mode == "unexpected":
                raise RuntimeError("secret unexpected payload")
            return []

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            if mode == "timeout":
                raise asyncio.TimeoutError("secret timeout detail")
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(linkedin_company_size.aiohttp, "ClientSession", Session)
    diagnostic = {}

    assert asyncio.run(
        linkedin_company_size.fetch_current_linkedin_company_size(
            "https://linkedin.com/company/acme",
            diagnostic=diagnostic,
        )
    ) is None
    assert diagnostic == {"failure_reason": expected_reason}
    assert "secret" not in repr(diagnostic)


@pytest.mark.parametrize(
    "body",
    [
        {"error": "upstream unavailable", "results": []},
        {
            "statuses": [
                {
                    "status": "error",
                    "error": {
                        "httpStatusCode": 504,
                        "tag": "CRAWL_LIVECRAWL_TIMEOUT",
                    },
                }
            ],
            "results": [],
        },
        {
            "status": "error",
            "results": [
                {
                    "url": "https://linkedin.com/company/acme",
                    "text": "## About\nCompany size 11-50 employees",
                }
            ],
        },
        {
            "statuses": [{"status": "error"}],
            "results": [
                {
                    "url": "https://linkedin.com/company/acme",
                    "text": "## About\nCompany size 11-50 employees",
                }
            ],
        },
        {
            "statuses": [{"status": "success"}],
            "results": [
                {
                    "status": "failed",
                    "error": "stale content retained",
                    "url": "https://linkedin.com/company/acme",
                    "text": "## About\nCompany size 11-50 employees",
                }
            ],
        },
        {
            "statuses": [{"status": "success"}],
            "results": [
                {
                    "url": "https://linkedin.com/company/other",
                    "text": "## About\nCompany size 11-50 employees",
                }
            ],
        },
    ],
)
def test_exa_contents_envelope_or_source_failure_is_unavailable(monkeypatch, body):
    calls = []

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return body

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, *_args, **kwargs):
            calls.append(kwargs["json"])
            return Response()

    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(linkedin_company_size.aiohttp, "ClientSession", Session)

    assert asyncio.run(
        linkedin_company_size.fetch_current_linkedin_company_size(
            "https://linkedin.com/company/acme"
        )
    ) is None
    assert calls == [{
        "ids": ["https://linkedin.com/company/acme"],
        "text": {"maxCharacters": 4000},
        "maxAgeHours": 0,
        "livecrawlTimeout": 20000,
    }]


def test_successful_exact_profile_without_company_size_is_insufficient(
    monkeypatch,
):
    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return {
                "statuses": [{"status": "success", "source": "crawled"}],
                "results": [
                    {
                        "url": "https://linkedin.com/company/acme",
                        "text": (
                            "## Over ons\nAcme bouwt workflowsoftware.\n"
                            "Website\nhttps://acme.example.com"
                        ),
                    }
                ],
            }

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(linkedin_company_size.aiohttp, "ClientSession", Session)

    diagnostic = {}
    assert asyncio.run(
        linkedin_company_size.fetch_current_linkedin_company_size(
            "https://linkedin.com/company/acme",
            diagnostic=diagnostic,
        )
    ) == {
        "outcome": "insufficient_evidence",
        "url": "https://linkedin.com/company/acme",
    }
    assert diagnostic == {}


@pytest.mark.parametrize(
    ("status_id", "expected"),
    [
        (
            "https://linkedin.com/company/acme",
            {
                "outcome": "insufficient_evidence",
                "url": "https://linkedin.com/company/acme",
            },
        ),
        ("https://linkedin.com/company/other", None),
    ],
)
def test_exact_profile_not_found_requires_requested_identity(
    monkeypatch,
    status_id,
    expected,
):
    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return {
                "statuses": [
                    {
                        "id": status_id,
                        "status": "error",
                        "error": {
                            "httpStatusCode": 404,
                            "tag": "CRAWL_NOT_FOUND",
                        },
                    }
                ],
                "results": [],
            }

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(linkedin_company_size.aiohttp, "ClientSession", Session)

    assert asyncio.run(
        linkedin_company_size.fetch_current_linkedin_company_size(
            "https://linkedin.com/company/acme"
        )
    ) == expected


def test_exact_profile_empty_text_is_insufficient(monkeypatch):
    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def json(self):
            return {
                "statuses": [{"status": "success", "source": "crawled"}],
                "results": [
                    {
                        "url": "https://linkedin.com/company/acme",
                        "text": "",
                    }
                ],
            }

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
    monkeypatch.setattr(linkedin_company_size.aiohttp, "ClientSession", Session)

    assert asyncio.run(
        linkedin_company_size.fetch_current_linkedin_company_size(
            "https://linkedin.com/company/acme"
        )
    ) == {
        "outcome": "insufficient_evidence",
        "url": "https://linkedin.com/company/acme",
    }


@pytest.mark.parametrize(
    ("sonar_size", "sonar_matches", "current_size", "expected"),
    [
        ("1", False, "11-50", COMPANY_FIT_MATCH),
        ("11-50", True, "51-200", COMPANY_FIT_MISMATCH),
        ("1-10", True, "11-50", COMPANY_FIT_MATCH),
        (None, None, "51-200", COMPANY_FIT_MISMATCH),
    ],
)
def test_current_profile_replaces_stale_linkedin_match_or_mismatch(
    monkeypatch,
    sonar_size,
    sonar_matches,
    current_size,
    expected,
):
    fetches = []

    async def provider(**_kwargs):
        return _verdict(
            observed_size=sonar_size,
            size_matches=sonar_matches,
        ), ""

    async def fetch(url, **_kwargs):
        fetches.append(url)
        return {
            "employee_count": current_size,
            "url": url,
            "quote": f"Company size {current_size} employees",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", fetch)

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_homepage_anchor(),
        )
    )

    assert result.decision == expected
    assert result.details["dimension_decisions"]["employee_size"] == expected
    assert result.details["provider_observations"]["observed_employee_count"] == (
        current_size
    )
    assert result.details["dimension_evidence"]["employee_size"] == {
        "url": "https://www.linkedin.com/company/acme",
        "quote": f"Company size {current_size} employees",
    }
    assert fetches == ["https://www.linkedin.com/company/acme"]


@pytest.mark.parametrize(
    ("employee_range", "expected"),
    [
        ({"start": 0, "end": 1}, "0-1"),
        ({"start": 2, "end": 10}, "2-10"),
        ({"start": 11, "end": 50}, "11-50"),
        ({"start": 51, "end": 200}, "51-200"),
        ({"start": 201, "end": 500}, "201-500"),
        ({"start": 501, "end": 1_000}, "501-1,000"),
        ({"start": 1_001, "end": 5_000}, "1,001-5,000"),
        ({"start": 5_001, "end": 10_000}, "5,001-10,000"),
        ({"start": 10_001, "end": None}, "10,001+"),
    ],
)
def test_structured_company_projects_only_canonical_employee_ranges(
    employee_range,
    expected,
):
    payload = _structured_company_payload(employeeCountRange=employee_range)

    assert linkedin_company_size.project_structured_linkedin_company_size(
        "acme.example.com",
        "https://www.linkedin.com/company/acme",
        payload,
    ) == {
        "employee_count": expected,
        "provider": "harvestapi_get_company",
        "source_field": "employeeCountRange",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://acme.example.com/",
    }


def test_structured_company_projects_exact_identity_public_company_metadata():
    payload = _structured_company_payload(companyType="Public Company")

    assert linkedin_company_size.project_structured_linkedin_public_company(
        "acme.example.com",
        "https://www.linkedin.com/company/acme",
        payload,
    ) == {
        "company_type": "Public Company",
        "provider": "harvestapi_get_company",
        "source_field": "companyType",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://acme.example.com/",
    }


def test_structured_company_projects_only_explicit_bound_private_metadata():
    payload = _structured_company_payload(companyType="Privately Held")

    assert linkedin_company_size.project_structured_linkedin_company_type(
        "acme.example.com",
        "https://www.linkedin.com/company/acme",
        payload,
    ) == {
        "company_type": "Privately Held",
        "provider": "harvestapi_get_company",
        "source_field": "companyType",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://acme.example.com/",
    }
    assert linkedin_company_size.project_structured_linkedin_company_type(
        "other.example.com",
        "https://www.linkedin.com/company/acme",
        payload,
    ) is None
    assert linkedin_company_size.project_structured_linkedin_company_type(
        "acme.example.com",
        "https://www.linkedin.com/company/acme",
        _structured_company_payload(companyType="Partnership"),
    ) is None


def test_structured_company_accepts_child_host_of_requested_registrable_root():
    payload = _structured_company_payload(
        website="https://news.microsoft.com/",
        linkedinUrl="https://www.linkedin.com/company/microsoft/",
        employeeCountRange={"start": 10_001, "end": None},
        companyType="Public Company",
    )

    assert linkedin_company_size.project_structured_linkedin_company_size(
        "microsoft.com",
        "https://www.linkedin.com/company/microsoft",
        payload,
    ) == {
        "employee_count": "10,001+",
        "provider": "harvestapi_get_company",
        "source_field": "employeeCountRange",
        "url": "https://www.linkedin.com/company/microsoft",
        "website": "https://microsoft.com/",
    }
    assert linkedin_company_size.project_structured_linkedin_public_company(
        "microsoft.com",
        "https://www.linkedin.com/company/microsoft",
        payload,
    ) == {
        "company_type": "Public Company",
        "provider": "harvestapi_get_company",
        "source_field": "companyType",
        "url": "https://www.linkedin.com/company/microsoft",
        "website": "https://microsoft.com/",
    }


@pytest.mark.parametrize(
    ("requested_domain", "observed_website"),
    [
        ("microsoft.com", "https://micros0ft.com/"),
        ("microsoft.com", "https://microsoft.com.example.net/"),
        ("careers.microsoft.com", "https://news.microsoft.com/"),
        ("alpha.github.io", "https://beta.github.io/"),
    ],
)
def test_structured_company_rejects_lookalikes_suffixes_and_sibling_hosts(
    requested_domain,
    observed_website,
):
    payload = _structured_company_payload(
        website=observed_website,
        companyType="Public Company",
    )

    assert linkedin_company_size.project_structured_linkedin_company_size(
        requested_domain,
        "https://www.linkedin.com/company/acme",
        payload,
    ) is None
    assert linkedin_company_size.project_structured_linkedin_public_company(
        requested_domain,
        "https://www.linkedin.com/company/acme",
        payload,
    ) is None


def test_structured_company_keeps_private_suffix_tenant_boundary():
    payload = _structured_company_payload(
        website="https://news.alpha.github.io/",
        companyType="Public Company",
    )

    assert linkedin_company_size.project_structured_linkedin_company_size(
        "alpha.github.io",
        "https://www.linkedin.com/company/acme",
        payload,
    ) is not None
    assert linkedin_company_size.project_structured_linkedin_public_company(
        "alpha.github.io",
        "https://www.linkedin.com/company/acme",
        payload,
    ) is not None


def test_structured_company_child_host_still_requires_exact_linkedin_slug():
    payload = _structured_company_payload(
        website="https://news.microsoft.com/",
        linkedinUrl="https://www.linkedin.com/company/not-microsoft/",
        companyType="Public Company",
    )

    assert linkedin_company_size.project_structured_linkedin_company_size(
        "microsoft.com",
        "https://www.linkedin.com/company/microsoft",
        payload,
    ) is None
    assert linkedin_company_size.project_structured_linkedin_public_company(
        "microsoft.com",
        "https://www.linkedin.com/company/microsoft",
        payload,
    ) is None


@pytest.mark.parametrize(
    "payload",
    [
        _structured_company_payload(companyType="Privately Held"),
        _structured_company_payload(companyType="public company"),
        _structured_company_payload(
            companyType="Public Company",
            website="https://other.example.com",
        ),
        _structured_company_payload(
            companyType="Public Company",
            linkedinUrl="https://www.linkedin.com/company/other",
        ),
        {
            "status": "completed",
            "result": {
                "error": "provider failed",
                "data": _structured_company_payload(
                    companyType="Public Company"
                )["result"]["data"],
            },
        },
    ],
)
def test_structured_public_company_rejects_labels_identity_conflicts_and_errors(
    payload,
):
    assert linkedin_company_size.project_structured_linkedin_public_company(
        "acme.example.com",
        "https://www.linkedin.com/company/acme",
        payload,
    ) is None


@pytest.mark.parametrize(
    "payload",
    [
        _structured_company_payload(website="https://other.example.com"),
        _structured_company_payload(
            linkedinUrl="https://www.linkedin.com/company/other"
        ),
        _structured_company_payload(
            linkedinUrl="https://www.linkedin.com/company/acme?trk=other"
        ),
        _structured_company_payload(employeeCountRange={"start": 1, "end": 10}),
        _structured_company_payload(employeeCountRange={"start": True, "end": 50}),
        _structured_company_payload(employeeCountRange=None, employeeCount=37),
        {
            "status": "completed",
            "result": {
                "data": {
                    "status": "200",
                    "element": _structured_company_payload()["result"]["data"][
                        "element"
                    ],
                }
            },
        },
        {
            "status": "completed",
            "result": {
                "data": {
                    "status": 500,
                    "element": _structured_company_payload()["result"]["data"][
                        "element"
                    ],
                }
            },
        },
        {
            "status": "completed",
            "result": {
                "data": {
                    "status": True,
                    "element": _structured_company_payload()["result"]["data"][
                        "element"
                    ],
                }
            },
        },
        {
            "status": "completed",
            "result": {
                "error": "provider failed",
                "data": _structured_company_payload()["result"]["data"],
            },
        },
        {
            "status": 503,
            "result": {
                "data": _structured_company_payload()["result"]["data"],
            },
        },
    ],
)
def test_structured_company_rejects_wrong_identity_status_or_range(payload):
    assert linkedin_company_size.project_structured_linkedin_company_size(
        "acme.example.com",
        "https://www.linkedin.com/company/acme",
        payload,
    ) is None


def test_structured_company_fetch_uses_one_bounded_approved_operation(monkeypatch):
    calls, pending = _install_exa_bodies(
        monkeypatch,
        _structured_company_payload(companyType="Public Company"),
    )
    monkeypatch.setenv("DEEPLINE_API_KEY", "test-deepline-key")
    public_company_evidence = {}

    result = asyncio.run(
        linkedin_company_size.fetch_structured_linkedin_company_size(
            "acme.example.com",
            "https://www.linkedin.com/company/acme",
            public_company_evidence=public_company_evidence,
        )
    )

    assert result == {
        "employee_count": "11-50",
        "provider": "harvestapi_get_company",
        "source_field": "employeeCountRange",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://acme.example.com/",
    }
    assert public_company_evidence == {
        "company_type": "Public Company",
        "provider": "harvestapi_get_company",
        "source_field": "companyType",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://acme.example.com/",
    }
    assert pending == []
    assert len(calls) == 1
    url, request = calls[0]
    assert url.endswith("/harvestapi_get_company/execute")
    assert request["json"] == {
        "payload": {"url": "https://www.linkedin.com/company/acme"}
    }


def test_structured_company_identity_projects_only_exact_main_profile():
    payload = _structured_company_payload(
        name="Truist",
        website="http://www.truist.com",
        linkedinUrl=(
            "https://www.linkedin.com/company/truistfinancialcorporation/"
        ),
    )
    payload["result"]["data"]["element"]["similarOrganizations"] = [{
        "name": "Wrong Parent",
        "website": "https://truist.com",
        "linkedinUrl": (
            "https://www.linkedin.com/company/truistfinancialcorporation/"
        ),
    }]

    assert linkedin_company_size.project_structured_linkedin_company_identity(
        "truist.com",
        "https://www.linkedin.com/company/truistfinancialcorporation",
        payload,
    ) == {
        "name": "Truist",
        "provider": "harvestapi_get_company",
        "source_field": "name",
        "url": (
            "https://www.linkedin.com/company/truistfinancialcorporation"
        ),
        "website": "https://truist.com/",
    }

    payload["result"]["data"]["element"].update(
        name="Wrong Parent",
        website="https://wrong.example",
    )
    assert linkedin_company_size.project_structured_linkedin_company_identity(
        "truist.com",
        "https://www.linkedin.com/company/truistfinancialcorporation",
        payload,
    ) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "Other Bank"),
        ("website", "https://other.example/"),
        ("url", "https://www.linkedin.com/company/other-bank"),
        ("provider", "other_provider"),
        ("source_field", "displayName"),
    ],
)
def test_structured_profile_alias_rejects_any_identity_mismatch(field, value):
    company = _company(linkedin="").model_copy(update={
        "company_name": "Truist",
        "company_website": "https://truist.com",
    })
    web_identity = lead_scorer._web_identity_receipt(
        company,
        {
            "observed_company_name": "Truist Financial Corporation",
            "observed_company_website": "https://truist.com",
            "observed_company_linkedin": (
                "https://www.linkedin.com/company/truistfinancialcorporation"
            ),
        },
        company_quality=True,
    )
    structured = {
        "name": "Truist",
        "provider": "harvestapi_get_company",
        "source_field": "name",
        "url": "https://www.linkedin.com/company/truistfinancialcorporation",
        "website": "https://truist.com/",
    }
    structured[field] = value

    assert not lead_scorer._structured_profile_alias_identity_receipt(
        company,
        web_identity,
        structured,
        "truist.com",
        company_quality=True,
    )


def test_structured_profile_alias_binds_brand_without_submitted_linkedin():
    company = _company(linkedin="").model_copy(update={
        "company_name": "Truist",
        "company_website": "https://truist.com",
    })
    web_identity = lead_scorer._web_identity_receipt(
        company,
        {
            "observed_company_name": "Truist Financial Corporation",
            "observed_company_website": "https://truist.com",
            "observed_company_linkedin": (
                "https://www.linkedin.com/company/truistfinancialcorporation"
            ),
        },
        company_quality=True,
    )
    structured = {
        "name": "Truist",
        "provider": "harvestapi_get_company",
        "source_field": "name",
        "url": "https://www.linkedin.com/company/truistfinancialcorporation",
        "website": "https://truist.com/",
    }

    assert lead_scorer._alias_unresolved_structured_profile_lookup(
        web_identity,
        "truist.com",
    )["linkedin_company_slug"] == "truistfinancialcorporation"
    resolved = lead_scorer._structured_profile_alias_identity_receipt(
        company,
        web_identity,
        structured,
        "truist.com",
        company_quality=True,
    )
    assert resolved["decision"] == COMPANY_FIT_MATCH
    assert resolved["reason_code"] == "structured_profile_alias_verified"
    assert not lead_scorer._structured_profile_alias_identity_receipt(
        company,
        web_identity,
        None,
        "truist.com",
        company_quality=True,
    )


@pytest.mark.parametrize(
    ("body", "expected_reason"),
    [
        (
            _structured_company_payload(website="https://other.example.com"),
            "source_blocked",
        ),
        (
            _structured_company_payload(employeeCountRange=None),
            "source_blocked",
        ),
        (
            {
                "status": "completed",
                "result": {
                    "data": {
                        "status": "200",
                        "element": _structured_company_payload()["result"]["data"][
                            "element"
                        ],
                    }
                },
            },
            "malformed_response",
        ),
        (
            {
                "status": 503,
                "result": {
                    "data": _structured_company_payload()["result"]["data"],
                },
            },
            "provider_error",
        ),
    ],
)
def test_structured_fetch_classifies_profile_local_and_provider_failures(
    monkeypatch,
    body,
    expected_reason,
):
    _install_exa_bodies(monkeypatch, body)
    monkeypatch.setenv("DEEPLINE_API_KEY", "test-deepline-key")
    diagnostic = {}

    result = asyncio.run(
        linkedin_company_size.fetch_structured_linkedin_company_size(
            "acme.example.com",
            "https://www.linkedin.com/company/acme",
            diagnostic=diagnostic,
        )
    )

    assert result is None
    assert diagnostic == {"failure_reason": expected_reason}


def test_structured_company_fetch_without_key_does_not_call_provider(monkeypatch):
    monkeypatch.delenv("DEEPLINE_API_KEY", raising=False)

    class UnexpectedSession:
        def __init__(self, **_kwargs):
            raise AssertionError("missing optional key must preserve Exa result")

    monkeypatch.setattr(
        linkedin_company_size.aiohttp,
        "ClientSession",
        UnexpectedSession,
    )
    assert asyncio.run(
        linkedin_company_size.fetch_structured_linkedin_company_size(
            "acme.example.com",
            "https://www.linkedin.com/company/acme",
        )
    ) is None


def test_login_wall_failure_allows_existing_retry_to_use_translated_profile(
    monkeypatch,
):
    provider_calls = []
    wall = """Cadastre-se | LinkedIn
# Cadastre-se no LinkedIn
E-mail
Senha (+ de 6 caracteres)
"""
    dutch_profile = """Acme | LinkedIn
## Over ons
Acme provides workflow software.
Bedrijfsgrootte
501 - 1.000 medewerkers
## Medewerkers van Acme
Alle 403 medewerkers weergeven
"""
    bodies = [
        {
            "statuses": [{"status": "success", "source": "crawled"}],
            "results": [
                {
                    "url": "https://linkedin.com/company/acme",
                    "text": wall,
                }
            ],
        },
        {
            "statuses": [{"status": "success", "source": "crawled"}],
            "results": [
                {
                    "url": "https://linkedin.com/company/acme",
                    "text": dutch_profile,
                }
            ],
        },
    ]
    exa_calls, pending = _install_exa_bodies(monkeypatch, *bodies)

    async def provider(**kwargs):
        provider_calls.append(kwargs["telemetry_purpose"])
        return _verdict(observed_size=None, size_matches=None), ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)

    first = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp().model_copy(update={"employee_count": "501-1,000"}),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_homepage_anchor(),
        )
    )
    retryable = scorer_breakdown_has_retryable_infrastructure_failure(
        {"verifier_gate_receipts": [first.receipt("company_fit")]}
    )
    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp().model_copy(update={"employee_count": "501-1,000"}),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_homepage_anchor(),
        )
    )

    assert first.decision == COMPANY_FIT_UNAVAILABLE
    assert first.details["failure_class"] == "employee_size_verification_failed"
    assert retryable
    assert result.decision == COMPANY_FIT_MATCH
    assert result.details["dimension_decisions"]["employee_size"] == (
        COMPANY_FIT_MATCH
    )
    assert result.details["provider_observations"]["observed_employee_count"] == (
        "501-1,000"
    )
    assert result.details["dimension_evidence"]["employee_size"] == {
        "url": "https://linkedin.com/company/acme",
        "quote": "Bedrijfsgrootte\n501 - 1.000 medewerkers",
    }
    assert provider_calls == [
        "lead_scorer_reverify",
        "lead_scorer_reverify_schema_repair",
        "lead_scorer_reverify",
    ]
    assert len(exa_calls) == 2
    assert pending == []


def test_failed_refresh_clears_stale_size_after_bounded_schema_retry(monkeypatch):
    provider_calls = []
    original_verdict = _verdict()
    wall = """Cadastre-se | LinkedIn
# Cadastre-se no LinkedIn
E-mail
Senha (+ de 6 caracteres)
"""
    body = {
        "statuses": [{"status": "success", "source": "crawled"}],
        "results": [
            {
                "url": "https://linkedin.com/company/acme",
                "text": wall,
            }
        ],
    }
    exa_calls, pending = _install_exa_bodies(monkeypatch, body)

    async def provider(**kwargs):
        provider_calls.append(kwargs["telemetry_purpose"])
        return original_verdict, ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_homepage_anchor(),
        )
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["dimension_decisions"]["employee_size"] == (
        COMPANY_FIT_UNAVAILABLE
    )
    assert result.details["provider_observations"]["observed_employee_count"] is None
    assert result.details["dimension_evidence"]["employee_size"] == {
        "url": "",
        "quote": "",
    }
    assert provider_calls == [
        "lead_scorer_reverify",
        "lead_scorer_reverify_schema_repair",
    ]
    assert len(exa_calls) == 1
    assert pending == []
    assert result.details["failure_class"] == (
        "employee_size_verification_failed"
    )
    assert result.details["failure_reason_code"] == "source_blocked"
    assert original_verdict["observed_employee_count"] == "1"
    assert original_verdict["employee_size_matches"] is False


@pytest.mark.parametrize("repair_kind", ["invalid_citation", "unbound_identity"])
def test_actual_profile_failure_survives_an_unusable_repair(
    monkeypatch,
    repair_kind,
):
    initial = _verdict()
    repaired = _verdict()
    if repair_kind == "invalid_citation":
        repaired["employee_size_evidence_url"] = (
            "https://www.linkedin.com/in/acme-employee"
        )
    else:
        repaired["observed_company_name"] = "Acme Holdings"
    verdicts = [initial, repaired]
    fetches = []

    async def provider(**_kwargs):
        return verdicts.pop(0), ""

    async def fetch(url, **_kwargs):
        fetches.append(url)
        return None

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", fetch)

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(linkedin=""),
            _icp(),
            require_company_fit_dimensions=True,
        )
    )

    assert fetches == ["https://www.linkedin.com/company/acme"]
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["failure_class"] == (
        "employee_size_verification_failed"
    )


@pytest.mark.parametrize(
    "employee_url",
    [
        "https://www.linkedin.com/company/dewaofficial",
        "https://www.linkedin.com/in/dewa-http-careers-dewa-gov-ae-0a579332",
    ],
)
def test_same_domain_dewa_alias_is_insufficient_without_a_profile_failure(
    monkeypatch,
    employee_url,
):
    company = _company(linkedin="").model_copy(
        update={
            "company_name": "Dubai Electricity and Water Authority",
            "company_website": "https://dewa.gov.ae/",
        }
    )
    verdict = _verdict(
        observed_size="5,001-10,000",
        size_matches=True,
        employee_url=employee_url,
    )
    verdict.update(
        observed_company_name="Dubai Electricity & Water Authority - DEWA",
        observed_company_website="https://dewa.gov.ae",
        observed_company_linkedin=(
            "https://www.linkedin.com/company/dewaofficial"
        ),
        employee_size_evidence_quote="Company size 5,001-10,000 employees",
        observed_company_stage="",
        stage_matches=None,
        stage_evidence_url="",
        stage_evidence_quote="",
    )
    provider_calls = []

    async def provider(**kwargs):
        provider_calls.append(kwargs["telemetry_purpose"])
        return verdict, ""

    async def unexpected_fetch(_url):
        raise AssertionError("an unbound profile must not be fetched")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer,
        "fetch_current_linkedin_company_size",
        unexpected_fetch,
    )

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            company,
            _icp().model_copy(update={"company_stage": "Series A"}),
            require_company_fit_dimensions=True,
        )
    )
    receipt = result.receipt("company_fit")

    assert provider_calls == [
        "lead_scorer_reverify",
        "lead_scorer_reverify_schema_repair",
    ]
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["failure_class"] == "insufficient_fit_evidence"
    assert result.details["identity_decision"] == COMPANY_FIT_UNAVAILABLE
    assert result.details["dimension_decisions"]["employee_size"] == (
        COMPANY_FIT_UNAVAILABLE
    )
    assert result.details["dimension_decisions"]["stage"] == (
        COMPANY_FIT_UNAVAILABLE
    )
    assert not scorer_breakdown_has_retryable_infrastructure_failure(
        {"verifier_gate_receipts": [receipt]}
    )


@pytest.mark.parametrize(
    ("homepage_reason", "expected_retryable"),
    [
        ("website returned HTTP 502", True),
        ("website returned HTTP 404", False),
        (
            "homepage identity evidence unavailable: LinkedIn company binding not found",
            False,
        ),
    ],
)
def test_same_domain_unproven_alias_preserves_homepage_failure_classification(
    monkeypatch,
    homepage_reason,
    expected_retryable,
):
    company = _company(linkedin="").model_copy(
        update={
            "company_name": "Dubai Electricity and Water Authority",
            "company_website": "https://dewa.gov.ae/",
        }
    )
    icp = _icp().model_copy(update={"company_stage": "Series A"})
    verdict = _verdict(
        observed_size="5,001-10,000",
        size_matches=True,
        employee_url="https://www.linkedin.com/company/dewaofficial",
    )
    verdict.update(
        observed_company_name="Dubai Electricity & Water Authority - DEWA",
        observed_company_website="https://dewa.gov.ae",
        observed_company_linkedin=(
            "https://www.linkedin.com/company/dewaofficial"
        ),
        employee_size_evidence_quote="Company size 5,001-10,000 employees",
        observed_company_stage="",
        stage_matches=None,
        stage_evidence_url="",
        stage_evidence_quote="",
    )

    async def prechecks(*_args, **_kwargs):
        return company_fit_match("prechecks passed")

    async def homepage(*_args, **_kwargs):
        return company_fit_unavailable(homepage_reason)

    async def provider(**_kwargs):
        return dict(verdict), ""

    async def unexpected_fetch(_url):
        raise AssertionError("an unbound profile must not be fetched")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(
        lead_scorer,
        "_request_company_reverify_json",
        provider,
    )
    monkeypatch.setattr(
        lead_scorer,
        "fetch_current_linkedin_company_size",
        unexpected_fetch,
    )

    result = asyncio.run(
        lead_scorer.score_company_competition_intent(
            company,
            icp,
            0.0,
            0.0,
            set(),
        )
    )
    breakdown = result.model_dump(mode="json")
    receipt = breakdown["verifier_gate_receipts"][0]

    assert result.final_score == 0.0
    assert receipt["decision"] == COMPANY_FIT_UNAVAILABLE
    assert receipt["dimension_evidence"]["identity"][
        "homepage_identity_reason"
    ] == homepage_reason
    assert scorer_breakdown_has_retryable_infrastructure_failure(
        breakdown
    ) is expected_retryable
    assert (receipt.get("failure_class") == "insufficient_fit_evidence") is (
        not expected_retryable
    )


@pytest.mark.parametrize("homepage_raises", [False, True])
def test_resolved_web_identity_does_not_make_missing_stage_retryable(
    monkeypatch, homepage_raises
):
    company = _company()
    icp = _icp().model_copy(update={"company_stage": "Series A"})
    verdict = _verdict(observed_size="11-50", size_matches=True)
    verdict.update(
        observed_company_stage="",
        stage_matches=None,
        stage_evidence_url="",
        stage_evidence_quote="",
    )
    provider_calls = []
    profile_fetches = []

    async def prechecks(*_args, **_kwargs):
        return company_fit_match("prechecks passed")

    async def homepage(*_args, **_kwargs):
        if homepage_raises:
            raise lead_scorer.aiohttp.ClientError("private transport detail")
        return company_fit_unavailable("website returned HTTP 502")

    async def provider(**kwargs):
        provider_calls.append(kwargs["telemetry_purpose"])
        return dict(verdict), ""

    async def profile(url, **_kwargs):
        profile_fetches.append(url)
        return {
            "employee_count": "11-50",
            "quote": "Company size 11-50 employees",
            "url": "https://www.linkedin.com/company/acme",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(
        lead_scorer,
        "_request_company_reverify_json",
        provider,
    )
    monkeypatch.setattr(
        lead_scorer,
        "fetch_current_linkedin_company_size",
        profile,
    )

    result = asyncio.run(
        lead_scorer.score_company_competition_intent(
            company,
            icp,
            0.0,
            0.0,
            set(),
        )
    )
    breakdown = result.model_dump(mode="json")
    receipt = breakdown["verifier_gate_receipts"][0]

    assert provider_calls == [
        "lead_scorer_reverify",
        "lead_scorer_reverify_schema_repair",
    ]
    assert profile_fetches == ["https://www.linkedin.com/company/acme"]
    assert result.final_score == 0.0
    assert receipt["decision"] == COMPANY_FIT_UNAVAILABLE
    assert receipt["company_fit_dimensions"]["identity"] == COMPANY_FIT_MATCH
    assert receipt["company_fit_dimensions"]["employee_size"] == (
        COMPANY_FIT_MATCH
    )
    assert receipt["company_fit_dimensions"]["stage"] == (
        COMPANY_FIT_UNAVAILABLE
    )
    homepage_reason = receipt["dimension_evidence"]["identity"][
        "homepage_identity_reason"
    ]
    assert (
        homepage_reason.startswith("company identity provider error:")
        if homepage_raises
        else homepage_reason == "website returned HTTP 502"
    )
    assert receipt["failure_class"] == "insufficient_fit_evidence"
    assert "failure_reason_code" not in receipt
    assert "failure_reason_code" not in receipt["dimension_evidence"]["identity"]
    assert not scorer_breakdown_has_retryable_infrastructure_failure(breakdown)


def test_identity_insufficient_path_does_not_hide_malformed_evidence():
    complete_alias_receipt = {
        "decision": "unavailable",
        "reason_code": "identity_not_proven",
        "evidence_source": "company_web_reverification",
        "submitted_name": "dubaielectricityandwaterauthority",
        "submitted_domain": "dewa.gov.ae",
        "submitted_linkedin_slug": "",
        "observed_name": "dubaielectricitywaterauthoritydewa",
        "observed_domain": "dewa.gov.ae",
        "observed_linkedin_slug": "dewaofficial",
    }

    assert not lead_scorer._has_explicitly_unproven_fit_dimensions(
        {},
        ("identity", "employee_size"),
        linkedin_refresh_outcome="insufficient_evidence",
        identity_receipt={
            "decision": "unavailable",
            "reason_code": "identity_observation_type_invalid",
        },
    )
    assert not lead_scorer._has_explicitly_unproven_fit_dimensions(
        {},
        ("identity", "employee_size", "industry"),
        linkedin_refresh_outcome="insufficient_evidence",
        identity_receipt=complete_alias_receipt,
    )


@pytest.mark.parametrize("observed_linkedin", ["", "https://linkedin.com/company/acme's"])
def test_unproven_identity_without_usable_linkedin_is_company_failure(
    monkeypatch, observed_linkedin,
):
    """The September 13 Lowe's shape is missing proof, not a broken judge."""
    company = _company(linkedin="")
    verdict = _verdict(observed_size=None, size_matches=None, employee_url="")
    verdict.update(
        observed_company_name="Acme Companies, Inc.",
        observed_company_linkedin=observed_linkedin,
        employee_size_evidence_quote="",
        observed_company_stage="Public",
        stage_matches=True,
        stage_evidence_url="https://linkedin.com/company/acme's",
        stage_evidence_quote="Type Public Company.",
    )
    calls = []

    async def provider(**kwargs):
        calls.append(kwargs["telemetry_purpose"])
        return dict(verdict), ""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    result = asyncio.run(lead_scorer._llm_reverify_company(
        company, _icp().model_copy(update={"company_stage": "Public"}),
        require_company_fit_dimensions=True,
    ))
    assert calls == ["lead_scorer_reverify", "lead_scorer_reverify_schema_repair"]
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["identity_decision"] == COMPANY_FIT_UNAVAILABLE
    assert result.details["failure_class"] == "insufficient_fit_evidence"
    assert not scorer_breakdown_has_retryable_infrastructure_failure(
        {"verifier_gate_receipts": [result.receipt("company_fit")]}
    )


@pytest.mark.parametrize("broken_judge", [False, True])
def test_homepage_failure_exhausts_only_company_unless_judge_is_broken(
    monkeypatch, broken_judge,
):
    from lab_arena import scoring

    company = _company(linkedin="")
    verdict = _verdict(observed_size=None, size_matches=None, employee_url="")
    verdict.update(
        observed_company_name="Acme Companies, Inc.",
        observed_company_linkedin="",
        employee_size_evidence_quote="",
    )
    calls = []

    async def prechecks(*_args, **_kwargs):
        return company_fit_match("prechecks passed")

    async def homepage(*_args, **_kwargs):
        return company_fit_unavailable(
            "website returned HTTP 502",
            details={"failure_reason_code": "source_blocked"},
        )

    async def provider(**kwargs):
        calls.append(kwargs["telemetry_purpose"])
        if broken_judge and kwargs["telemetry_purpose"].endswith("schema_repair"):
            kwargs["diagnostic"]["failure_reason"] = "malformed_response"
            return None, "malformed response"
        return dict(verdict), ""

    async def scorer(companies, _icp_doc, _reference):
        rows = []
        for item in companies:
            if item["company_name"] == "Other":
                rows.append({"final_score": 77.0, "failure_reason": ""})
            else:
                result = await lead_scorer.score_company_competition_intent(
                    company, _icp(), 0.0, 0.0, set(),
                )
                rows.append(result.model_dump(mode="json"))
        return rows

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    kwargs = dict(
        icp={"max_companies": 2, "employee_count": "11-50"},
        companies=[
            {"company_name": "Acme", "employee_count": "11-50"},
            {"company_name": "Other", "employee_count": "11-50"},
        ],
        scorer=scorer,
    )
    if broken_judge:
        with pytest.raises(scoring.ScoringError) as error:
            scoring.score_work_item({"scored_run_id": "source-failure"}, **kwargs)
        assert error.value.failure_reason == "malformed_response"
    else:
        rows = scoring.score_work_item({"scored_run_id": "source-failure"}, **kwargs)
        assert [row["final_score"] for row in rows] == [0.0, 77.0]
        assert rows[0]["verifier_gate_receipts"][0]["failure_class"] == (
            "company_verification_exhausted"
        )
    assert len(calls) == 6  # two existing judge calls in each of three attempts


def _complete_same_slug_name_alias_receipt(**updates):
    receipt = {
        "decision": "unavailable",
        "reason_code": "identity_name_alias_unresolved",
        "evidence_source": "company_web_reverification",
        "submitted_name": "acmeinternationalholdings",
        "submitted_domain": "acme.example.com",
        "submitted_linkedin_slug": "acme",
        "observed_name": "acme",
        "observed_domain": "acme.example.com",
        "observed_linkedin_slug": "acme",
    }
    receipt.update(updates)
    return receipt


def test_complete_same_slug_name_alias_is_explicitly_unproven_identity():
    receipt = _complete_same_slug_name_alias_receipt()

    assert lead_scorer._has_explicitly_unproven_fit_dimensions(
        {},
        ("identity",),
        identity_receipt=receipt,
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"submitted_linkedin_slug": ""},
        {"observed_linkedin_slug": "other"},
        {"observed_domain": "other.example.com"},
        {"observed_name": ""},
        {"evidence_source": "company_homepage"},
        {"submitted_name": "acme"},
    ],
)
def test_incomplete_or_mismatched_same_slug_name_alias_stays_retryable(updates):
    assert not lead_scorer._has_explicitly_unproven_fit_dimensions(
        {},
        ("identity",),
        identity_receipt=_complete_same_slug_name_alias_receipt(**updates),
    )


@pytest.mark.parametrize(
    ("homepage_reason", "expected_retryable"),
    [
        ("website returned HTTP 404", False),
        ("website returned HTTP 502", True),
    ],
)
def test_same_slug_unproven_name_alias_preserves_homepage_failure_classification(
    monkeypatch,
    homepage_reason,
    expected_retryable,
):
    company = _company().model_copy(
        update={
            "company_name": "Acme International Holdings",
            "country": "Canada",
        }
    )
    icp = _icp().model_copy(
        update={"geography": "Canada", "country": "Canada"}
    )
    verdict = _verdict(observed_size=None, size_matches=None, employee_url="")
    verdict.update(
        observed_company_name="Acme",
        observed_hq_country="Canada",
        employee_size_evidence_quote="",
        geography_evidence_quote="Acme is headquartered in Canada.",
    )
    provider_calls = []

    async def prechecks(*_args, **_kwargs):
        return company_fit_match("prechecks passed")

    async def homepage(*_args, **_kwargs):
        return company_fit_unavailable(homepage_reason)

    async def provider(**kwargs):
        provider_calls.append(kwargs["telemetry_purpose"])
        return dict(verdict), ""

    async def unexpected_fetch(_url):
        raise AssertionError("missing employee-size evidence must not be fetched")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(
        lead_scorer,
        "_request_company_reverify_json",
        provider,
    )
    monkeypatch.setattr(
        lead_scorer,
        "fetch_current_linkedin_company_size",
        unexpected_fetch,
    )

    result = asyncio.run(
        lead_scorer.score_company_competition_intent(
            company,
            icp,
            0.0,
            0.0,
            set(),
            company_quality=True,
        )
    )
    breakdown = result.model_dump(mode="json")
    receipt = breakdown["verifier_gate_receipts"][0]

    assert provider_calls == [
        "lead_scorer_reverify",
        "lead_scorer_reverify_schema_repair",
    ]
    assert result.final_score == 0.0
    assert result.failure_reason.startswith("Company fit unavailable:")
    assert receipt["decision"] == COMPANY_FIT_UNAVAILABLE
    assert receipt["dimension_evidence"]["identity"]["web_identity_receipt"][
        "reason_code"
    ] == "identity_name_alias_unresolved"
    assert scorer_breakdown_has_retryable_infrastructure_failure(
        breakdown
    ) is expected_retryable
    assert (receipt.get("failure_class") == "insufficient_fit_evidence") is (
        not expected_retryable
    )


def test_direct_size_repair_clears_an_earlier_invalid_linkedin_citation(
    monkeypatch,
):
    initial = _verdict(
        employee_url="https://www.linkedin.com/in/acme-employee",
    )
    repaired = _verdict(
        observed_size="11-50",
        size_matches=True,
        employee_url="https://acme.example.com/about",
    )
    repaired["employee_size_evidence_quote"] = "Acme has 11-50 employees."
    verdicts = [initial, repaired]
    provider_calls = []

    async def provider(**kwargs):
        provider_calls.append(kwargs["telemetry_purpose"])
        return verdicts.pop(0), ""

    async def unexpected_fetch(_url):
        raise AssertionError("neither citation is a LinkedIn company profile")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer,
        "fetch_current_linkedin_company_size",
        unexpected_fetch,
    )

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
        )
    )

    assert provider_calls == [
        "lead_scorer_reverify",
        "lead_scorer_reverify_schema_repair",
    ]
    assert result.decision == COMPANY_FIT_MATCH
    assert "failure_class" not in result.details


@pytest.mark.parametrize(
    ("observed_size", "size_matches"),
    [
        ("1", False),
        ("1-10", True),
        (None, None),
    ],
)
def test_successful_profile_without_size_is_reused_as_insufficient(
    monkeypatch,
    observed_size,
    size_matches,
):
    provider_calls = []
    fetches = []

    async def provider(**kwargs):
        provider_calls.append(kwargs["telemetry_purpose"])
        return _verdict(
            observed_size=observed_size,
            size_matches=size_matches,
        ), ""

    async def fetch(url, **_kwargs):
        fetches.append(url)
        return {
            "outcome": "insufficient_evidence",
            "url": url,
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", fetch)

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_homepage_anchor(),
        )
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["failure_class"] == "insufficient_fit_evidence"
    assert result.details["dimension_decisions"]["employee_size"] == (
        COMPANY_FIT_UNAVAILABLE
    )
    assert provider_calls == [
        "lead_scorer_reverify",
        "lead_scorer_reverify_schema_repair",
    ]
    assert fetches == ["https://www.linkedin.com/company/acme"]


def test_repair_reuses_successful_refresh_and_non_linkedin_evidence_is_unchanged(
    monkeypatch,
):
    verdicts = [_verdict(), _verdict()]
    verdicts[0]["industry_matches"] = None
    fetches = []

    async def provider(**_kwargs):
        return verdicts.pop(0), ""

    async def fetch(url, **_kwargs):
        fetches.append(url)
        return {
            "employee_count": "11-50",
            "url": url,
            "quote": "Company size 11-50 employees",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", fetch)

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_homepage_anchor(),
        )
    )
    assert result.decision == COMPANY_FIT_MATCH
    assert fetches == ["https://www.linkedin.com/company/acme"]

    async def direct_provider(**_kwargs):
        return _verdict(
            observed_size="11-50",
            size_matches=True,
            employee_url="https://acme.example.com/about",
        ), ""

    async def unexpected_fetch(_url):
        raise AssertionError("non-LinkedIn evidence must retain the existing path")

    monkeypatch.setattr(
        lead_scorer,
        "_request_company_reverify_json",
        direct_provider,
    )
    monkeypatch.setattr(
        lead_scorer,
        "fetch_current_linkedin_company_size",
        unexpected_fetch,
    )
    direct = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
        )
    )
    assert direct.decision == COMPANY_FIT_MATCH
    assert direct.details["dimension_evidence"]["employee_size"]["url"] == (
        "https://acme.example.com/about"
    )


@pytest.mark.parametrize(
    "employee_url",
    ["", "https://acme.example.com/about"],
)
def test_verified_homepage_anchor_supplies_missing_size_profile(
    monkeypatch,
    employee_url,
):
    fetches = []

    async def provider(**_kwargs):
        verdict = _verdict(
            observed_size=None,
            size_matches=None,
            employee_url=employee_url,
        )
        verdict["employee_size_evidence_quote"] = ""
        return verdict, ""

    async def fetch(url, **_kwargs):
        fetches.append(url)
        return {
            "employee_count": "11-50",
            "url": url,
            "quote": "Company size 11-50 employees",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", fetch)

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_homepage_anchor(),
        )
    )

    assert result.decision == COMPANY_FIT_MATCH
    assert fetches == ["https://www.linkedin.com/company/acme"]
    assert result.details["provider_observations"]["observed_employee_count"] == (
        "11-50"
    )


def test_structured_size_fallback_is_reused_across_sonar_repair(monkeypatch):
    provider_calls = []
    exa_fetches = []
    structured_fetches = []

    async def provider(**kwargs):
        provider_calls.append(kwargs["telemetry_purpose"])
        verdict = _verdict(
            observed_size=None,
            size_matches=None,
            employee_url="",
        )
        verdict["employee_size_evidence_quote"] = ""
        verdict["industry_matches"] = True if len(provider_calls) > 1 else None
        return verdict, ""

    async def exa_fetch(url, *, diagnostic):
        exa_fetches.append(url)
        return {"outcome": "insufficient_evidence", "url": url}

    async def structured_fetch(domain, url, **_kwargs):
        structured_fetches.append((domain, url))
        return {
            "employee_count": "11-50",
            "provider": "harvestapi_get_company",
            "source_field": "employeeCountRange",
            "url": url,
            "website": "https://acme.example.com/",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", exa_fetch)
    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        structured_fetch,
    )

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_homepage_anchor(),
        )
    )

    assert result.decision == COMPANY_FIT_MATCH
    assert provider_calls == [
        "lead_scorer_reverify",
        "lead_scorer_reverify_schema_repair",
    ]
    assert exa_fetches == ["https://www.linkedin.com/company/acme"]
    assert structured_fetches == [
        ("acme.example.com", "https://www.linkedin.com/company/acme")
    ]
    assert "failure_class" not in result.details
    assert result.details["dimension_evidence"]["employee_size"] == {
        "employee_count": "11-50",
        "provider": "harvestapi_get_company",
        "source_field": "employeeCountRange",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://acme.example.com/",
    }


def test_structured_size_proof_survives_full_company_fit_merge(monkeypatch):
    async def prechecks(*_args, **_kwargs):
        return company_fit_match("prechecks passed")

    async def homepage(*_args, **_kwargs):
        return _homepage_anchor()

    async def provider(**_kwargs):
        verdict = _verdict(
            observed_size=None,
            size_matches=None,
            employee_url="",
        )
        verdict["employee_size_evidence_quote"] = ""
        return verdict, ""

    async def exa_fetch(url, **_kwargs):
        return {"outcome": "insufficient_evidence", "url": url}

    async def structured_fetch(domain, url, **_kwargs):
        return {
            "employee_count": "11-50",
            "provider": "harvestapi_get_company",
            "source_field": "employeeCountRange",
            "url": url,
            "website": f"https://{domain}/",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", exa_fetch)
    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        structured_fetch,
    )

    result = asyncio.run(
        lead_scorer._verify_company_fit(
            _company(),
            _icp(),
            0.0,
            0.0,
            set(),
            require_https_transport=True,
        )
    )

    assert result.decision == COMPANY_FIT_MATCH
    employee_evidence = result.details["dimension_evidence"]["employee_size"]
    assert employee_evidence["decision"] == COMPANY_FIT_MATCH
    assert employee_evidence["web_evidence"]["provider"] == (
        "harvestapi_get_company"
    )
    assert "quote" not in employee_evidence["web_evidence"]


def test_structured_public_profile_repairs_unavailable_stage_with_one_fetch(
    monkeypatch,
):
    provider_calls = []
    structured_fetches = []
    payload = _structured_company_payload(
        companyType="Public Company",
        website="https://example.com",
    )

    async def provider(**kwargs):
        provider_calls.append(kwargs["telemetry_purpose"])
        verdict = _verdict()
        verdict.update(
            observed_company_website="https://example.com/about",
            observed_company_stage="Public",
            stage_matches=True,
            stage_evidence_url="https://acme.example.com/about",
            stage_evidence_quote="Company type: Public Company",
        )
        return verdict, ""

    async def exa_fetch(url, **_kwargs):
        return {
            "employee_count": "11-50",
            "url": url,
            "quote": "Company size 11-50 employees",
        }

    async def structured_fetch(
        domain, url, *, diagnostic, public_company_evidence
    ):
        del diagnostic
        structured_fetches.append((domain, url))
        public = (
            linkedin_company_size.project_structured_linkedin_public_company(
                domain,
                url,
                payload,
            )
        )
        assert public is not None
        public_company_evidence.update(public)
        return linkedin_company_size.project_structured_linkedin_company_size(
            domain,
            url,
            payload,
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", exa_fetch)
    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        structured_fetch,
    )

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company().model_copy(
                update={
                    "company_stage": "Public",
                    "company_website": "https://example.com",
                }
            ),
            _icp().model_copy(update={"company_stage": "Public"}),
            require_company_fit_dimensions=True,
            verified_homepage_identity=company_fit_match(
                "homepage identity verified",
                details={
                    "identity": {
                        "decision": COMPANY_FIT_MATCH,
                        "evidence_source": "company_homepage",
                        "observed_name": "acme",
                        "observed_domain": "example.com",
                        "observed_linkedin_slug": "acme",
                    }
                },
            ),
        )
    )

    assert result.decision == COMPANY_FIT_MATCH, result.details
    assert provider_calls == ["lead_scorer_reverify"]
    assert structured_fetches == [
        ("example.com", "https://www.linkedin.com/company/acme")
    ]
    assert result.details["dimension_decisions"]["stage"] == COMPANY_FIT_MATCH
    assert result.details["dimension_evidence"]["stage"] == {
        "company_type": "Public Company",
        "provider": "harvestapi_get_company",
        "source_field": "companyType",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://example.com/",
    }


@pytest.mark.parametrize(
    ("company_type", "profile_website", "profile_slug", "expected"),
    [
        ("Public Company", "https://www.statestreet.com", "state-street", "match"),
        ("Privately Held", "https://www.statestreet.com", "state-street", "unavailable"),
        ("Public Company", "https://unrelated.example", "state-street", "unavailable"),
        ("Public Company", "https://www.statestreet.com", "unrelated", "unavailable"),
        (None, "https://www.statestreet.com", "state-street", "unavailable"),
    ],
)
def test_public_stage_uses_verified_web_identity_without_homepage_linkedin(
    monkeypatch, company_type, profile_website, profile_slug, expected
):
    """Reproduce the saved Sep20 identity/quote through the real fit verifier."""

    company = _company().model_copy(update={
        "company_name": "State Street",
        "company_website": "https://www.statestreet.com",
        "company_linkedin": "https://www.linkedin.com/company/state-street",
        "company_stage": "Public",
        "industry": "Financial services",
        "employee_count": "10,001+",
        "state": "Massachusetts",
    })
    icp = _icp().model_copy(update={
        "company_stage": "Public",
        "industry": "Financial services",
        "sub_industry": "investment services and investment management",
        "employee_count": "10,001+",
    })
    profile_url = "https://www.linkedin.com/company/state-street"
    verdict = _verdict(observed_size="10,001+", size_matches=True)
    verdict.update(
        observed_company_name="State Street Corporation",
        observed_company_website="https://www.statestreet.com",
        observed_company_linkedin=profile_url,
        employee_size_evidence_url=profile_url,
        employee_size_evidence_quote="Company size 10,001+ employees",
        observed_industry="Financial services",
        observed_subindustry="investment services and investment management",
        industry_evidence_url="https://www.statestreet.com/",
        industry_evidence_quote=(
            "State Street is one of the world’s leading providers of financial "
            "services to institutional investors, including investment servicing, "
            "investment management and investment research and trading."
        ),
        observed_hq_state="Massachusetts",
        geography_evidence_url="https://www.statestreet.com/about",
        geography_evidence_quote=(
            "Our corporate headquarters is located at One Congress Street, "
            "Boston, Massachusetts 02114."
        ),
        observed_company_stage="Public",
        stage_matches=True,
        stage_evidence_url=profile_url,
        stage_evidence_quote=(
            "Public Company · Founded 1792 · 10001+ employees · 852053 followers"
        ),
    )
    calls = []

    async def provider(**kwargs):
        calls.append(kwargs["telemetry_purpose"])
        return dict(verdict), ""

    async def current_profile(url, **_kwargs):
        return {"employee_count": "10,001+", "url": url,
                "quote": "Company size 10,001+ employees"}

    async def structured_profile(domain, url, *, diagnostic, public_company_evidence):
        assert (domain, url) == ("statestreet.com", profile_url)
        calls.append("structured_profile")
        if company_type is not None:
            public_company_evidence.update({
                "company_type": company_type,
                "provider": "harvestapi_get_company",
                "source_field": "companyType",
                "url": "https://www.linkedin.com/company/" + profile_slug,
                "website": profile_website,
            })
        return None

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", current_profile)
    monkeypatch.setattr(lead_scorer, "fetch_structured_linkedin_company_size", structured_profile)
    result = asyncio.run(lead_scorer._llm_reverify_company(
        company, icp, require_company_fit_dimensions=True, company_quality=True,
        verified_homepage_identity=company_fit_unavailable(
            "homepage identity evidence unavailable: LinkedIn company binding not found",
            details={"verified_homepage_transport_domain": "statestreet.com"},
        ),
    ))
    assert result.details["identity_decision"] == COMPANY_FIT_MATCH
    assert result.details["dimension_decisions"]["stage"] == expected
    assert result.decision == expected
    assert calls.count("structured_profile") == 1
    if expected == COMPANY_FIT_MATCH:
        assert calls == ["lead_scorer_reverify", "structured_profile"]
        assert result.details["dimension_evidence"]["stage"]["provider"] == (
            "harvestapi_get_company"
        )


@pytest.mark.parametrize("missing_size_observation", [False, True])
def test_state_street_web_identity_reuses_structured_profile_for_size_and_stage(
    monkeypatch, missing_size_observation,
):
    """Reproduce the saved rerun337 profile through the full outer fit gate."""

    company = _company().model_copy(update={
        "company_name": "State Street",
        "company_website": "https://www.statestreet.com",
        "company_linkedin": "https://www.linkedin.com/company/state-street",
        "company_stage": "Public",
        "industry": "Financial services",
        "employee_count": "10,001+",
        "state": "Massachusetts",
    })
    icp = _icp().model_copy(update={
        "company_stage": "Public",
        "industry": "Financial services",
        "sub_industry": "investment services and investment management",
        "employee_count": "10,001+",
    })
    profile_url = "https://www.linkedin.com/company/state-street"
    verdict = _verdict(
        observed_size=52000,
        size_matches=False,
        employee_url="https://www.statestreet.com/about/annual-report",
    )
    verdict.update(
        observed_company_name="State Street Corporation",
        observed_company_website="https://www.statestreet.com",
        observed_company_linkedin=profile_url,
        employee_size_evidence_quote="approximately 52,000 employees worldwide",
        observed_industry="Financial services",
        observed_subindustry="investment services and investment management",
        industry_evidence_url="https://www.statestreet.com/",
        industry_evidence_quote=(
            "State Street provides investment servicing and investment management "
            "to institutional investors."
        ),
        observed_hq_country="United States",
        observed_hq_state="Massachusetts",
        geography_evidence_url="https://www.statestreet.com/about",
        geography_evidence_quote="Corporate headquarters: Boston, Massachusetts.",
        observed_company_stage="Public",
        stage_matches=True,
        stage_evidence_url=profile_url,
        stage_evidence_quote="Public Company",
    )
    if missing_size_observation:
        verdict.update(
            observed_employee_count=None,
            employee_size_matches=None,
            employee_size_evidence_url="",
            employee_size_evidence_quote="",
        )
    payload = {
        "status": "completed",
        "result": {"data": {"status": 200, "element": {
            "name": "State Street",
            "universalName": "state-street",
            "linkedinUrl": profile_url + "/",
            "website": "http://www.statestreet.com",
            "employeeCount": 47424,
            "employeeCountRange": {"start": 10001, "end": None},
            "companyType": "Public Company",
        }}},
    }
    structured_calls = []
    exa_calls = []

    async def prechecks(*_args, **_kwargs):
        return company_fit_match("prechecks passed")

    async def homepage(*_args, **_kwargs):
        return company_fit_unavailable(
            "homepage identity evidence unavailable: LinkedIn binding not found",
            details={"verified_homepage_transport_domain": "statestreet.com"},
        )

    async def provider(**_kwargs):
        return dict(verdict), ""

    async def structured_profile(
        domain, url, *, diagnostic, public_company_evidence
    ):
        del diagnostic
        structured_calls.append((domain, url))
        public = linkedin_company_size.project_structured_linkedin_company_type(
            domain, url, payload
        )
        assert public is not None
        public_company_evidence.update(public)
        return linkedin_company_size.project_structured_linkedin_company_size(
            domain, url, payload
        )

    async def exa_profile(url, **_kwargs):
        exa_calls.append(url)
        return {"outcome": "insufficient_evidence", "url": url}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer, "fetch_structured_linkedin_company_size", structured_profile
    )
    monkeypatch.setattr(
        lead_scorer, "fetch_current_linkedin_company_size", exa_profile
    )

    result = asyncio.run(
        lead_scorer._verify_company_fit(
            company,
            icp,
            0.0,
            0.0,
            set(),
            require_https_transport=True,
            company_quality=True,
            evidence_investigator=True,
        )
    )

    assert result.decision == COMPANY_FIT_MATCH, result.details
    assert exa_calls == [profile_url]
    assert structured_calls == [("statestreet.com", profile_url)]
    assert result.details["company_fit_dimensions"]["employee_size"] == COMPANY_FIT_MATCH
    assert result.details["company_fit_dimensions"]["stage"] == COMPANY_FIT_MATCH
    assert result.details["dimension_evidence"]["employee_size"]["web_evidence"] == {
        "employee_count": "10,001+",
        "provider": "harvestapi_get_company",
        "source_field": "employeeCountRange",
        "url": profile_url,
        "website": "https://statestreet.com/",
    }
    assert result.details["dimension_evidence"]["stage"]["web_evidence"] == {
        "company_type": "Public Company",
        "provider": "harvestapi_get_company",
        "source_field": "companyType",
        "url": profile_url,
        "website": "https://statestreet.com/",
    }


@pytest.mark.parametrize("field,value", [
    ("decision", "unavailable"), ("evidence_source", "submitted"),
    ("observed_domain", "unrelated.example"),
    ("submitted_domain", "unrelated.example"),
    ("observed_name", ""), ("observed_linkedin_slug", ""),
])
def test_structured_profile_fallback_rejects_incomplete_or_conflicting_identity(field, value):
    receipt = {
        "decision": "match", "evidence_source": "company_web_reverification",
        "observed_domain": "example.com", "submitted_domain": "example.com",
        "observed_name": "acme", "observed_linkedin_slug": "acme",
        field: value,
    }
    assert not lead_scorer._structured_profile_identity_anchor({}, receipt, "example.com")
    assert not lead_scorer._structured_profile_identity_anchor({}, receipt, "")


@pytest.mark.parametrize("provider_failure", [False, True])
def test_structured_public_stage_scorer_entrypoint_transition(
    monkeypatch,
    provider_failure,
):
    # The real scorer starts in an isolated container with these placeholders.
    # Do not inherit credentials from other tests in this shared test process.
    for name, value in scorer_entrypoint.PLACEHOLDER_CREDENTIALS.items():
        monkeypatch.setenv(name, value)
    calls = 0
    structured_stage_evidence = {
        "company_type": "Public Company",
        "provider": "harvestapi_get_company",
        "source_field": "companyType",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://example.com/",
    }

    async def prechecks(*_args, **_kwargs):
        return company_fit_match("prechecks passed")

    async def homepage(*_args, **_kwargs):
        return company_fit_unavailable(
            "homepage identity evidence unavailable: LinkedIn binding not found",
            details={"verified_homepage_transport_domain": "example.com"},
        )

    async def web_verification(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if provider_failure:
            return company_fit_unavailable(
                "company verification provider unavailable",
                details={"failure_reason_code": "provider_error"},
            )
        web_evidence = {
            dimension: {
                "url": f"https://example.com/{dimension}",
                "quote": f"Acme {dimension} evidence.",
            }
            for dimension in ("employee_size", "industry", "geography")
        }
        web_evidence["stage"] = structured_stage_evidence
        return company_fit_match(
            "company fit verified",
            details={
                "dimension_decisions": {
                    dimension: COMPANY_FIT_MATCH
                    for dimension in (
                        "employee_size",
                        "industry",
                        "geography",
                        "stage",
                    )
                },
                "dimension_evidence": web_evidence,
                "identity_decision": COMPANY_FIT_MATCH,
                "identity_receipt": {
                    "decision": COMPANY_FIT_MATCH,
                    "evidence_source": "company_web_reverification",
                    "submitted_name": "acme",
                    "submitted_domain": "example.com",
                    "submitted_linkedin_slug": "acme",
                    "observed_name": "acme",
                    "observed_domain": "example.com",
                    "observed_linkedin_slug": "acme",
                },
                "required_attribute_decision": COMPANY_FIT_MATCH,
            },
        )

    async def intent_score(*_args, **_kwargs):
        return 60.0, 100, "verified", "2026-08-01", 0

    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(
        lead_scorer, "_llm_reverify_company", web_verification
    )
    monkeypatch.setattr(
        lead_scorer, "_score_single_intent_signal", intent_score
    )
    icp = _icp().model_copy(update={
        "company_stage": "Public",
        "intent_signals": ["Launched a product"],
    })
    document = arena_scoring.build_scoring_input(
        scored_run_id="structured-public-stage",
        icp=icp.model_dump(mode="json"),
        companies=[_competition_company()],
        policy=arena_scoring.build_scorer_policy(),
        evaluation_date="2026-09-19",
    )

    output = scorer_entrypoint.score_input(document)

    if provider_failure:
        assert calls == 3
        assert output["failure"] == "judge_error"
        assert output["reason"] == "provider_error"
        return
    assert calls == 1
    receipt = output["breakdowns"][0]["verifier_gate_receipts"][0]
    assert receipt["decision"] == COMPANY_FIT_MATCH
    assert receipt["company_fit_dimensions"]["stage"] == COMPANY_FIT_MATCH
    assert receipt["dimension_evidence"]["stage"]["web_evidence"] == (
        structured_stage_evidence
    )


def test_truist_structured_identity_recovers_full_scorer_entrypoint(monkeypatch):
    for name, value in scorer_entrypoint.PLACEHOLDER_CREDENTIALS.items():
        monkeypatch.setenv(name, value)
    calls = {"web": 0, "structured": 0, "exa": 0}
    profile_url = (
        "https://www.linkedin.com/company/truistfinancialcorporation"
    )

    async def prechecks(*_args, **_kwargs):
        return company_fit_match("prechecks passed")

    async def homepage(*_args, **_kwargs):
        return company_fit_unavailable(
            "website fetch error: homepage response body is unusable",
            details={
                "failure_reason_code": "malformed_response",
                "verified_homepage_transport_domain": "truist.com",
            },
        )

    async def provider(**_kwargs):
        calls["web"] += 1
        return {
            "observed_company_name": "Truist Financial Corporation",
            "observed_company_website": "https://www.truist.com",
            "observed_company_linkedin": profile_url,
            "observed_employee_count": 29447,
            "employee_size_matches": True,
            "employee_size_evidence_url": profile_url,
            "employee_size_evidence_quote": "29,447 associated members.",
            "observed_industry": "Financial Services",
            "observed_subindustry": "Banking",
            "industry_matches": True,
            "industry_activity_role": "supplier_operator",
            "industry_evidence_url": "https://www.truist.com/about",
            "industry_evidence_quote": "Truist provides banking services.",
            "observed_hq_country": "United States",
            "observed_hq_state": "North Carolina",
            "geography_matches": True,
            "geography_evidence_url": "https://www.truist.com/about",
            "geography_evidence_quote": (
                "Truist is headquartered in Charlotte, North Carolina."
            ),
            "observed_company_stage": "Public",
            "stage_matches": True,
            "stage_evidence_url": "https://ir.truist.com/",
            "stage_evidence_quote": "Truist Financial Corporation NYSE TFC",
            "attribute_satisfied": None,
            "required_attribute_evidence_url": "",
            "required_attribute_evidence_quote": "",
            "reason": "Independent sources support the company.",
        }, ""

    async def structured_profile(
        domain,
        url,
        *,
        diagnostic,
        public_company_evidence,
        company_identity_evidence,
    ):
        del diagnostic
        calls["structured"] += 1
        assert (domain, url) == ("truist.com", profile_url)
        company_identity_evidence.update({
            "name": "Truist",
            "provider": "harvestapi_get_company",
            "source_field": "name",
            "url": profile_url,
            "website": "https://truist.com/",
        })
        public_company_evidence.update({
            "company_type": "Public Company",
            "provider": "harvestapi_get_company",
            "source_field": "companyType",
            "url": profile_url,
            "website": "https://truist.com/",
        })
        return {
            "employee_count": "10,001+",
            "provider": "harvestapi_get_company",
            "source_field": "employeeCountRange",
            "url": profile_url,
            "website": "https://truist.com/",
        }

    async def exa_profile(url, **_kwargs):
        calls["exa"] += 1
        return {"outcome": "insufficient_evidence", "url": url}

    async def intent_score(*_args, **_kwargs):
        return 60.0, 100, "verified", "2026-09-03", 0

    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        structured_profile,
    )
    monkeypatch.setattr(
        lead_scorer,
        "fetch_current_linkedin_company_size",
        exa_profile,
    )
    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", intent_score)
    icp = _icp().model_copy(update={
        "industry": "Financial Services",
        "employee_count": "10,001+",
        "company_stage": "Public",
        "intent_signals": ["Leadership change"],
    })
    company = _competition_company()
    company.update({
        "company_name": "Truist",
        "company_website": "https://www.truist.com/",
        "company_linkedin": profile_url,
        "industry": "Financial Services",
        "employee_count": "10,001+",
        "company_stage": "Public",
        "state": "North Carolina",
    })
    company["intent_signals"][0].update({
        "description": "Truist appointed a national brokerage director.",
        "date": "2026-09-03",
        "url": (
            "https://ir.truist.com/2026-09-03-Tory-Sherman-joins-Truist-"
            "Wealth-as-Wealth-Brokerage-national-director"
        ),
    })
    document = arena_scoring.build_scoring_input(
        scored_run_id="truist-structured-identity",
        icp=icp.model_dump(mode="json"),
        companies=[company],
        policy=arena_scoring.build_scorer_policy(
            scoring_adapter_version="qualification_integrity_v2",
            company_quality=True,
        ),
        evaluation_date="2026-09-20",
    )
    [company_ref] = company_judgments.build_company_scopes(
        scoring_input=document,
        round_id="arena-2026-09-20",
        network_name="finney",
        netuid=71,
        scorer_image_digest="sha256:" + "a" * 64,
        scorer_image_reference="registry/scorer@sha256:" + "a" * 64,
        integrity_policy="arena_integrity_v1",
        company_quality_policy="company_quality_v1",
    )
    document["company_judgment_cache"] = {
        "schema_version": company_judgments.LEASE_SCHEMA_VERSION,
        "hits": [],
        "misses": [{
            "company_index": company_ref["company_index"],
            "cache_key": company_ref["cache_key"],
            "company_input_hash": company_ref["company_input_hash"],
            "authority_slot": 0,
        }],
    }

    output = scorer_entrypoint.score_input(document)

    assert "failure" not in output, (output, calls)
    assert calls == {"web": 1, "structured": 1, "exa": 1}, output
    receipt = output["breakdowns"][0]["verifier_gate_receipts"][0]
    assert receipt["decision"] == COMPANY_FIT_MATCH
    assert receipt["dimension_evidence"]["identity"][
        "web_identity_receipt"
    ]["reason_code"] == "structured_profile_alias_verified"
    assert receipt["company_fit_dimensions"] == {
        "identity": COMPANY_FIT_MATCH,
        "employee_size": COMPANY_FIT_MATCH,
        "industry": COMPANY_FIT_MATCH,
        "geography": COMPANY_FIT_MATCH,
        "stage": COMPANY_FIT_MATCH,
    }


@pytest.mark.parametrize("recovers", [True, False])
def test_tiny_homepage_placeholder_retries_full_scorer(
    monkeypatch,
    recovers,
):
    for name, value in scorer_entrypoint.PLACEHOLDER_CREDENTIALS.items():
        monkeypatch.setenv(name, value)

    root_url = "https://example.com/"
    newsroom_url = "https://media.example.com/"
    bad_body = b"Provider account capacity details redacted."
    root_body = (
        b"<title>Acme</title>"
        b'<a href="https://media.example.com/">Newsroom</a>'
    )
    newsroom_body = (
        b"<title>Acme Newsroom</title>"
        b'<a href="https://www.linkedin.com/company/acme">LinkedIn</a>'
    )

    class Content:
        def __init__(self, body):
            self.body = body

        async def read(self, limit):
            value, self.body = self.body[:limit], self.body[limit:]
            return value

    class Response:
        def __init__(self, body, url):
            self.status = 200
            self.url = url
            self.headers = {"Content-Type": "text/html"}
            self.content = Content(body)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    pending = (
        [Response(bad_body, root_url), Response(root_body, root_url),
         Response(newsroom_body, newsroom_url)]
        if recovers
        else [Response(bad_body, root_url) for _ in range(3)]
    )
    fetched_urls = []

    class Session:
        def __init__(self, **_kwargs):
            self.response = pending.pop(0)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def get(self, url, **_kwargs):
            fetched_urls.append(url)
            return self.response

    async def prechecks(*_args, **_kwargs):
        return company_fit_match("prechecks passed")

    async def web_verification(*_args, verified_homepage_identity, **_kwargs):
        identity_match = verified_homepage_identity.decision == COMPANY_FIT_MATCH
        decision = COMPANY_FIT_MATCH if identity_match else COMPANY_FIT_UNAVAILABLE
        evidence = {
            dimension: {
                "url": f"https://example.com/{dimension}",
                "quote": f"Acme {dimension} evidence.",
            }
            for dimension in ("employee_size", "industry", "geography")
        }
        details = {
            "dimension_decisions": {
                "employee_size": COMPANY_FIT_MATCH,
                "industry": COMPANY_FIT_MATCH,
                "geography": COMPANY_FIT_MATCH,
                "stage": COMPANY_FIT_MATCH,
            },
            "dimension_evidence": evidence,
            "identity_decision": decision,
            "identity_receipt": {
                "decision": decision,
                "evidence_source": "company_web_reverification",
                "submitted_name": "acme",
                "submitted_domain": "example.com",
                "submitted_linkedin_slug": "acme",
                "observed_name": "acme",
                "observed_domain": "example.com",
                "observed_linkedin_slug": "acme",
            },
            "required_attribute_decision": COMPANY_FIT_MATCH,
            **({"failure_class": "insufficient_fit_evidence"}
               if not identity_match else {}),
        }
        if identity_match:
            return company_fit_match("company fit verified", details=details)
        return company_fit_unavailable("company identity unavailable", details=details)

    async def intent_score(*_args, **_kwargs):
        return 60.0, 100, "verified", "2026-08-01", 0

    monkeypatch.setattr(company_verification.aiohttp, "ClientSession", Session)
    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "_llm_reverify_company", web_verification)
    monkeypatch.setattr(lead_scorer, "_score_single_intent_signal", intent_score)
    icp = _icp().model_copy(update={"intent_signals": ["Launched a product"]})
    document = arena_scoring.build_scoring_input(
        scored_run_id="tiny-homepage-retry",
        icp=icp.model_dump(mode="json"),
        companies=[_competition_company()],
        policy=arena_scoring.build_scorer_policy(),
        evaluation_date="2026-09-19",
    )

    output = scorer_entrypoint.score_input(document)

    if not recovers:
        assert fetched_urls == [root_url] * 3
        assert output.get("failure") == "judge_error", output
        assert output["reason"] == "malformed_response"
        return
    assert fetched_urls == [root_url, root_url, newsroom_url]
    receipt = output["breakdowns"][0]["verifier_gate_receipts"][0]
    assert receipt["decision"] == COMPANY_FIT_MATCH
    assert receipt["dimension_evidence"]["identity"][
        "homepage_identity_decision"
    ] == COMPANY_FIT_MATCH


def test_structured_profile_cache_reuses_one_response_for_stage_and_size(
    monkeypatch,
):
    calls = []

    async def structured_fetch(
        domain, url, *, diagnostic, public_company_evidence
    ):
        del diagnostic
        calls.append((domain, url))
        public_company_evidence.update({
            "company_type": "Public Company",
            "provider": "harvestapi_get_company",
            "source_field": "companyType",
            "url": url,
            "website": f"https://{domain}/",
        })
        return {
            "employee_count": "11-50",
            "provider": "harvestapi_get_company",
            "source_field": "employeeCountRange",
            "url": url,
            "website": f"https://{domain}/",
        }

    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        structured_fetch,
    )
    cache = {}
    identity = {
        "normalized_name": "acme",
        "registrable_dns_domain": "example.com",
        "linkedin_company_slug": "acme",
    }

    asyncio.run(
        lead_scorer._fetch_structured_linkedin_profile_once(
            identity,
            cache,
            collect_employee_size=False,
        )
    )
    asyncio.run(
        lead_scorer._fetch_structured_linkedin_profile_once(
            identity,
            cache,
            collect_employee_size=True,
        )
    )

    assert calls == [("example.com", "https://www.linkedin.com/company/acme")]
    assert cache["structured_employee_size_applicable"] is True
    assert cache["structured_evidence"]["employee_count"] == "11-50"
    assert cache["structured_public_company_evidence"]["company_type"] == (
        "Public Company"
    )


@pytest.mark.parametrize("direct_conflict", [False, True])
def test_size_refresh_reuses_profile_fetched_first_for_public_stage(
    monkeypatch,
    direct_conflict,
):
    structured_calls = []
    exa_calls = []

    async def structured_fetch(
        domain, url, *, diagnostic, public_company_evidence
    ):
        del diagnostic
        structured_calls.append((domain, url))
        public_company_evidence.update({
            "company_type": "Public Company",
            "provider": "harvestapi_get_company",
            "source_field": "companyType",
            "url": url,
            "website": f"https://{domain}/",
        })
        return {
            "employee_count": "11-50",
            "provider": "harvestapi_get_company",
            "source_field": "employeeCountRange",
            "url": url,
            "website": f"https://{domain}/",
        }

    async def exa_fetch(url, **_kwargs):
        exa_calls.append(url)
        return {"outcome": "insufficient_evidence", "url": url}

    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        structured_fetch,
    )
    monkeypatch.setattr(
        lead_scorer,
        "fetch_current_linkedin_company_size",
        exa_fetch,
    )
    cache = {}
    identity = {
        "normalized_name": "acme",
        "registrable_dns_domain": "acme.example.com",
        "linkedin_company_slug": "acme",
    }
    asyncio.run(
        lead_scorer._fetch_structured_linkedin_profile_once(
            identity,
            cache,
            collect_employee_size=False,
        )
    )
    verdict = (
        _verdict(
            observed_size=7,
            size_matches=False,
            employee_url="https://evidence.example/headcount",
        )
        if direct_conflict
        else _verdict(
            observed_size=None,
            size_matches=None,
            employee_url="",
        )
    )
    if not direct_conflict:
        verdict["employee_size_evidence_quote"] = ""

    asyncio.run(
        lead_scorer._refresh_linkedin_employee_size_observation(
            verdict,
            _company(),
            _icp(),
            verified_homepage_identity=identity,
            invocation_cache=cache,
            collect_structured_conflict=True,
        )
    )

    assert structured_calls == [
        ("acme.example.com", "https://www.linkedin.com/company/acme")
    ]
    assert cache["structured_employee_size_applicable"] is True
    assert cache["structured_evidence"]["employee_count"] == "11-50"
    assert len(exa_calls) == (0 if direct_conflict else 1)


@pytest.mark.parametrize(
    "stage_updates",
    [
        {
            "observed_company_stage": "Series C+",
            "stage_matches": False,
            "stage_evidence_url": "https://acme.example.com/news",
            "stage_evidence_quote": "Acme closed a Series D round.",
        },
        {
            "observed_company_stage": "",
            "stage_matches": None,
            "stage_evidence_url": "https://acme.example.com/news",
            "stage_evidence_quote": "Acme was taken private by a private equity firm.",
        },
        {
            "observed_company_stage": "",
            "stage_matches": None,
            "stage_evidence_url": "https://acme.example.com/about",
            "stage_evidence_quote": "Acme remains not publicly traded.",
        },
    ],
)
def test_structured_public_profile_never_overrides_proven_nonpublic_stage(
    stage_updates,
):
    verdict = _verdict(employee_url="https://evidence.example/headcount")
    verdict.update(stage_updates)
    result = lead_scorer._reverify_decision(
        verdict,
        "",
        "public",
        icp=_icp().model_copy(update={"company_stage": "Public"}),
        company=_company().model_copy(update={"company_stage": "Public"}),
        verified_homepage_identity={
            "normalized_name": "acme",
            "registrable_dns_domain": "acme.example.com",
            "linkedin_company_slug": "acme",
        },
        structured_public_company_evidence={
            "company_type": "Public Company",
            "provider": "harvestapi_get_company",
            "source_field": "companyType",
            "url": "https://www.linkedin.com/company/acme",
            "website": "https://acme.example.com/",
        },
    )

    assert result.decision != COMPANY_FIT_MATCH
    assert result.details["dimension_decisions"]["stage"] != COMPANY_FIT_MATCH


def test_exact_structured_private_company_marks_public_stage_conflict_unavailable():
    verdict = _verdict(observed_size=25, size_matches=True)
    verdict.update(
        observed_company_stage="Public",
        stage_matches=True,
        stage_evidence_url="https://www.sec.gov/Archives/old-filing.htm",
        stage_evidence_quote="The company filed this report in 2024.",
    )
    private_evidence = {
        "company_type": "Privately Held",
        "provider": "harvestapi_get_company",
        "source_field": "companyType",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://example.com/",
    }

    result = lead_scorer._reverify_decision(
        verdict,
        "",
        "public",
        icp=_icp().model_copy(update={"company_stage": "Public"}),
        company=_company().model_copy(update={"company_stage": "Public"}),
        verified_homepage_identity={
            "normalized_name": "acme",
            "registrable_dns_domain": "example.com",
            "linkedin_company_slug": "acme",
        },
        structured_public_company_evidence=private_evidence,
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["dimension_decisions"]["stage"] == COMPANY_FIT_UNAVAILABLE
    assert result.details["dimension_evidence"]["stage"] == private_evidence
    assert lead_scorer._targeted_company_investigation_dimensions(
        result,
        icp_stage="public",
        employee_size_conflict=False,
    ) == ("stage",)


@pytest.mark.parametrize(
    ("stage_url", "stage_quote", "private_update", "expected"),
    [
        (
            "https://www.sec.gov/Archives/edgar/data/1337619/"
            "000133761924000003/env-20240222.htm",
            "ENVESTNET, INC. (Exact name of registrant as specified in its "
            "charter) Delaware 001-34835 20-1409613 (State or Other "
            "Jurisdiction of Incorporation) (Commission File Number) "
            "(I.R.S. Employer Identification Number) 1000 Chesterbrook "
            "Boulevard , Suite 250 , Berwyn , Pennsylvania 19312 (Address "
            "of principal executive offices) (Zip Code) ( 312 ) 827-2800 "
            "(Registrant’s telephone number, including area code) Not "
            "Applicable (Former name or former address, if changed since "
            "last report) Check the appropriate box below if the Form 8-K "
            "filing is intended to simultaneously satisfy the filing "
            "obligation of the registrant under any of the following "
            "provisions (see General Instruction A.2. below): ☐ Written "
            "communications pursuant to Rule 425 under the Securities Act "
            "(17 CFR 230.425) ☐ Soliciting material pursuant to Rule 14a-12 "
            "under the Exchange Act (17 CFR 240.14a-12) ☐ Pre-commencement "
            "communications pursuant to Rule 14d-2(b) under the Exchange "
            "Act (17 CFR 240.14d-2(b)) ☐ Pre-commencement communications "
            "pursuant to Rule 13e-4(c) under the Exchange Act (17 CFR "
            "240-13e-4(c)) Securities registered pursuant to Section 12(b) "
            "of the Act: Title of each class Trading symbol(s) Name of "
            "exchange on which registered Common Stock, par value $0.005 "
            "per share ENV New York Stock Exchange",
            {},
            COMPANY_FIT_UNAVAILABLE,
        ),
        (
            "https://www.envestnet.com/investor-relations",
            "Envestnet common stock is currently listed on the New York "
            "Stock Exchange under ticker ENV.",
            {},
            COMPANY_FIT_MATCH,
        ),
        (
            "https://www.sec.gov/Archives/edgar/data/1337619/"
            "000133761924000003/env-20240222.htm",
            "Envestnet common stock is registered on the New York Stock "
            "Exchange under ticker ENV.",
            {"website": "https://unrelated.example/"},
            COMPANY_FIT_MATCH,
        ),
    ],
)
def test_archived_sec_snapshot_cannot_override_bound_current_private_type(
    stage_url, stage_quote, private_update, expected
):
    verdict = _verdict(observed_size=25, size_matches=True)
    verdict.update(
        observed_company_name="Envestnet",
        observed_company_website="https://envestnet.com/",
        observed_company_linkedin="https://www.linkedin.com/company/envestnet",
        observed_company_stage="Public",
        stage_matches=True,
        stage_evidence_url=stage_url,
        stage_evidence_quote=stage_quote,
    )
    finding = {
        "target": "stage",
        "status": "VERIFIED",
        "observed_value": "Public",
        "evidence_url": stage_url,
        "evidence_quote": stage_quote,
    }
    private_evidence = {
        "company_type": "Privately Held",
        "provider": "harvestapi_get_company",
        "source_field": "companyType",
        "url": "https://www.linkedin.com/company/envestnet",
        "website": "https://envestnet.com/",
        **private_update,
    }
    result = lead_scorer._reverify_decision(
        verdict,
        "",
        "public",
        icp=_icp().model_copy(update={"company_stage": "Public"}),
        company=_company().model_copy(update={
            "company_name": "Envestnet",
            "company_website": "https://envestnet.com",
            "company_linkedin": "https://www.linkedin.com/company/envestnet",
            "company_stage": "Public",
        }),
        verified_homepage_identity={
            "normalized_name": "envestnet",
            "registrable_dns_domain": "envestnet.com",
            "linkedin_company_slug": "envestnet",
        },
        validated_stage_finding=finding,
        structured_public_company_evidence=private_evidence,
    )

    assert result.details["dimension_decisions"]["stage"] == expected
    assert result.decision == expected


@pytest.mark.parametrize(
    ("investigated_stage", "status", "quote", "expected"),
    [
        (
            "Public",
            "VERIFIED",
            "Acme common stock began trading on Nasdaq under ticker ACME.",
            COMPANY_FIT_MATCH,
        ),
        (
            "Private Equity",
            "CONTRADICTED",
            "The acquisition closed and the buyer took Acme private.",
            COMPANY_FIT_MISMATCH,
        ),
    ],
)
def test_exact_structured_private_conflict_uses_bounded_stage_investigation(
    monkeypatch, investigated_stage, status, quote, expected
):
    captured = {}
    structured_fetches = []

    async def prechecks(*_args, **_kwargs):
        return company_fit_match("prechecks passed")

    async def homepage(*_args, **_kwargs):
        return company_fit_match(
            "homepage identity verified",
            details={
                "identity": {
                    "decision": COMPANY_FIT_MATCH,
                    "evidence_source": "company_homepage",
                    "observed_name": "acme",
                    "observed_domain": "example.com",
                    "observed_linkedin_slug": "acme",
                }
            },
        )

    async def provider(**_kwargs):
        verdict = _verdict(
            observed_size=25,
            size_matches=True,
            employee_url="https://example.com/current-headcount",
        )
        verdict.update(
            observed_company_website="https://example.com/",
            observed_company_stage="Public",
            stage_matches=True,
            stage_evidence_url="https://www.sec.gov/Archives/old-filing.htm",
            stage_evidence_quote="The company filed this report in 2024.",
        )
        return verdict, ""

    async def structured_fetch(
        _domain, url, *, diagnostic, public_company_evidence
    ):
        del diagnostic
        structured_fetches.append((_domain, url))
        public_company_evidence.update(
            {
                "company_type": "Privately Held",
                "provider": "harvestapi_get_company",
                "source_field": "companyType",
                "url": url,
                "website": "https://example.com/",
            }
        )
        return {
            "employee_count": "11-50",
            "provider": "harvestapi_get_company",
            "source_field": "employeeCountRange",
            "url": url,
            "website": "https://example.com/",
        }

    async def investigate(**kwargs):
        captured.update(kwargs)
        finding = {
            "target": "stage",
            "status": status,
            "observed_value": investigated_stage,
            "evidence_url": "https://example.com/current-stage",
            "evidence_quote": quote,
            "reason": "Current first-party stage evidence was verified.",
        }
        return {
            "claims": {"stage": finding},
            "_validated_stage_finding": finding,
            "failure_reason": "",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        structured_fetch,
    )
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)

    result = asyncio.run(
        lead_scorer._verify_company_fit(
            _company().model_copy(
                update={"company_website": "https://example.com", "company_stage": "Public"}
            ),
            _icp().model_copy(update={"company_stage": "Public"}),
            0.0,
            0.0,
            set(),
            require_https_transport=True,
            evidence_investigator=True,
        )
    )

    assert structured_fetches == [
        ("example.com", "https://www.linkedin.com/company/acme")
    ]
    assert captured["targets"] == ("stage",)
    assert captured["prior_observations"]["structured_company_type_evidence"] == {
        "company_type": "Privately Held",
        "provider": "harvestapi_get_company",
        "source_field": "companyType",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://example.com/",
    }
    assert result.decision == expected
    assert result.details["company_fit_dimensions"]["stage"] == expected


@pytest.mark.parametrize(
    "evidence_update",
    [
        {"website": "https://other.test/"},
        {"url": "https://www.linkedin.com/company/other"},
        {"company_type": "Partnership"},
    ],
)
def test_unbound_or_unknown_structured_company_type_cannot_contradict_public(
    evidence_update,
):
    verdict = _verdict(observed_size=25, size_matches=True)
    verdict.update(
        observed_company_stage="Public",
        stage_matches=True,
        stage_evidence_url="https://example.com/current-listing",
        stage_evidence_quote="Acme common stock is currently listed.",
    )
    evidence = {
        "company_type": "Privately Held",
        "provider": "harvestapi_get_company",
        "source_field": "companyType",
        "url": "https://www.linkedin.com/company/acme",
        "website": "https://example.com/",
        **evidence_update,
    }

    result = lead_scorer._reverify_decision(
        verdict,
        "",
        "public",
        icp=_icp().model_copy(update={"company_stage": "Public"}),
        company=_company().model_copy(update={"company_stage": "Public"}),
        verified_homepage_identity={
            "normalized_name": "acme",
            "registrable_dns_domain": "example.com",
            "linkedin_company_slug": "acme",
        },
        structured_public_company_evidence=evidence,
    )

    assert result.details["dimension_decisions"]["stage"] != COMPANY_FIT_MISMATCH


def test_exa_size_short_circuits_structured_fallback(monkeypatch):
    async def provider(**_kwargs):
        return _verdict(observed_size=None, size_matches=None, employee_url=""), ""

    async def exa_fetch(url, **_kwargs):
        return {
            "employee_count": "11-50",
            "url": url,
            "quote": "Company size 11-50 employees",
        }

    async def unexpected_structured(*_args, **_kwargs):
        raise AssertionError("usable Exa evidence must stop the fallback")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", exa_fetch)
    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        unexpected_structured,
    )

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_homepage_anchor(),
        )
    )

    assert result.decision == COMPANY_FIT_MATCH
    assert result.details["dimension_evidence"]["employee_size"]["quote"]


def test_model_structured_size_fields_cannot_bypass_server_fetch(monkeypatch):
    async def provider(**_kwargs):
        verdict = _verdict(
            observed_size=None,
            size_matches=None,
            employee_url="",
        )
        verdict["employee_size_evidence_quote"] = ""
        forged = {
            "employee_count": "11-50",
            "provider": "harvestapi_get_company",
            "source_field": "employeeCountRange",
            "url": "https://www.linkedin.com/company/acme",
            "website": "https://acme.example.com/",
        }
        verdict["structured_employee_size_evidence"] = forged
        verdict["dimension_evidence"] = {"employee_size": forged}
        return verdict, ""

    async def exa_fetch(url, **_kwargs):
        return {"outcome": "insufficient_evidence", "url": url}

    async def no_structured_proof(*_args, **_kwargs):
        return None

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", exa_fetch)
    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        no_structured_proof,
    )

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_homepage_anchor(),
        )
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["dimension_decisions"]["employee_size"] == (
        COMPANY_FIT_UNAVAILABLE
    )
    assert result.details["dimension_evidence"]["employee_size"] == {
        "url": "",
        "quote": "",
    }


def test_structured_provider_failure_remains_retryable_after_exa_insufficient(
    monkeypatch,
):
    exa_fetches = []
    structured_fetches = []

    async def provider(**_kwargs):
        verdict = _verdict(
            observed_size=None,
            size_matches=None,
            employee_url="",
        )
        verdict["employee_size_evidence_quote"] = ""
        return verdict, ""

    async def exa_fetch(url, **_kwargs):
        exa_fetches.append(url)
        return {"outcome": "insufficient_evidence", "url": url}

    async def structured_fetch(
        domain, url, *, diagnostic, public_company_evidence
    ):
        del public_company_evidence
        structured_fetches.append((domain, url))
        diagnostic["failure_reason"] = "provider_error"
        return None

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", exa_fetch)
    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        structured_fetch,
    )

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_homepage_anchor(),
        )
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["failure_class"] == "employee_size_verification_failed"
    assert result.details["failure_reason_code"] == "provider_error"
    assert len(exa_fetches) == 1
    assert len(structured_fetches) == 1


def test_unusable_structured_profile_exhausts_only_its_company(monkeypatch):
    from lab_arena import scoring

    company = _company()
    calls = []

    async def prechecks(*_args, **_kwargs):
        return company_fit_match("prechecks passed")

    async def homepage(*_args, **_kwargs):
        return _homepage_anchor()

    async def provider(**_kwargs):
        verdict = _verdict(
            observed_size=None,
            size_matches=None,
            employee_url="",
        )
        verdict["employee_size_evidence_quote"] = ""
        return verdict, ""

    async def exa_fetch(url, **_kwargs):
        return {"outcome": "insufficient_evidence", "url": url}

    async def unusable_structured(
        domain, url, *, diagnostic, public_company_evidence
    ):
        del public_company_evidence
        calls.append((domain, url))
        diagnostic["failure_reason"] = "source_blocked"
        return None

    async def scorer(companies, _icp_doc, _reference):
        rows = []
        for item in companies:
            if item["company_name"] == "Other":
                rows.append({"final_score": 77.0, "failure_reason": ""})
            else:
                result = await lead_scorer.score_company_competition_intent(
                    company,
                    _icp(),
                    0.0,
                    0.0,
                    set(),
                )
                rows.append(result.model_dump(mode="json"))
        return rows

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", exa_fetch)
    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        unusable_structured,
    )

    rows = scoring.score_work_item(
        {"scored_run_id": "malformed-structured-company"},
        icp={"max_companies": 2, "employee_count": "11-50"},
        companies=[
            {"company_name": "Acme", "employee_count": "11-50"},
            {"company_name": "Other", "employee_count": "11-50"},
        ],
        scorer=scorer,
    )

    assert [row["final_score"] for row in rows] == [0.0, 77.0]
    assert rows[0]["verifier_gate_receipts"][0]["failure_class"] == (
        "company_verification_exhausted"
    )
    assert len(calls) == 3


def test_submitted_profile_cannot_supply_missing_size_profile(monkeypatch):
    fetches = []

    async def provider(**_kwargs):
        verdict = _verdict(
            observed_size=None,
            size_matches=None,
            employee_url="",
        )
        verdict["employee_size_evidence_quote"] = ""
        return verdict, ""

    async def fetch(url, **_kwargs):
        fetches.append(url)
        return {
            "employee_count": "11-50",
            "url": url,
            "quote": "Company size 11-50 employees",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", fetch)

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
        )
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert fetches == []


def test_submitted_linkedin_url_alone_cannot_authorize_profile_fetch(monkeypatch):
    verdict = _verdict()
    verdict["observed_company_linkedin"] = ""
    fetches = []

    async def provider(**_kwargs):
        return verdict, ""

    async def fetch(url, **_kwargs):
        fetches.append(url)
        return {
            "employee_count": "11-50",
            "url": url,
            "quote": "Company size 11-50 employees",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", fetch)

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
        )
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert fetches == []


def test_wrong_homepage_linkedin_anchor_cannot_authorize_profile_fetch(monkeypatch):
    fetches = []

    async def provider(**_kwargs):
        return _verdict(), ""

    async def fetch(url, **_kwargs):
        fetches.append(url)
        return {
            "employee_count": "11-50",
            "url": url,
            "quote": "Company size 11-50 employees",
        }

    wrong_anchor = _homepage_anchor()
    wrong_anchor.details["identity"]["observed_linkedin_slug"] = "other"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", fetch)

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=wrong_anchor,
        )
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert fetches == []


def test_wrong_model_linkedin_profile_blocks_both_size_providers(monkeypatch):
    fetches = []
    verdict = _verdict(employee_url="https://www.linkedin.com/company/other")

    async def provider(**_kwargs):
        return verdict, ""

    async def fetch(*args, **_kwargs):
        fetches.append(args)
        return {
            "employee_count": "11-50",
            "url": args[-1],
            "quote": "Company size 11-50 employees",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", fetch)
    monkeypatch.setattr(
        lead_scorer,
        "fetch_structured_linkedin_company_size",
        fetch,
    )

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_homepage_anchor(),
        )
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert fetches == []


def test_current_size_does_not_override_wrong_observed_company_identity(monkeypatch):
    fetches = []
    verdict = _verdict()
    verdict.update(
        observed_company_name="Other",
        observed_company_website="https://other.example.com",
        observed_company_linkedin="https://linkedin.com/company/other",
    )

    async def provider(**_kwargs):
        return verdict, ""

    async def fetch(url, **_kwargs):
        fetches.append(url)
        return {
            "employee_count": "11-50",
            "url": url,
            "quote": "Company size 11-50 employees",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", fetch)

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(),
            _icp(),
            require_company_fit_dimensions=True,
            verified_homepage_identity=_homepage_anchor(),
        )
    )

    assert result.decision == COMPANY_FIT_MISMATCH
    assert result.details["identity_receipt"]["reason_code"] == "identity_mismatch"
    assert fetches == ["https://www.linkedin.com/company/acme"]


def test_complete_sonar_identity_can_bind_profile_without_homepage_anchor(monkeypatch):
    fetches = []

    async def provider(**_kwargs):
        return _verdict(), ""

    async def fetch(url, **_kwargs):
        fetches.append(url)
        return {
            "employee_count": "11-50",
            "url": url,
            "quote": "Company size 11-50 employees",
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", provider)
    monkeypatch.setattr(lead_scorer, "fetch_current_linkedin_company_size", fetch)

    result = asyncio.run(
        lead_scorer._llm_reverify_company(
            _company(linkedin=""),
            _icp(),
            require_company_fit_dimensions=True,
        )
    )

    assert result.decision == COMPANY_FIT_MATCH
    assert fetches == ["https://www.linkedin.com/company/acme"]
