"""
Role Format Normalization
=========================
Normalizes miner-submitted role titles for consistent formatting.
Called during consensus (epoch_lifecycle.py) for approved leads,
before the final leads_private update.

Pipeline:
1. Unicode cleanup (invisible chars, mojibake)
2. Dotted abbreviation conversion (C.E.O. → CEO)
3. Dot stripping from common abbreviations (Sr. → Sr)
4. 'and' → '&'
5. '/' → '&' (except abbreviation pairs like UI/UX)
6. Comma title conversion (Co-Founder, CTO → Co-Founder & CTO)
7. Abbreviation expansion (Sr → Senior, VP → Vice President)
8. Word-level casing (uppercase abbrevs, special words, title case)
"""

import re

# ─── UPPERCASE ABBREVIATIONS ──────────────────────────────────────────────
UPPERCASE_WORDS = {
    # C-Suite
    'ceo', 'cfo', 'cto', 'coo', 'cmo', 'cio', 'cro', 'cpo', 'cso', 'cdo', 'cco',
    'cao', 'clo', 'cno', 'cvo', 'cwo',
    # VP levels
    'vp', 'svp', 'evp', 'avp', 'gvp',
    # Departments / functions
    'it', 'hr', 'qa', 'ai', 'ux', 'ui', 'pr', 'bi', 'ml', 'nlp',
    # Business
    'gm', 'pm', 'pmo', 'gtm', 'sdr', 'bdr', 'smb', 'sme',
    # Tech
    'sre', 'seo', 'sem', 'sql', 'api', 'sdk', 'gcp', 'aws', 'crm', 'erp', 'sdet',
    'sap', 'cad', 'cnc', 'bim', 'etl', 'rpa', 'hris', 'ats', 'lms',
    'cms', 'cdn', 'dns', 'ssl', 'vpn', 'sla', 'kpi', 'okr', 'roi',
    'fpga', 'asic', 'plc', 'scada', 'mes', 'dba',
    'edi', 'vfx', 'gis', 'rf',
    # Industry
    'hvac', 'leed', 'oem', 'cpg', 'fmcg', 'mep',
    'bom', 'mrp', 'wms', 'tms', 'plm', 'pdm', 'mro', 'epc',
    # Regions
    'emea', 'apac', 'latam', 'amer', 'dach', 'apj', 'anz', 'mea', 'nam',
    # Geography
    'uk', 'us', 'usa',
    # Compliance / Government
    'esg', 'dei', 'sox', 'gdpr', 'fda', 'hipaa', 'pci', 'iso',
    'aml', 'kyc', 'bsa', 'osha', 'faa', 'irs', 'epa', 'dea',
    'fcc', 'sec', 'cfpb', 'finra', 'ferc', 'nrc', 'doj', 'hud',
    # Finance / Accounting
    'ipo', 'ebitda', 'npi', 'gaap', 'ifrs', 'capex', 'opex',
    'rfp', 'rfi', 'rfq', 'adr', 'fp',
    # Credentials
    'mba', 'pmp', 'csm', 'cspo', 'itil', 'cissp', 'cism', 'cisa',
    'cpa', 'cfa', 'cfp', 'clu', 'chfc', 'frm',
    'ccna', 'ccnp', 'mcse', 'rhce', 'ceh',
    'eit', 'jd',
    # Medical
    'rn', 'bsn', 'msn', 'lpn', 'cna', 'dvm', 'dds', 'rph',
    'bcba', 'lcsw', 'lpc', 'lmft', 'lmhc',
    'fnp', 'aprn', 'crna', 'crnp', 'np',
    # Other
    'llc', 'ip', 'cx', 'hse', 'ehs', 'qc',
    'iam', 'grc', 'soc', 'rcm', 'hcm',
    'osp', 'mfg', 'mts',
    'sba', 'csr', 'ppc', 'dtc',
    'ems', 'noc', 'pnc',
    'ar', 'vr', 'xr', 'tv', 'dc',
    'pv', 'scp',
    # Roles/titles
    'rvp', 'cxo', 'ciso', 'hrbp', 'shrm', 'nmls', 'sphr',
    # Industry-specific
    'aia', 'cmc', 'coe', 'cra', 'cre', 'ndt', 'npd', 'epm',
    'otc', 'sqa', 'sde', 'scm', 'vdc', 'vip', 'peo',
    'dod', 'ehr', 'ev', 'hw', 'sw', 'ic', 'pc',
    'msp', 'msw', 'msc', 'phr', 'eu', 'nyc',
    'php', 'pt', 'ot',
    # Additional confirmed abbreviations
    'ap', 'ad', 'av', 'bd', 'ae', 'ea', 'ta', 'bu', 'bc', 'ww',
    'ag', 'cs', 'cp', 'pe', 'md', 'fx',
    'iida', 'ncidq', 'noma', 'edac', 'pnw', 'vc',
}

# ─── MINOR WORDS ──────────────────────────────────────────────────────────
MINOR_WORDS = {
    'of', 'and', 'in', 'for', 'the', 'at', 'to', 'by', 'on',
    'or', 'as', 'a', 'an', 'with', 'from', 'but', 'nor', 'so', 'yet',
    'vs', 'via', 'per',
}

# ─── ROMAN NUMERALS ───────────────────────────────────────────────────────
ROMAN_NUMERALS = {
    'ii': 'II', 'iii': 'III', 'iv': 'IV',
    'vi': 'VI', 'vii': 'VII', 'viii': 'VIII',
    'ix': 'IX', 'xi': 'XI', 'xii': 'XII',
    'xiii': 'XIII', 'xiv': 'XIV', 'xv': 'XV',
    'll': 'II', 'lll': 'III',
}

# ─── SPECIAL CASING ───────────────────────────────────────────────────────
SPECIAL_WORDS = {
    'devops': 'DevOps', 'devsecops': 'DevSecOps', 'finops': 'FinOps',
    'mlops': 'MLOps', 'secops': 'SecOps', 'dataops': 'DataOps',
    'aiops': 'AIOps', 'gitops': 'GitOps',
    'saas': 'SaaS', 'iaas': 'IaaS', 'paas': 'PaaS',
    'ios': 'iOS', 'iphone': 'iPhone', 'ipad': 'iPad',
    'iot': 'IoT', 'iiot': 'IIoT',
    'macos': 'macOS', 'watchos': 'watchOS', 'tvos': 'tvOS',
    'linkedin': 'LinkedIn', 'phd': 'PhD', '.net': '.NET',
    # Tech products
    'servicenow': 'ServiceNow', 'hubspot': 'HubSpot', 'sharepoint': 'SharePoint',
    'javascript': 'JavaScript', 'typescript': 'TypeScript',
    'youtube': 'YouTube', 'youtuber': 'YouTuber',
    'tiktok': 'TikTok', 'mysql': 'MySQL', 'postgresql': 'PostgreSQL',
    'mongodb': 'MongoDB', 'openai': 'OpenAI', 'whatsapp': 'WhatsApp',
    'powerbi': 'PowerBI', 'restful': 'RESTful', 'wordpress': 'WordPress',
    'shopify': 'Shopify', 'github': 'GitHub', 'gitlab': 'GitLab',
    'bitbucket': 'Bitbucket', 'elasticsearch': 'Elasticsearch',
    'snowflake': 'Snowflake', 'databricks': 'Databricks',
    'salesforce': 'Salesforce', 'workday': 'Workday', 'netsuite': 'NetSuite',
    'marketo': 'Marketo', 'tableau': 'Tableau', 'appsheet': 'AppSheet',
    'outsystems': 'OutSystems', 'clickup': 'ClickUp',
    # Compound abbreviations
    'fp&a': 'FP&A', 'r&d': 'R&D', 'm&a': 'M&A', 'p&l': 'P&L',
    'eh&s': 'EH&S', 'iv&v': 'IV&V',
    'b2b': 'B2B', 'b2c': 'B2C', 'd2c': 'D2C', 'c2c': 'C2C',
    'b2g': 'B2G', 'p2p': 'P2P', 's2p': 'S2P', 'o2c': 'O2C',
}

# ─── DOTTED ABBREVIATIONS ─────────────────────────────────────────────────
_DOTTED_ABBREVS = [
    ('c.e.o.', 'CEO'), ('c.e.o', 'CEO'),
    ('c.f.o.', 'CFO'), ('c.f.o', 'CFO'),
    ('c.t.o.', 'CTO'), ('c.t.o', 'CTO'),
    ('c.o.o.', 'COO'), ('c.o.o', 'COO'),
    ('c.m.o.', 'CMO'), ('c.m.o', 'CMO'),
    ('c.i.o.', 'CIO'), ('c.i.o', 'CIO'),
    ('e.i.t.', 'EIT'), ('e.i.t', 'EIT'),
    ('m.b.a.', 'MBA'), ('m.b.a', 'MBA'),
    ('ph.d.', 'PhD'), ('ph.d', 'PhD'),
    ('v.p.', 'VP'), ('i.t.', 'IT'), ('h.r.', 'HR'), ('a.i.', 'AI'),
    ('p.e.', 'PE'), ('r.n.', 'RN'), ('j.d.', 'JD'),
    ('b.s.', 'BS'), ('m.s.', 'MS'), ('b.a.', 'BA'), ('m.a.', 'MA'),
    ('m.d.', 'MD'), ('d.o.', 'DO'), ('o.d.', 'OD'), ('u.s.', 'US'),
    ('c.s.o.', 'CSO'), ('c.s.o', 'CSO'),
    ('c.b.o.', 'CBO'), ('c.b.o', 'CBO'),
    ('c.p.a.', 'CPA'), ('c.p.a', 'CPA'),
    ('l.l.c.', 'LLC'), ('l.l.c', 'LLC'),
    ('l.i.o.n.', 'LION'), ('l.i.o.n', 'LION'),
    ('d.a.ch.', 'DACH'), ('d.a.ch', 'DACH'),
    ('e.d.', 'EdD'), ('q.c.', 'QC'), ('s.e.', 'SE'),
    ('g.m.', 'GM'), ('d.c.', 'DC'),
]
_DOTTED_ABBREVS.sort(key=lambda x: -len(x[0]))

_DOTTED_PATTERNS = [
    (re.compile(re.escape(dotted), re.I), repl)
    for dotted, repl in _DOTTED_ABBREVS
]

# ─── UNICODE CLEANUP ──────────────────────────────────────────────────────
_INVISIBLE_CHARS = re.compile(
    r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u202a\u202c\u2069\u2066\u2067\u2068'
    r'\ufe0f\ufe0e]'
)

_MOJIBAKE_MAP = [
    ('\u00e2\u20ac\u2122', '\u2019'),  # â€™ → '
    ('\u00e2\u20ac\u0153', '\u201c'),  # â€œ → "
    ('\u00e2\u20ac\u009d', '\u201d'),  # â€\x9d → "
    ('\u00e2\u20ac\u201d', '\u2014'),  # â€" → —
    ('\u00e2\u20ac\u201c', '\u2013'),  # â€" → –
    ('\u00e2\u20ac\u00a2', '\u2022'),  # â€¢ → •
    ('\u00c2\u00bb', '\u00bb'),        # Â» → »
    ('\u00c2\u00a0', ' '),             # Â  → space
]

_ORDINAL_RE = re.compile(r'^(\d+)(st|nd|rd|th)$', re.I)
_DIGIT_UPPER_RE = re.compile(r'^\d+[A-Z]+$')
_MC_MAC_RE = re.compile(r'^(Mc|Mac)([A-Z][a-z]+.*)$')

# ─── DOTTED ABBREVIATION STRIP ────────────────────────────────────────────
_DOTTED_ABBREV_STRIP = re.compile(
    r'\b(Sr|Jr|Dr|Mr|Mrs|Ms|Prof|Gen|Gov|Sgt|Cpl|Pvt|Capt|Lt|Col|Cmdr'
    r'|Mgr|Dir|Eng|Engr|Dept|Corp|Inc|Ltd|Assoc|Asst|Supt|Admin|Coord'
    r'|Acct|Mfg|Dist|Exec|Oper|Pres|Maint|Natl|Intl)\.',
    re.IGNORECASE
)

# ─── ABBREVIATION EXPANSION ───────────────────────────────────────────────
ABBREV_EXPAND = {
    'sr': 'Senior', 'jr': 'Junior',
    'vp': 'Vice President', 'svp': 'Senior Vice President',
    'evp': 'Executive Vice President', 'avp': 'Assistant Vice President',
    'gvp': 'Group Vice President', 'rvp': 'Regional Vice President',
    'mgr': 'Manager', 'dir': 'Director',
    'eng': 'Engineer', 'engr': 'Engineer',
    'asst': 'Assistant', 'assoc': 'Associate', 'exec': 'Executive',
    'dept': 'Department', 'coord': 'Coordinator', 'admin': 'Administrator',
    'acct': 'Accountant', 'supt': 'Superintendent',
    'mfg': 'Manufacturing', 'dist': 'District',
    'oper': 'Operations', 'pres': 'President',
    'maint': 'Maintenance', 'natl': 'National', 'intl': 'International',
    'rep': 'Representative', 'supv': 'Supervisor',
    'svc': 'Service', 'svcs': 'Services',
    'mktg': 'Marketing', 'acctg': 'Accounting',
    'tech': 'Technician', 'snr': 'Senior', 'gm': 'General Manager',
}

# ─── TITLE WORDS (for comma→& conversion) ─────────────────────────────────
TITLE_WORDS = {
    'ceo', 'cfo', 'cto', 'coo', 'cmo', 'cio', 'cro', 'cpo', 'cso', 'cdo', 'cco',
    'cao', 'clo', 'cxo', 'ciso',
    'founder', 'co-founder', 'cofounder',
    'president', 'vice', 'owner', 'partner', 'managing',
    'chairman', 'chairwoman', 'chairperson', 'chair',
    'principal', 'chief', 'head', 'executive',
    'director', 'manager', 'trustee',
}


def _clean_unicode(role):
    """Remove invisible chars and fix mojibake."""
    role = _INVISIBLE_CHARS.sub('', role)
    for bad, good in _MOJIBAKE_MAP:
        if bad in role:
            role = role.replace(bad, good)
    role = role.replace('\u00a0', ' ')
    role = role.replace('\u2011', '-')
    role = role.replace('\u00ad', '')
    return role


def _clean_whitespace(role):
    """Collapse multiple spaces, strip."""
    role = role.strip()
    role = re.sub(r' {2,}', ' ', role)
    return role


def _fix_dotted_abbreviations(role):
    """Convert C.E.O. → CEO, V.P. → VP, etc."""
    for pattern, repl in _DOTTED_PATTERNS:
        role = pattern.sub(repl, role)
    return role


def _strip_trailing_dots(line):
    """Strip dots from common abbreviations: Sr. → Sr, Jr. → Jr, etc."""
    return _DOTTED_ABBREV_STRIP.sub(lambda m: m.group(1), line)


def _normalize_and(line):
    """Normalize ' and ' → ' & ' (case-insensitive)."""
    return re.sub(r'\s+[Aa][Nn][Dd]\s+', ' & ', line)


def _normalize_slashes(line):
    """Normalize slashes to ampersand, but keep abbreviation pairs like UI/UX."""
    def _slash_repl(m):
        before = m.group(1)
        after = m.group(2)
        if re.match(r'^[A-Za-z]{2,4}$', before) and re.match(r'^[A-Za-z]{2,4}$', after):
            return f'{before}/{after}'
        return f'{before} & {after}'
    return re.sub(r'(\S+)\s*/\s*(\S+)', _slash_repl, line)


def _normalize_comma_titles(line):
    """Convert 'Title, Title' commas to &, leave 'Title, Department' as-is."""
    if ', ' not in line:
        return line
    parts = line.split(', ')
    result = [parts[0]]
    for i in range(1, len(parts)):
        first_word = parts[i].strip().split()[0].lower() if parts[i].strip() else ''
        if first_word in TITLE_WORDS:
            result.append(' & ' + parts[i])
        else:
            result.append(', ' + parts[i])
    return ''.join(result)


def _expand_abbreviations(line):
    """Expand common abbreviations to full form: Sr → Senior, VP → Vice President, etc."""
    words = line.split(' ')
    result = []
    for word in words:
        prefix = ''
        suffix = ''
        core = word
        while core and not core[0].isalnum():
            prefix += core[0]
            core = core[1:]
        while core and not core[-1].isalnum():
            suffix = core[-1] + suffix
            core = core[:-1]
        core_lower = core.lower() if core else ''
        if core_lower in ABBREV_EXPAND:
            result.append(prefix + ABBREV_EXPAND[core_lower] + suffix)
        else:
            result.append(word)
    return ' '.join(result)


def _normalize_word(word, is_first=False):
    """Normalize a single word: abbreviations, casing, special words."""
    if not word:
        return word

    word_lower = word.lower()

    # Slash-separated (Ux/Ui → UX/UI)
    if '/' in word and len(word) > 1:
        parts = word.split('/')
        return '/'.join(
            _normalize_word(p, is_first and i == 0)
            for i, p in enumerate(parts)
        )

    # Hyphen-separated (Co-founder → Co-Founder)
    if '-' in word and len(word) > 2 and not word.startswith('-'):
        parts = word.split('-')
        return '-'.join(
            _normalize_word(p, is_first and i == 0)
            for i, p in enumerate(parts)
        )

    # Full-word special check before stripping
    if word_lower in SPECIAL_WORDS:
        return SPECIAL_WORDS[word_lower]
    stripped = word_lower.rstrip('.,;:!?)')
    if stripped in SPECIAL_WORDS:
        trail = word[len(stripped):]
        return SPECIAL_WORDS[stripped] + trail

    # Strip surrounding punctuation
    prefix = ''
    suffix = ''
    core = word
    while core and not core[0].isalnum():
        prefix += core[0]
        core = core[1:]
    while core and not core[-1].isalnum():
        suffix = core[-1] + suffix
        core = core[:-1]

    if not core:
        return word

    # Handle (.Net) → (.NET)
    if prefix.endswith('.') and ('.' + core.lower()) in SPECIAL_WORDS:
        return prefix[:-1] + SPECIAL_WORDS['.' + core.lower()] + suffix

    core_lower = core.lower()

    # Special casing (DevOps, SaaS, etc.)
    if core_lower in SPECIAL_WORDS:
        return prefix + SPECIAL_WORDS[core_lower] + suffix

    # Uppercase abbreviations
    if core_lower in UPPERCASE_WORDS:
        return prefix + core.upper() + suffix

    # Roman numerals
    if core_lower in ROMAN_NUMERALS:
        return prefix + ROMAN_NUMERALS[core_lower] + suffix

    # Ordinals (1St → 1st)
    m = _ORDINAL_RE.match(core)
    if m:
        return prefix + m.group(1) + m.group(2).lower() + suffix

    # Keep digit+uppercase patterns (3D, 5G, 4K)
    if _DIGIT_UPPER_RE.match(core):
        return prefix + core + suffix

    # Minor words (lowercase unless first word)
    if not is_first and core_lower in MINOR_WORDS:
        return prefix + core_lower + suffix

    # Preserve McX/MacX patterns (McKinsey, McDonald, MacArthur)
    mc_match = _MC_MAC_RE.match(core)
    if mc_match:
        return prefix + core + suffix
    # Fix lowercased Mc names: mckinsey → McKinsey
    if core_lower.startswith('mc') and len(core) > 2:
        return prefix + 'Mc' + core[2:].capitalize() + suffix

    # Default: title case
    return prefix + core.capitalize() + suffix


def _normalize_line(line):
    """Normalize a single line of a role."""
    line = _clean_whitespace(line)
    line = _fix_dotted_abbreviations(line)
    line = _strip_trailing_dots(line)
    if not line:
        return line
    line = _normalize_and(line)
    line = _normalize_slashes(line)
    line = _normalize_comma_titles(line)
    line = _expand_abbreviations(line)
    words = line.split(' ')
    return ' '.join(_normalize_word(w, is_first=(i == 0)) for i, w in enumerate(words))


def normalize_role_format(role):
    """
    Normalize a role string for consistent formatting.

    Returns the normalized role, or the original if no changes needed.
    Safe to call with None or empty strings.
    """
    if not role or not isinstance(role, str) or not role.strip():
        return role

    result = _clean_unicode(role)

    if '\n' in result:
        lines = result.split('\n')
        return '\n'.join(_normalize_line(l) for l in lines)

    return _normalize_line(result)
