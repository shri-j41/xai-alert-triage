"""
Context enrichment module.

Takes a raw alerts DataFrame and appends five SME-specific context columns:
  - asset_criticality   : int  1-5  (5 = most critical)
  - business_function   : str       (e.g. "Finance", "IT Infrastructure")
  - internet_exposed    : bool      (True if asset is reachable from internet)
  - user_role           : str       (e.g. "Privileged", "Standard", "Service")
  - outside_working_hours: bool     (True if alert fired Mon-Fri 08:00-18:00 NPT)

These columns reflect the operational reality of a Nepali SME where context
about *who owns the asset* and *when* an event happened is critical for triage.
"""

import re
from typing import Optional

import pandas as pd

# ── SME asset knowledge base ──────────────────────────────────────────────────
# Each entry reflects a realistic Nepali SME asset profile.

ASSET_PROFILES: dict[str, dict] = {
    "fin-server-01":    {"criticality": 5, "function": "Finance",          "internet_exposed": False},
    "erp-server-01":    {"criticality": 4, "function": "Operations",       "internet_exposed": False},
    "dc-server-01":     {"criticality": 5, "function": "IT Infrastructure","internet_exposed": False},
    "web-server-01":    {"criticality": 3, "function": "Marketing/Web",    "internet_exposed": True},
    "db-server-01":     {"criticality": 5, "function": "Data Management",  "internet_exposed": False},
    "backup-server-01": {"criticality": 4, "function": "IT Infrastructure","internet_exposed": False},
    "workstation-acct-01": {"criticality": 3, "function": "Finance",       "internet_exposed": False},
    "workstation-acct-02": {"criticality": 3, "function": "Finance",       "internet_exposed": False},
    "workstation-hr-01":   {"criticality": 2, "function": "HR",            "internet_exposed": False},
    "workstation-mgmt-01": {"criticality": 4, "function": "Management",    "internet_exposed": False},
    "pos-terminal-01":  {"criticality": 3, "function": "Sales/POS",        "internet_exposed": True},
    "pos-terminal-02":  {"criticality": 3, "function": "Sales/POS",        "internet_exposed": True},
    "laptop-sales-01":  {"criticality": 2, "function": "Sales",            "internet_exposed": False},
    "laptop-remote-01": {"criticality": 2, "function": "Remote Work",      "internet_exposed": True},
    "printer-office-01":{"criticality": 1, "function": "Office",           "internet_exposed": False},
}

# Default for any asset not in the knowledge base
_DEFAULT_ASSET = {"criticality": 2, "function": "Unknown", "internet_exposed": False}

# ── User role classification ───────────────────────────────────────────────────

USER_ROLES: dict[str, str] = {
    "admin":           "Privileged",
    "sysadmin":        "Privileged",
    "fin_manager":     "Privileged",
    "ceo":             "Privileged",
    "it_support":      "Privileged",
    "service_account": "Service",
    "SYSTEM":          "Service",
    "accountant1":     "Standard",
    "accountant2":     "Standard",
    "hr_officer":      "Standard",
    "sales_exec1":     "Standard",
    "sales_exec2":     "Standard",
    "workstation-mgmt-01": "Standard",
    "guest_user":      "Guest",
}

_DEFAULT_ROLE = "Standard"

# ── Keyword fallback (second tier, only used when the exact-match lookups
#    above find nothing) ─────────────────────────────────────────────────────
# Each rule is a whole-token match against the asset/user name (see
# _tokenize/_keyword_matches below) — NOT a raw substring match. Raw
# substring matching let e.g. "web" match inside "AppleWebKit" (a browser
# User-Agent string, not a hostname), silently mislabelling unrelated rows.
# Checked in order; first match wins. Kept short and literal on purpose so
# each rule can be justified individually (see thesis defense notes).

_ASSET_KEYWORD_RULES: list[tuple[tuple[str, ...], int, str, bool]] = [
    # keywords                      criticality  business_function     internet_exposed
    (("fin", "finance", "payment"), 5,           "Finance",            False),  # money-handling systems -> highest criticality, kept internal
    (("dc", "domain", "ad-server"), 5,           "IT Infrastructure",  False),  # domain controllers/AD are core infra -> highest criticality
    (("erp",),                      4,           "Operations",         False),  # ERP runs core business operations
    (("pos",),                      3,           "Sales/POS",          True),   # POS terminals are customer-facing -> internet exposed
    (("web", "www"),                3,           "Marketing/Web",      True),   # web servers are public-facing by definition
    (("dev", "test", "staging"),    1,           "Development/Test",  False),  # non-production -> lowest criticality
]

# Signals internet exposure alone when no asset rule above matched.
_EXPOSURE_KEYWORDS = ("public", "dmz")

_USER_KEYWORD_RULES: list[tuple[tuple[str, ...], str]] = [
    (("admin", "root", "sysadmin"), "Privileged"),  # administrative accounts
    (("svc", "service", "bot"),     "Service"),      # non-human / service accounts
    (("guest", "temp"),             "Guest"),        # temporary or guest accounts
]


_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _tokenize(name: str) -> list[str]:
    """
    Split into lowercase alphanumeric tokens on any non-alphanumeric
    separator (-, _, ., /, space, ...). Used so keyword rules only match
    whole words/segments of a name — "web" matches "web-prod-03" or
    "web_server_1" (tokens: "web", "prod"/"server", ...) but never a
    substring occurrence like "AppleWebKit" -> one token "applewebkit".
    """
    return [t for t in _TOKEN_SPLIT_RE.split(str(name).lower()) if t]


def _keyword_matches(name_tokens: list[str], keyword: str) -> bool:
    """
    True if `keyword` (itself possibly multi-word, e.g. "ad-server") appears
    as a run of exact, consecutive tokens in name_tokens.
    """
    kw_tokens = _tokenize(keyword)
    if not kw_tokens:
        return False
    n = len(kw_tokens)
    return any(
        name_tokens[i:i + n] == kw_tokens
        for i in range(len(name_tokens) - n + 1)
    )


def _keyword_asset_lookup(asset: str) -> Optional[dict]:
    """
    Second-tier fallback for an affected_asset value not present in
    ASSET_PROFILES. Returns a profile dict on a keyword match, else None
    (caller then falls through to the generic default).
    """
    tokens = _tokenize(asset)
    for keywords, criticality, function, exposed in _ASSET_KEYWORD_RULES:
        if any(_keyword_matches(tokens, kw) for kw in keywords):
            return {"criticality": criticality, "function": function, "internet_exposed": exposed}
    if any(_keyword_matches(tokens, kw) for kw in _EXPOSURE_KEYWORDS):
        return {**_DEFAULT_ASSET, "internet_exposed": True}
    return None


def _keyword_user_lookup(user: str) -> Optional[str]:
    """
    Second-tier fallback for a user value not present in USER_ROLES.
    Returns a role on a keyword match, else None (caller falls through to
    the generic "Standard" default).
    """
    tokens = _tokenize(user)
    for keywords, role in _USER_KEYWORD_RULES:
        if any(_keyword_matches(tokens, kw) for kw in keywords):
            return role
    return None


def _resolve_asset(asset: str) -> tuple[dict, str]:
    """Resolve one asset's context: exact match -> keyword fallback -> default."""
    if asset in ASSET_PROFILES:
        return ASSET_PROFILES[asset], "exact_match"
    keyword_hit = _keyword_asset_lookup(asset)
    if keyword_hit is not None:
        return keyword_hit, "keyword_fallback"
    return _DEFAULT_ASSET, "default"


def _resolve_user(user: str) -> tuple[str, str]:
    """Resolve one user's role: exact match -> keyword fallback -> default."""
    if user in USER_ROLES:
        return USER_ROLES[user], "exact_match"
    keyword_hit = _keyword_user_lookup(user)
    if keyword_hit is not None:
        return keyword_hit, "keyword_fallback"
    return _DEFAULT_ROLE, "default"


# ── Working hours (Nepal Standard Time UTC+5:45) ───────────────────────────────
# Nepali SMEs typically operate Mon-Fri 10:00-18:00; we use 08:00-18:00
# to give a reasonable window that includes early IT activity.
_WORK_HOUR_START = 8
_WORK_HOUR_END   = 18
# Weekdays: Monday=0 … Friday=4


def _is_outside_working_hours(timestamp_str: str) -> bool:
    """Return True if the alert timestamp falls outside Mon-Fri 08:00-18:00."""
    try:
        ts = pd.to_datetime(timestamp_str)
        if ts.weekday() >= 5:          # Saturday or Sunday
            return True
        return not (_WORK_HOUR_START <= ts.hour < _WORK_HOUR_END)
    except Exception:
        return False


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add context columns to a raw alerts DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: affected_asset, user, timestamp.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with five new columns appended.
    """
    df = df.copy()

    # Asset-level context: exact match -> keyword fallback -> default.
    # asset_context_source records which tier resolved each row.
    asset_resolved = df["affected_asset"].map(_resolve_asset)
    df["asset_criticality"]    = asset_resolved.map(lambda t: t[0]["criticality"])
    df["business_function"]    = asset_resolved.map(lambda t: t[0]["function"])
    df["internet_exposed"]     = asset_resolved.map(lambda t: t[0]["internet_exposed"])
    df["asset_context_source"] = asset_resolved.map(lambda t: t[1])

    # User-level context: exact match -> keyword fallback -> default.
    user_resolved = df["user"].map(_resolve_user)
    df["user_role"]           = user_resolved.map(lambda t: t[0])
    df["user_context_source"] = user_resolved.map(lambda t: t[1])

    # Temporal context
    df["outside_working_hours"] = df["timestamp"].map(_is_outside_working_hours)

    return df


def enrichment_coverage(df: pd.DataFrame) -> dict:
    """
    Report how many rows' affected_asset / user context came from an exact
    match, a keyword fallback, or the generic default. asset_criticality,
    business_function and internet_exposed are all derived from
    affected_asset (one lookup); user_role is derived from user.

    Requires enrich() to have already run (needs asset_context_source /
    user_context_source columns) — used to warn the analyst about how much
    of an uploaded CSV's context is real vs guessed vs generic.
    """
    total = len(df)

    def _tier_counts(source_col: str) -> dict:
        counts = df[source_col].value_counts()
        return {
            "exact_match":     int(counts.get("exact_match", 0)),
            "keyword_fallback": int(counts.get("keyword_fallback", 0)),
            "default":         int(counts.get("default", 0)),
        }

    return {
        "total_rows": total,
        "asset": _tier_counts("asset_context_source"),
        "user":  _tier_counts("user_context_source"),
    }


if __name__ == "__main__":
    # Quick smoke test
    import sys
    sys.path.insert(0, ".")
    from data.generate_alerts import generate

    raw = generate(n=10, out_path="data/alerts_raw_test.csv")
    enriched = enrich(raw)
    print(enriched[[
        "alert_id", "affected_asset", "asset_criticality",
        "business_function", "internet_exposed",
        "user", "user_role", "outside_working_hours"
    ]].to_string())
