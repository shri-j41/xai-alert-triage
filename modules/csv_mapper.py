"""
CSV column mapper.

Uploaded CSVs rarely use our exact column names. This module:
  1. Tries to auto-map uploaded columns to the seven required fields using
     a dictionary of known aliases (case-insensitive, whitespace-normalised).
  2. For any column that cannot be auto-mapped, returns a mapping spec so the
     Streamlit caller can render a selectbox and let the analyst choose.
  3. After mapping is confirmed, normalises the DataFrame into our canonical
     schema and fills safe defaults for any truly absent columns.

Required canonical columns
--------------------------
alert_id           unique identifier for the alert
timestamp          datetime string
source_tool        originating tool / sensor name
alert_category     one of the 12 standard categories
severity           Low / Medium / High / Critical
affected_asset     hostname or asset identifier
user               username associated with the event
source_ip          IP address of the event origin
alert_description  free-text description
"""

from __future__ import annotations

import difflib
import re
import uuid
from typing import Optional

import pandas as pd

# ── Canonical schema ──────────────────────────────────────────────────────────

REQUIRED_COLS = [
    "alert_id",
    "timestamp",
    "source_tool",
    "alert_category",
    "severity",
    "affected_asset",
    "user",
    "source_ip",
    "alert_description",
]

# Columns the pipeline can fabricate if truly absent
OPTIONAL_WITH_DEFAULT = {
    "alert_id":          lambda df: [f"UP-{i+1:04d}" for i in range(len(df))],
    "source_tool":       lambda df: ["Unknown"] * len(df),
    "alert_category":    lambda df: ["Policy Violation"] * len(df),
    "severity":          lambda df: ["Medium"] * len(df),
    "user":              lambda df: ["unknown"] * len(df),
    "source_ip":         lambda df: ["0.0.0.0"] * len(df),
    "alert_description": lambda df: ["No description provided."] * len(df),
}

# ── Alias dictionary ──────────────────────────────────────────────────────────
# Keys are normalised upload column names → canonical column name.
# Add more aliases here as you encounter new export formats.

ALIASES: dict[str, str] = {
    # alert_id
    "id": "alert_id", "alert_id": "alert_id", "alertid": "alert_id",
    "alert id": "alert_id", "event_id": "alert_id", "eventid": "alert_id",
    "uid": "alert_id",

    # timestamp
    "timestamp": "timestamp", "time": "timestamp", "datetime": "timestamp",
    "event_time": "timestamp", "eventtime": "timestamp", "date": "timestamp",
    "alert_time": "timestamp", "created_at": "timestamp", "occurred": "timestamp",

    # source_tool
    "source_tool": "source_tool", "tool": "source_tool", "sensor": "source_tool",
    "sourcetool": "source_tool", "source": "source_tool", "product": "source_tool",
    "vendor": "source_tool", "detector": "source_tool",

    # alert_category
    "alert_category": "alert_category", "category": "alert_category",
    "alertcategory": "alert_category", "type": "alert_category",
    "alert_type": "alert_category", "event_type": "alert_category",
    "threat_type": "alert_category", "classification": "alert_category",

    # severity
    "severity": "severity", "level": "severity", "priority": "severity",
    "risk": "severity", "risk_level": "severity", "alert_severity": "severity",
    "criticality": "severity",

    # affected_asset
    "affected_asset": "affected_asset", "asset": "affected_asset",
    "hostname": "affected_asset", "host": "affected_asset",
    "computer": "affected_asset", "computer_name": "affected_asset",
    "device": "affected_asset", "agent_name": "affected_asset",
    "agentname": "affected_asset", "target": "affected_asset",
    "destination": "affected_asset", "dst_host": "affected_asset",

    # user
    "user": "user", "username": "user", "user_name": "user",
    "account": "user", "accountname": "user", "account_name": "user",
    "dstuser": "user", "dst_user": "user", "subject_user": "user",

    # source_ip
    "source_ip": "source_ip", "src_ip": "source_ip", "srcip": "source_ip",
    "source_address": "source_ip", "src_addr": "source_ip",
    "ip_address": "source_ip", "ipaddress": "source_ip",
    "remote_ip": "source_ip", "client_ip": "source_ip",
    "attacker_ip": "source_ip",

    # alert_description
    "alert_description": "alert_description", "description": "alert_description",
    "message": "alert_description", "details": "alert_description",
    "event_description": "alert_description", "summary": "alert_description",
    "rule_description": "alert_description", "rule description": "alert_description",
    "alert_message": "alert_description", "log_message": "alert_description",
    "payload_data": "alert_description", "payload": "alert_description",
}

# ── Severity normaliser ───────────────────────────────────────────────────────

_SEV_NORM: dict[str, str] = {
    "low": "Low", "1": "Low", "info": "Low", "informational": "Low",
    "medium": "Medium", "2": "Medium", "moderate": "Medium", "warning": "Medium",
    "high": "High", "3": "High", "major": "High",
    "critical": "Critical", "4": "Critical", "5": "Critical",
    "emergency": "Critical", "fatal": "Critical",
}


def _normalise_key(s: str) -> str:
    """Lower-case and collapse whitespace/underscores for fuzzy matching."""
    return re.sub(r"[\s_]+", " ", s.strip().lower())


# Pre-build a normalised alias lookup so both "risk_level" and "risk level"
# match correctly regardless of what the upload column uses.
_ALIASES_NORM: dict[str, str] = {
    _normalise_key(k): v for k, v in ALIASES.items()
}

# Reverse index: canonical column -> list of normalised alias keys that
# belong to it. Used as the candidate pool for fuzzy matching.
_ALIAS_KEYS_BY_CANON: dict[str, list[str]] = {}
for _norm_key, _canon in _ALIASES_NORM.items():
    _ALIAS_KEYS_BY_CANON.setdefault(_canon, []).append(_norm_key)

# Minimum similarity score (0-1) required to accept a fuzzy match.
# Kept conservative so unrelated columns don't get silently mismapped.
FUZZY_THRESHOLD = 0.85


# Words shorter than this are too generic to safely trigger an automatic
# whole-word match — short tokens like "id", "time", "type" show up inside
# many unrelated compound column names (e.g. "session_id", "protocol_type").
_MIN_CONTAINMENT_WORD_LEN = 5

# Bare alias words that are long enough to pass the length check above but
# are still too generic to use for fuzzy matching — they show up as
# prefixes/qualifiers on columns that belong to a *different* canonical
# field (e.g. "source" matching inside "Source IP Address" and stealing
# that column from source_ip). Still valid for exact-name matches (Pass 1),
# just excluded from the fuzzy fallback (Pass 2).
_FUZZY_EXCLUDED_ALIASES = {"source", "destination", "target", "level", "risk"}


def _fuzzy_score(norm_key: str, alias_key: str) -> float:
    """
    Similarity between an upload column name and a known alias.
    Whole-word containment (e.g. "device information" contains "device")
    scores a perfect match, but only when every alias word is distinctive
    enough (see _MIN_CONTAINMENT_WORD_LEN); otherwise fall back to
    edit-distance ratio (e.g. "hostnames" vs "hostname").
    """
    key_words = set(norm_key.split())
    alias_words = alias_key.split()
    if (
        alias_words
        and all(len(w) >= _MIN_CONTAINMENT_WORD_LEN for w in alias_words)
        and all(w in key_words for w in alias_words)
    ):
        return 1.0
    return difflib.SequenceMatcher(None, norm_key, alias_key).ratio()


def auto_map(upload_cols: list[str]) -> dict[str, Optional[str]]:
    """
    Return {canonical_col: upload_col_or_None} for all REQUIRED_COLS.
    None means no match was found and the caller must ask the analyst.
    """
    normalised = {_normalise_key(c): c for c in upload_cols}
    mapping: dict[str, Optional[str]] = {}
    used_cols: set[str] = set()

    # Pass 1: exact alias lookup (highest priority, unchanged behaviour).
    for canon in REQUIRED_COLS:
        matched = None
        for norm_key, orig_col in normalised.items():
            if orig_col in used_cols:
                continue
            if _ALIASES_NORM.get(norm_key) == canon:
                matched = orig_col
                break
        mapping[canon] = matched
        if matched:
            used_cols.add(matched)

    # Pass 2: conservative fuzzy fallback for whatever exact lookup missed.
    for canon in REQUIRED_COLS:
        if mapping[canon] is not None:
            continue
        candidate_keys = [
            k for k in _ALIAS_KEYS_BY_CANON.get(canon, [])
            if k not in _FUZZY_EXCLUDED_ALIASES
        ]
        best_col, best_score = None, 0.0
        for norm_key, orig_col in normalised.items():
            if orig_col in used_cols:
                continue
            score = max(
                (_fuzzy_score(norm_key, alias_key) for alias_key in candidate_keys),
                default=0.0,
            )
            if score > best_score:
                best_score, best_col = score, orig_col
        if best_score >= FUZZY_THRESHOLD:
            mapping[canon] = best_col
            used_cols.add(best_col)

    return mapping


def apply_mapping(
    df: pd.DataFrame,
    mapping: dict[str, Optional[str]],
) -> pd.DataFrame:
    """
    Rename and select columns according to the confirmed mapping.
    Fills defaults for every canonical column that has no source column.
    Normalises severity values to Low / Medium / High / Critical.
    """
    out = pd.DataFrame()

    for canon, src in mapping.items():
        if src and src in df.columns:
            out[canon] = df[src].astype(str).str.strip()
        elif canon in OPTIONAL_WITH_DEFAULT:
            out[canon] = OPTIONAL_WITH_DEFAULT[canon](df)
        else:
            # canon has no source column and no fallback default — raise early
            raise ValueError(
                f"Column '{canon}' is required but could not be mapped and "
                f"has no default. Please ensure your CSV contains a column "
                f"that can be mapped to '{canon}'."
            )

    # Normalise severity
    out["severity"] = out["severity"].map(
        lambda v: _SEV_NORM.get(_normalise_key(str(v)), "Medium")
    )

    return out.reset_index(drop=True)


def needs_manual_mapping(mapping: dict[str, Optional[str]]) -> list[str]:
    """Return list of canonical columns that still need analyst input."""
    return [
        c for c, src in mapping.items()
        if src is None and c not in OPTIONAL_WITH_DEFAULT
    ]
