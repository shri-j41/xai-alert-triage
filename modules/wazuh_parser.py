"""
Wazuh JSON log parser.

Accepts a Wazuh alert export in any of three common formats:
  (a) NDJSON  — one JSON object per line  (the most common Wazuh filebeat output)
  (b) JSON array  — [ {...}, {...}, ... ]  (Kibana / Discover export)
  (c) Wrapped array — {"hits": {"hits": [{"_source": {...}}, ...]}}
                      (OpenSearch / Elasticsearch export)

Extracts:
  timestamp        from alert.timestamp  (ISO-8601)
  rule description from alert.rule.description
  rule level       from alert.rule.level  (int 1-15)
  agent name       from alert.agent.name  (→ affected_asset)
  source IP        from alert.data.srcip  or alert.data.src_ip
  user             from alert.data.dstuser / alert.data.win.eventdata.targetUserName
                        / alert.syscheck.uname_after

Severity mapping (Wazuh convention)
  1–6   → Low
  7–11  → Medium
  12–15 → Critical    (mapped to "High" in our 3-tier model for XGBoost)

Category mapping
  Rule description keywords are matched against our 12 standard categories.
  Unknown descriptions fall back to "Policy Violation".

Output: pandas DataFrame in our canonical 9-column schema, ready for
context_enricher.enrich() and then model.predict().
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Union

import pandas as pd

# ── Severity mapping ──────────────────────────────────────────────────────────

def level_to_severity(level: int) -> str:
    if level <= 6:
        return "Low"
    if level <= 11:
        return "Medium"
    return "Critical"          # 12-15 → shown as Critical in table; XGBoost maps to 4


# ── Category keyword matcher ──────────────────────────────────────────────────
# Each entry: (compiled_regex, canonical_category)
# Patterns are tried in order; first match wins.

_CATEGORY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"ransomware|file.encr|mass.renam|locky|wannacry|cryptolocker",   re.I), "Ransomware Indicator"),
    (re.compile(r"exfil|data.transfer|large.upload|outbound.transfer",             re.I), "Data Exfiltration Attempt"),
    (re.compile(r"privilege.escal|sudo|runas|elevation|priv.escal|token.imperson", re.I), "Privilege Escalation"),
    (re.compile(r"lateral.mov|pass.the.hash|pass.the.ticket|psexec|wmiexec",       re.I), "Lateral Movement"),
    (re.compile(r"malware|trojan|virus|rootkit|spyware|backdoor|worm|clamav|yara", re.I), "Malware Detected"),
    (re.compile(r"brute.force|brute-force|password.spray|credential.stuff",        re.I), "Brute Force Attack"),
    (re.compile(r"suspicious.proc|suspicious.exec|unusual.proc|cmd.*powershell|"
                r"encoded.command|obfuscat",                                        re.I), "Suspicious Process Execution"),
    (re.compile(r"unauthori[sz]ed.file|file.access|sensitive.file|syscheck",       re.I), "Unauthorised File Access"),
    (re.compile(r"firewall|iptables|blocked.connect|packet.filter|acl.deny",       re.I), "Firewall Rule Violation"),
    (re.compile(r"auth.fail|login.fail|logon.fail|invalid.user|bad.password|"
                r"failed.pass|multiple.auth",                                       re.I), "Authentication Failure"),
    (re.compile(r"port.scan|recon|nmap|masscan|network.scan|host.discov",          re.I), "Reconnaissance / Port Scan"),
]

_DEFAULT_CATEGORY = "Policy Violation"


def _map_category(description: str) -> str:
    for pattern, category in _CATEGORY_PATTERNS:
        if pattern.search(description):
            return category
    return _DEFAULT_CATEGORY


# ── Field extractors ──────────────────────────────────────────────────────────

def _get(d: dict, *keys, default=""):
    """Safe nested dict access: _get(d, 'rule', 'level', default=0)."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


def _extract_user(alert: dict) -> str:
    """Try several common Wazuh paths for the username."""
    candidates = [
        _get(alert, "data", "dstuser"),
        _get(alert, "data", "win", "eventdata", "targetUserName"),
        _get(alert, "data", "win", "eventdata", "subjectUserName"),
        _get(alert, "syscheck", "uname_after"),
        _get(alert, "data", "srcuser"),
        _get(alert, "rule", "firedtimes"),     # not a user — skip if int
    ]
    for c in candidates:
        if c and isinstance(c, str) and c.strip() not in ("", "-", "N/A", "0"):
            return c.strip()
    return "unknown"


def _extract_src_ip(alert: dict) -> str:
    candidates = [
        _get(alert, "data", "srcip"),
        _get(alert, "data", "src_ip"),
        _get(alert, "data", "win", "eventdata", "ipAddress"),
        _get(alert, "network", "srcip"),
    ]
    for c in candidates:
        if c and str(c).strip() not in ("", "-", "N/A", "::1", "127.0.0.1"):
            return str(c).strip()
    return "0.0.0.0"


def _parse_one(alert: dict, idx: int) -> dict:
    """Convert a single Wazuh alert dict to our canonical row dict."""
    rule_level = int(_get(alert, "rule", "level", default=3))
    description = str(_get(alert, "rule", "description", default="No description"))
    agent_name  = str(_get(alert, "agent", "name", default="unknown-host"))
    timestamp   = str(_get(alert, "timestamp", default=""))
    alert_id_raw= str(_get(alert, "id", default=""))

    # Normalise timestamp (Wazuh uses ISO-8601 with timezone)
    try:
        ts = pd.to_datetime(timestamp, utc=True).tz_convert(None)   # strip tz
        timestamp_clean = ts.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        timestamp_clean = timestamp[:19] if len(timestamp) >= 19 else timestamp

    alert_id = alert_id_raw if alert_id_raw else f"WZ-{idx+1:04d}"

    return {
        "alert_id":          alert_id,
        "timestamp":         timestamp_clean,
        "source_tool":       "Wazuh-HIDS",
        "alert_category":    _map_category(description),
        "severity":          level_to_severity(rule_level),
        "affected_asset":    agent_name,
        "user":              _extract_user(alert),
        "source_ip":         _extract_src_ip(alert),
        "alert_description": description,
    }


# ── Format sniffers ───────────────────────────────────────────────────────────

def _load_raw_alerts(content: str) -> list[dict]:
    """
    Try to decode the content as:
    1. JSON array  [ {...}, ... ]
    2. OpenSearch/Kibana wrapped export  {"hits": {"hits": [{"_source": {...}}]}}
    3. NDJSON (one JSON per line)
    """
    content = content.strip()

    # --- Attempt 1: full JSON parse (array or wrapped) ---
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed

        # Kibana / OpenSearch export
        hits = (
            parsed.get("hits", {}).get("hits", None)
            or parsed.get("responses", [{}])[0].get("hits", {}).get("hits", None)
        )
        if hits:
            return [h.get("_source", h) for h in hits]

        # Single alert wrapped in an object
        if "rule" in parsed:
            return [parsed]

    except json.JSONDecodeError:
        pass

    # --- Attempt 2: NDJSON ---
    alerts = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                alerts.append(obj)
        except json.JSONDecodeError:
            continue

    return alerts


# ── Public API ────────────────────────────────────────────────────────────────

def parse_file(path: Union[str, Path]) -> pd.DataFrame:
    """
    Parse a single Wazuh JSON export file.

    Parameters
    ----------
    path : str or Path
        Path to the .json file.

    Returns
    -------
    pd.DataFrame in our canonical 9-column schema.
    """
    path = Path(path)
    content = path.read_text(encoding="utf-8", errors="replace")
    return parse_string(content)


def parse_string(content: str) -> pd.DataFrame:
    """
    Parse Wazuh JSON content supplied as a string (e.g. from st.file_uploader).

    Parameters
    ----------
    content : str
        Raw JSON text (NDJSON, array, or wrapped export).

    Returns
    -------
    pd.DataFrame in our canonical 9-column schema.
    """
    raw_alerts = _load_raw_alerts(content)
    if not raw_alerts:
        raise ValueError(
            "No valid Wazuh alert objects found in the uploaded file. "
            "Expected NDJSON, a JSON array, or an OpenSearch/Kibana export."
        )

    rows = [_parse_one(a, i) for i, a in enumerate(raw_alerts)]
    df = pd.DataFrame(rows)

    # Deduplicate alert_ids in case the export contains duplicates
    counts: dict[str, int] = {}
    new_ids = []
    for aid in df["alert_id"]:
        counts[aid] = counts.get(aid, 0) + 1
        suffix = f"-{counts[aid]}" if counts[aid] > 1 else ""
        new_ids.append(f"{aid}{suffix}")
    df["alert_id"] = new_ids

    return df


def parse_folder(folder: Union[str, Path]) -> pd.DataFrame:
    """
    Parse all *.json files inside a folder and concatenate the results.

    Parameters
    ----------
    folder : str or Path

    Returns
    -------
    pd.DataFrame (all alerts from all files, deduplicated by alert_id).
    """
    folder = Path(folder)
    frames = []
    for fpath in sorted(folder.glob("*.json")):
        try:
            frames.append(parse_file(fpath))
        except Exception as e:
            print(f"[wazuh_parser] Skipping {fpath.name}: {e}")

    if not frames:
        raise ValueError(f"No valid .json files found in folder: {folder}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["alert_id"])
    return combined.reset_index(drop=True)


# ── CLI helper ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m modules.wazuh_parser <file_or_folder> [out.csv]")
        sys.exit(1)

    target = Path(sys.argv[1])
    out_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/wazuh_parsed.csv"

    if target.is_dir():
        df = parse_folder(target)
    else:
        df = parse_file(target)

    os.makedirs(Path(out_path).parent, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Parsed {len(df)} alerts -> {out_path}")
    print(df.head(5).to_string())
