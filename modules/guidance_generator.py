"""
Triage guidance generator.

Rule-based "recommended next action" text per alert, keyed off
alert_category (same style as CATEGORY_RISK in modules/model.py). This is
the fifth of the six proposal layers ("Triage Guidance") — it does not
touch scoring or SHAP, it only produces analyst-facing action text.
"""

CATEGORY_GUIDANCE: dict[str, str] = {
    "Ransomware Indicator":
        "Isolate the host from the network immediately, check backup integrity, and notify the asset owner.",
    "Data Exfiltration Attempt":
        "Block outbound traffic from the host, identify what data left, and notify the data owner.",
    "Privilege Escalation":
        "Suspend the account, review recent privilege changes, and check for lateral movement.",
    "Lateral Movement":
        "Isolate the affected hosts, review authentication logs across the segment, and notify IT security.",
    "Malware Detected":
        "Quarantine the file or host, run a full AV/EDR scan, and confirm no further spread.",
    "Brute Force Attack":
        "Block the source IP, review the account lockout policy, and confirm no successful logins.",
    "Suspicious Process Execution":
        "Investigate the parent process and command line, and isolate the host if unconfirmed.",
    "Unauthorised File Access":
        "Verify access was authorised with the data owner and review file permissions.",
    "Firewall Rule Violation":
        "Review the firewall rule and traffic source; tighten the rule if unintended.",
    "Reconnaissance / Port Scan":
        "Log the source and monitor for follow-up activity from the same origin.",
    "Authentication Failure":
        "Verify with the user directly and check for further failed attempts.",
    "Policy Violation":
        "Review the activity against policy and follow up with the user if needed.",
}

DEFAULT_GUIDANCE = "Review the alert details and confirm whether follow-up action is required."


def guidance_for(row) -> str:
    """Recommended next action for one alert row (needs alert_category)."""
    return CATEGORY_GUIDANCE.get(row.get("alert_category"), DEFAULT_GUIDANCE)
