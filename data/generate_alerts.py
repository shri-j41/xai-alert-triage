"""
Generates 200 synthetic Wazuh-style security alerts representative of a
Nepali SME environment. Run once to produce alerts_raw.csv.
"""

import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Domain constants ──────────────────────────────────────────────────────────

ALERT_CATEGORIES = [
    "Authentication Failure",
    "Privilege Escalation",
    "Malware Detected",
    "Brute Force Attack",
    "Suspicious Process Execution",
    "Data Exfiltration Attempt",
    "Firewall Rule Violation",
    "Unauthorised File Access",
    "Lateral Movement",
    "Reconnaissance / Port Scan",
    "Ransomware Indicator",
    "Policy Violation",
]

SEVERITY_MAP = {
    "Authentication Failure":       ("Low",      "Medium"),
    "Privilege Escalation":         ("High",     "Critical"),
    "Malware Detected":             ("High",     "Critical"),
    "Brute Force Attack":           ("Medium",   "High"),
    "Suspicious Process Execution": ("Medium",   "High"),
    "Data Exfiltration Attempt":    ("High",     "Critical"),
    "Firewall Rule Violation":      ("Low",      "Medium"),
    "Unauthorised File Access":     ("Medium",   "High"),
    "Lateral Movement":             ("High",     "Critical"),
    "Reconnaissance / Port Scan":   ("Low",      "Medium"),
    "Ransomware Indicator":         ("Critical", "Critical"),
    "Policy Violation":             ("Low",      "Medium"),
}

SOURCE_TOOLS = [
    "Wazuh-HIDS", "Wazuh-NIDS", "Wazuh-FIM",
    "Wazuh-SCA",  "Suricata",   "Sysmon",
]

# Assets typical of a Nepali SME (finance / retail / healthcare sector mix)
ASSETS = [
    "fin-server-01",   # core banking / accounting server
    "erp-server-01",   # ERP / inventory system
    "dc-server-01",    # domain controller
    "web-server-01",   # public-facing website
    "db-server-01",    # customer database
    "backup-server-01",
    "workstation-acct-01",
    "workstation-acct-02",
    "workstation-hr-01",
    "workstation-mgmt-01",
    "pos-terminal-01", # point-of-sale
    "pos-terminal-02",
    "laptop-sales-01",
    "laptop-remote-01",
    "printer-office-01",
]

USERS = [
    "admin",
    "sysadmin",
    "fin_manager",
    "accountant1",
    "accountant2",
    "hr_officer",
    "sales_exec1",
    "sales_exec2",
    "it_support",
    "ceo",
    "guest_user",
    "service_account",
    "SYSTEM",
]

DESCRIPTION_TEMPLATES = {
    "Authentication Failure":
        "Multiple failed login attempts detected for user {user} on {asset} from {ip}.",
    "Privilege Escalation":
        "User {user} attempted to escalate privileges on {asset} using sudo/runas.",
    "Malware Detected":
        "Wazuh detected malware signature on {asset}. File flagged by ClamAV/YARA rule.",
    "Brute Force Attack":
        "Brute-force SSH/RDP attack detected against {asset} from external IP {ip}.",
    "Suspicious Process Execution":
        "Suspicious process spawned by {user} on {asset}: cmd.exe -> powershell -enc ...",
    "Data Exfiltration Attempt":
        "Large outbound transfer (>500 MB) detected from {asset} by user {user} to {ip}.",
    "Firewall Rule Violation":
        "Traffic from {ip} to {asset} violated firewall policy on port 3389/22.",
    "Unauthorised File Access":
        "User {user} accessed restricted directory /finance/payroll on {asset}.",
    "Lateral Movement":
        "SMB lateral movement detected: {user} connecting from {ip} to internal hosts.",
    "Reconnaissance / Port Scan":
        "Port scan from {ip} targeting internal subnet; {asset} probed on 1024+ ports.",
    "Ransomware Indicator":
        "Mass file-rename with extension .enc detected on {asset}. Possible ransomware.",
    "Policy Violation":
        "USB mass-storage device inserted by {user} on {asset} outside approved hours.",
}

INTERNAL_IPS = [f"192.168.1.{i}" for i in range(10, 50)]
EXTERNAL_IPS = [
    "103.45.67.89", "185.220.101.45", "45.33.32.156",
    "198.51.100.23", "203.0.113.77",  "91.108.4.0",
    "77.88.55.66",  "162.158.0.1",    "104.21.0.0",
    "172.67.0.0",
]


def random_ip(external_prob: float = 0.4) -> str:
    if random.random() < external_prob:
        return random.choice(EXTERNAL_IPS)
    return random.choice(INTERNAL_IPS)


def random_timestamp(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def build_alert(alert_id: int) -> dict:
    category = random.choice(ALERT_CATEGORIES)
    sev_choices = SEVERITY_MAP[category]
    severity = random.choice(sev_choices) if isinstance(sev_choices, tuple) else sev_choices

    asset = random.choice(ASSETS)
    user  = random.choice(USERS)
    ip    = random_ip()

    ts = random_timestamp(
        datetime(2024, 1, 1, 0, 0, 0),
        datetime(2024, 6, 30, 23, 59, 59),
    )

    description = DESCRIPTION_TEMPLATES[category].format(
        user=user, asset=asset, ip=ip
    )

    return {
        "alert_id":          f"ALT-{alert_id:04d}",
        "timestamp":         ts.strftime("%Y-%m-%d %H:%M:%S"),
        "source_tool":       random.choice(SOURCE_TOOLS),
        "alert_category":    category,
        "severity":          severity,
        "affected_asset":    asset,
        "user":              user,
        "source_ip":         ip,
        "alert_description": description,
    }


def generate(n: int = 200, out_path: str = "data/alerts_raw.csv") -> pd.DataFrame:
    alerts = [build_alert(i + 1) for i in range(n)]
    df = pd.DataFrame(alerts)
    df.to_csv(out_path, index=False)
    print(f"[generate_alerts] Wrote {len(df)} alerts -> {out_path}")
    return df


if __name__ == "__main__":
    generate()
