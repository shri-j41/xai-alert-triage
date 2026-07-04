"""
Fallback alert-description generator.

When a mapped alert_description is missing, the tool's own default
placeholder, or unusable filler text (e.g. Lorem Ipsum from a generic
public CSV), this module builds a plain-English description from the
alert's own already-computed fields instead of showing nothing useful.

This module only produces display text — it does not touch the model,
SHAP explanations, or any dataframe columns used elsewhere in the pipeline.
"""

# The exact placeholder csv_mapper.py fills in when alert_description has
# no source column at all (see modules/csv_mapper.py OPTIONAL_WITH_DEFAULT).
_TOOL_DEFAULT_TEXT = "no description provided."

# A few common Lorem-Ipsum markers — enough to catch the standard filler
# text and its usual variants without false-positiving on real prose.
_FILLER_MARKERS = ("lorem ipsum", "dolor sit amet", "consectetur adipiscing")

# Below this length, text is too short to carry real information.
_MIN_USABLE_LENGTH = 8


def is_usable_description(text) -> bool:
    """True if `text` looks like real analyst-facing content."""
    if text is None:
        return False
    s = str(text).strip()
    if len(s) < _MIN_USABLE_LENGTH:
        return False
    lowered = s.lower()
    if lowered == _TOOL_DEFAULT_TEXT:
        return False
    if any(marker in lowered for marker in _FILLER_MARKERS):
        return False
    return True


def generate_fallback_description(row) -> str:
    """
    Build a plain-English description from an alert's own computed fields:
    severity, alert_category, business_function, asset_criticality,
    user_role, internet_exposed, outside_working_hours.
    """
    severity    = row.get("severity", "Unknown")
    category    = row.get("alert_category", "Unknown")
    function    = row.get("business_function", "Unknown")
    criticality = row.get("asset_criticality", "N/A")

    sentence_1 = (
        f"{severity} severity {category} alert on a {function} asset "
        f"(criticality {criticality}/5)."
    )

    user_role = row.get("user_role", "Standard")
    clauses = [f"{user_role} user"]
    if row.get("outside_working_hours", False):
        clauses.append("outside working hours")
    if row.get("internet_exposed", False):
        clauses.append("internet-exposed")
    sentence_2 = ", ".join(clauses) + "."

    return f"{sentence_1} {sentence_2}"


def get_display_description(row) -> tuple[str, bool]:
    """
    Returns (description_text, is_auto_generated).

    Uses the original alert_description if it's present and usable;
    otherwise falls back to a templated description built from the row's
    own fields, flagged so the UI can show it's auto-generated.
    """
    original = row.get("alert_description")
    if is_usable_description(original):
        return str(original).strip(), False
    return generate_fallback_description(row), True
