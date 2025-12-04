"""
Prompt templates for Gemini AI analysis of employment/job scams
Based on academic research (Ravenelle et al., 2022) and FTC guidelines
"""

import json
from pathlib import Path
from functools import lru_cache

# Load classification framework once at module level
_FRAMEWORK_PATH = Path(__file__).parent / "scam_classification_framework.json"
_FRAMEWORK = None
_FRAMEWORK_SUMMARY = None  # Cache the summary string

def _load_framework():
    """Load the classification framework from JSON file (cached)"""
    global _FRAMEWORK
    if _FRAMEWORK is None:
        try:
            with open(_FRAMEWORK_PATH, 'r') as f:
                _FRAMEWORK = json.load(f)
        except FileNotFoundError:
            # Fallback to empty framework if file not found
            _FRAMEWORK = {}
    return _FRAMEWORK

def _compute_framework_summary():
    """Compute the framework summary (internal function)"""
    framework = _load_framework()
    if not framework:
        return "Use standard job scam classification (Ravenelle et al., 2022)."
    
    # Build summary with red flag categories and their items
    summary_parts = []
    
    if "scam_categories" in framework:
        categories = list(framework["scam_categories"].keys())
        summary_parts.append(f"Scam Categories: {', '.join(categories)}")
    
    if "red_flag_categories" in framework:
        red_flags = framework["red_flag_categories"]
        summary_parts.append("Red Flag Categories:")
        for category, items in red_flags.items():
            if isinstance(items, dict):
                # Dictionary format: show category and item count
                item_list = list(items.values())[:3]  # Show first 3 as examples
                examples = "; ".join(item_list)
                summary_parts.append(f"  - {category}: {examples}... ({len(items)} items)")
            elif isinstance(items, list):
                # Legacy array format
                examples = "; ".join(items[:3])
                summary_parts.append(f"  - {category}: {examples}... ({len(items)} items)")
    
    if "vulnerability_factors" in framework:
        vuln_factors = framework["vulnerability_factors"]
        if isinstance(vuln_factors, dict):
            examples = "; ".join(list(vuln_factors.values())[:3])
            summary_parts.append(f"Vulnerability Factors: {examples}... ({len(vuln_factors)} items)")
        elif isinstance(vuln_factors, list):
            examples = "; ".join(vuln_factors[:3])
            summary_parts.append(f"Vulnerability Factors: {examples}... ({len(vuln_factors)} items)")
    
    return "\n".join(summary_parts) if summary_parts else ""

@lru_cache(maxsize=1)
def _get_framework_summary():
    """Get a minimal condensed summary of the framework for the prompt (cached)"""
    return _compute_framework_summary()

# Pre-compute and cache the framework summary at module load time for instant access
_FRAMEWORK_SUMMARY = _compute_framework_summary()

SCAM_ANALYSIS_PROMPT_TEMPLATE = """Analyze this job complaint for scam indicators.

Framework Reference:
{framework_summary}

IMPORTANT: Use the exact red_flag categories from the framework:
- "communication": Communication-related red flags (unsolicited contact, poor grammar, pressure tactics, etc.)
- "financial": Financial red flags (payment requests, bank info requests, fake checks, etc.)
- "job_posting": Job posting red flags (unrealistic pay, vague descriptions, suspicious postings, etc.)
- "hiring_process": Hiring process red flags (no interview, immediate hiring, document requests, etc.)
- "work_activity": Work activity red flags (money mule tasks, package reshipping, payment processing, etc.)

For each category, identify specific red flags found in the complaint. Match them to framework items when possible, but describe them naturally if they don't exactly match.

For vulnerability_factors, identify factors that made the victim susceptible (employment desperation, limited experience, seeking remote work, etc.)

Return JSON:
{{
    "scam_probability": <0-100>,
    "scam_type": {{"primary_category": "<name>", "subcategory": "<type>"}},
    "red_flags": {{"communication": [], "financial": [], "job_posting": [], "hiring_process": [], "work_activity": []}},
    "financial_risk": {{"level": "<Low/Medium/High/Critical>", "potential_loss": "<amount>", "explanation": "<brief>"}},
    "victim_profile": {{"vulnerability_factors": [], "risk_level": "<Low/Medium/High>"}},
    "evidence_strength": "<Strong/Moderate/Weak>",
    "recommendations": {{"immediate_actions": [], "verification_steps": [], "reporting": []}},
    "confidence": <0-100>
}}

Complaint:
{text}"""


def get_scam_analysis_prompt(text: str) -> str:
    """
    Generate the prompt for employment scam analysis
    
    Args:
        text: The complaint text to analyze
        
    Returns:
        Formatted prompt string with comprehensive job scam classification framework
    """
    # Use pre-computed cached summary for instant access
    framework_summary = _FRAMEWORK_SUMMARY if _FRAMEWORK_SUMMARY is not None else _get_framework_summary()
    return SCAM_ANALYSIS_PROMPT_TEMPLATE.format(
        text=text,
        framework_summary=framework_summary
    )


# Optional: Validation prompt for quality control
VALIDATION_PROMPT_TEMPLATE = """
Review the following scam analysis for quality and accuracy:

Original Complaint:
{text}

Analysis Results:
{analysis_json}

Validate:
1. Are red flags correctly identified and categorized?
2. Is the scam type classification accurate and specific?
3. Are recommendations actionable and appropriate?
4. Is the financial risk assessment realistic?
5. Are there any false positives (legitimate practices marked as scams)?
6. Are there any missed red flags?

Provide validation results in JSON format:
{{
    "validation_passed": <true/false>,
    "accuracy_score": <0-100>,
    "issues_found": ["<issue1>", "<issue2>"],
    "suggested_corrections": {{
        "<field>": "<corrected value>"
    }},
    "missed_red_flags": ["<flag1>", "<flag2>"],
    "false_positives": ["<item1>", "<item2>"]
}}
"""


def get_validation_prompt(text: str, analysis_json: str) -> str:
    """
    Generate validation prompt for quality control
    
    Args:
        text: Original complaint text
        analysis_json: JSON string of the analysis results
        
    Returns:
        Formatted validation prompt
    """
    return VALIDATION_PROMPT_TEMPLATE.format(
        text=text,
        analysis_json=analysis_json
    )