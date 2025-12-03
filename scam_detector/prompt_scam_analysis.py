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
    
    # Ultra-minimal summary - just category names
    if "scam_categories" in framework:
        categories = list(framework["scam_categories"].keys())
        return f"Categories: {', '.join(categories)}"
    
    return ""

@lru_cache(maxsize=1)
def _get_framework_summary():
    """Get a minimal condensed summary of the framework for the prompt (cached)"""
    return _compute_framework_summary()

# Pre-compute and cache the framework summary at module load time for instant access
_FRAMEWORK_SUMMARY = _compute_framework_summary()

SCAM_ANALYSIS_PROMPT_TEMPLATE = """Analyze this job complaint for scam indicators. Categories: {framework_summary}

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