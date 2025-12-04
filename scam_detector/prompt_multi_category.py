"""
Prompt template for multi-category classification (4 categories)
Focuses on: scam_job, scam_other, not_scam_irrelevant, not_scam_job_relevant
Used for evaluation and general complaint analysis
"""

import json
from pathlib import Path
from functools import lru_cache

# Load classification framework once at module level (shared with job scam prompt)
_FRAMEWORK_PATH = Path(__file__).parent / "scam_classification_framework.json"
_FRAMEWORK = None
_FRAMEWORK_SUMMARY = None

def _load_framework():
    """Load the classification framework from JSON file (cached)"""
    global _FRAMEWORK
    if _FRAMEWORK is None:
        try:
            with open(_FRAMEWORK_PATH, 'r') as f:
                _FRAMEWORK = json.load(f)
        except FileNotFoundError:
            _FRAMEWORK = {}
    return _FRAMEWORK

def _compute_framework_summary():
    """Compute a minimal framework summary for classification task"""
    framework = _load_framework()
    if not framework:
        return "Use standard scam classification."
    
    summary_parts = []
    
    if "scam_categories" in framework:
        categories = list(framework["scam_categories"].keys())
        summary_parts.append(f"Job Scam Categories: {', '.join(categories)}")
    
    return "\n".join(summary_parts) if summary_parts else ""

# Pre-compute and cache the framework summary
_FRAMEWORK_SUMMARY = _compute_framework_summary()

MULTI_CATEGORY_PROMPT_TEMPLATE = """Analyze this consumer complaint and classify it into one of four categories.

CLASSIFICATION TASK:
1. Determine if this is a SCAM (fraud) or NOT a scam
2. If it's a scam, determine if it's JOB-RELATED or OTHER type of scam
3. If it's not a scam, determine if it's JOB-RELATED or IRRELEVANT

Categories:
- "scam_job": Job-related scams (employment fraud, fake job offers, work-from-home scams, hiring scams, etc.)
- "scam_other": Other types of scams (wire fraud, romance scams, business scams, service scams, etc.) - NOT job-related
- "not_scam_job_relevant": Not a scam, but job-related (legitimate job complaints, employment issues, etc.)
- "not_scam_irrelevant": Not a scam and not job-related (general complaints, unrelated issues)

Framework Reference (for job-related scams only):
{framework_summary}

IMPORTANT GUIDELINES:
- Job-related scams involve: employment, job offers, work opportunities, hiring processes, job applications, work-from-home offers
- Non-job scams include: wire fraud, romance scams, business fraud, service scams, identity theft (non-employment), etc.
- If the complaint mentions "job", "employment", "work", "hire" but it's NOT a scam, classify as "not_scam_job_relevant"
- If the complaint has no job/employment context and is not a scam, classify as "not_scam_irrelevant"

For red flags (only populate if job-related scam):
- Use framework categories: "communication", "financial", "job_posting", "hiring_process", "work_activity"
- For non-job scams, only use "communication" and "financial" if relevant, leave others empty

Return JSON:
{{
    "scam_probability": <0-100>,
    "is_job_related": <true/false>,
    "category": "<scam_job|scam_other|not_scam_job_relevant|not_scam_irrelevant>",
    "scam_type": {{"primary_category": "<name>", "subcategory": "<type>"}},
    "red_flags": {{"communication": [], "financial": [], "job_posting": [], "hiring_process": [], "work_activity": []}},
    "financial_risk": {{"level": "<Low/Medium/High/Critical|N/A>", "potential_loss": "<amount|N/A>", "explanation": "<brief>"}},
    "confidence": <0-100>
}}

Complaint:
{text}"""


def get_multi_category_prompt(text: str) -> str:
    """
    Generate the prompt for multi-category classification (4 categories)
    
    Args:
        text: The complaint text to analyze
        
    Returns:
        Formatted prompt string for 4-category classification
    """
    framework_summary = _FRAMEWORK_SUMMARY if _FRAMEWORK_SUMMARY is not None else _compute_framework_summary()
    return MULTI_CATEGORY_PROMPT_TEMPLATE.format(
        text=text,
        framework_summary=framework_summary
    )

