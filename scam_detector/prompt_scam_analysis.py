"""
Prompt templates for Gemini AI analysis
"""

SCAM_ANALYSIS_PROMPT_TEMPLATE = """
Analyze the following job-related complaint for potential scam indicators. 
Provide a comprehensive analysis focusing on:

1. SCAM PROBABILITY (0-100): Rate how likely this is a scam
2. RED FLAGS: List specific red flags found
3. FINANCIAL RISK: Assess the financial risk level (Low/Medium/High)
4. SCAM TYPE: Identify the type of scam (e.g., fake check, advance fee, etc.)
5. VICTIM PROFILE: Describe the victim's vulnerability factors
6. RECOMMENDATIONS: Suggest prevention measures

Complaint Text:
{text}

Please respond in JSON format with the following structure:
{{
    "scam_probability": 0-100,
    "red_flags": ["flag1", "flag2"],
    "financial_risk": "Low/Medium/High",
    "scam_type": "description",
    "victim_profile": "description",
    "recommendations": ["rec1", "rec2"],
    "confidence": 0-100
}}
"""


def get_scam_analysis_prompt(text: str) -> str:
    """
    Generate the prompt for scam analysis
    
    Args:
        text: The complaint text to analyze
        
    Returns:
        Formatted prompt string
    """
    return SCAM_ANALYSIS_PROMPT_TEMPLATE.format(text=text)

