"""
Configuration and constants for the Job Scam Detection System
"""

import os
from pathlib import Path

# Default output directory for detection results
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "detect_res"

# Scam indicators based on analysis of the dataset
SCAM_INDICATORS = {
    'financial_red_flags': [
        'fake check', 'bounced check', 'check bounced', 'check returned',
        'mobile deposit', 'deposit check', 'send money', 'transfer money',
        'zelle', 'venmo', 'paypal', 'wire transfer', 'money order',
        'equipment purchase', 'startup materials', 'office supplies'
    ],
    'urgency_pressure': [
        'immediately', 'urgent', 'asap', 'right away', 'hurry',
        'deadline', 'time sensitive', 'quickly', 'fast'
    ],
    'communication_red_flags': [
        'email only', 'no phone interview', 'text message', 'whatsapp',
        'telegram', 'no video call', 'remote interview only'
    ],
    'job_red_flags': [
        'no experience required', 'work from home', 'remote position',
        'easy money', 'quick cash', 'part time', 'flexible hours',
        'personal assistant', 'virtual assistant', 'data entry'
    ],
    'company_red_flags': [
        'no website', 'new company', 'startup', 'no office',
        'work from anywhere', 'international company'
    ]
}

# Gemini model fallback list (prioritize faster models)
# gemini-1.5-flash is faster and cheaper, good for batch processing
# gemini-1.5-pro is more accurate but slower
GEMINI_MODELS = ['gemini-1.5-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.0-pro']

# Default file names
DEFAULT_RESULTS_CSV = "scam_analysis_results.csv"
DEFAULT_REPORT_JSON = "scam_analysis_report.json"

# Performance optimization settings
MAX_TEXT_LENGTH = 3000  # Truncate complaints longer than this (chars) to speed up processing
DEFAULT_WORKERS = 2  # Default number of parallel workers for batch processing (set to 2 to avoid rate limits)
