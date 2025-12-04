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

# Gemini model fallback list (prioritize models with better rate limits: https://ai.google.dev/gemini-api/docs/rate-limits)
# Priority order based on RPD (Requests Per Day) and RPM (Requests Per Minute):
# 1. gemini-2.5-flash-lite: 1,000 RPD, 15 RPM (best for batch processing - 4x more requests than 2.5-flash)
# 2. gemini-2.0-flash-lite: 200 RPD, 30 RPM (fastest, but lower daily limit)
# 3. gemini-2.0-flash: 200 RPD, 15 RPM, 1M TPM (high token limit)
# 4. gemini-2.5-flash: 250 RPD, 10 RPM (current default, but hits quota easily)
# 5. gemini-2.5-pro: 50 RPD, 2 RPM (best quality but slowest and lowest quota)
GEMINI_MODELS = [
    'gemini-2.5-flash-lite',      # Best balance: 1,000 RPD, 15 RPM
    'gemini-2.0-flash-lite',       # Fastest: 30 RPM, 200 RPD
    'gemini-2.0-flash',            # High TPM: 1M tokens, 200 RPD
    'gemini-2.5-flash',            # Current default (250 RPD limit)
    'gemini-1.5-flash',            # Fallback (deprecated but still works)
    'gemini-2.5-pro',              # Best quality (slow, low quota)
    'gemini-1.5-pro'               # Legacy fallback
]

# Default file names
DEFAULT_RESULTS_CSV = "scam_analysis_results.csv"
DEFAULT_REPORT_JSON = "scam_analysis_report.json"

# Performance optimization settings
MAX_TEXT_LENGTH = 3000  # Truncate complaints longer than this (chars) to speed up processing
DEFAULT_WORKERS = 1  # Default number of parallel workers (1 for free tier to avoid rate limits, increase if you have paid tier)

# Rate limiting settings for gemini-2.5-flash-lite (15 RPM free tier)
# Very conservative: 12 requests per minute to leave buffer for concurrent workers
# With 1 worker: ~5 seconds between requests
# With 2 workers: ~10 seconds between requests (each worker waits)
RATE_LIMIT_RPM = 12  # Requests per minute (very conservative for free tier)
RATE_LIMIT_WINDOW = 60  # Time window in seconds
MIN_REQUEST_INTERVAL = RATE_LIMIT_WINDOW / RATE_LIMIT_RPM  # ~5 seconds between requests
