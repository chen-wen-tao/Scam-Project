"""
Job Scam Detection System
A modular system for detecting job scams using Google Gemini AI
"""

from .detector import JobScamDetector
from .config import DEFAULT_OUTPUT_DIR, SCAM_INDICATORS

__version__ = "1.0.0"
__all__ = ["JobScamDetector", "DEFAULT_OUTPUT_DIR", "SCAM_INDICATORS"]

