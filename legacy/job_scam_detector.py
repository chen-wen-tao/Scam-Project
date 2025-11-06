#!/usr/bin/env python3
"""
Job Scam Detection System using Google Gemini
[LEGACY WRAPPER - For backward compatibility]

This file is maintained for backward compatibility.
New code should import from 'scam_detector' package instead:
    from scam_detector import JobScamDetector
"""

import warnings

# Import from the new modular package
from scam_detector import JobScamDetector as _JobScamDetector

# Show deprecation warning
warnings.warn(
    "Importing from 'job_scam_detector' is deprecated. "
    "Please use 'from scam_detector import JobScamDetector' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backward compatibility
JobScamDetector = _JobScamDetector

__all__ = ['JobScamDetector']
