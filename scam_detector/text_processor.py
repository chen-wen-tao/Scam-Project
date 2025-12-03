"""
Text preprocessing and scam indicator extraction
"""

import re
import pandas as pd
from typing import Dict, List
from .config import SCAM_INDICATORS, MAX_TEXT_LENGTH


class TextProcessor:
    """Handles text preprocessing and indicator extraction"""
    
    def __init__(self, scam_indicators: Dict[str, List[str]] = None):
        """
        Initialize text processor
        
        Args:
            scam_indicators: Dictionary of scam indicators by category
        """
        self.scam_indicators = scam_indicators or SCAM_INDICATORS
    
    def preprocess_text(self, text: str, max_length: int = None) -> str:
        """
        Clean and preprocess text for analysis
        
        Args:
            text: Raw text to preprocess
            max_length: Maximum text length (truncate if longer to speed up processing)
                       If None, uses MAX_TEXT_LENGTH from config
            
        Returns:
            Cleaned text
        """
        if not text or pd.isna(text):
            return ""
        
        if max_length is None:
            max_length = MAX_TEXT_LENGTH
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', str(text).strip())
        
        # Truncate if too long (keep first part which usually has most important info)
        if len(text) > max_length:
            text = text[:max_length] + "... [truncated]"
        
        # Remove personal information patterns (basic anonymization)
        text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '[DATE]', text)  # Dates
        text = re.sub(r'\$\d+\.?\d*', '[AMOUNT]', text)  # Dollar amounts
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)  # SSN pattern
        
        return text
    
    def extract_scam_indicators(self, text: str) -> Dict[str, List[str]]:
        """
        Extract potential scam indicators from text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of found indicators by category
        """
        text_lower = text.lower()
        found_indicators = {}
        
        for category, indicators in self.scam_indicators.items():
            found = []
            for indicator in indicators:
                if indicator in text_lower:
                    found.append(indicator)
            if found:
                found_indicators[category] = found
        
        return found_indicators

