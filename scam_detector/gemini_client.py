"""
Google Gemini API client wrapper
"""

import re
import json
import logging
import google.generativeai as genai
from typing import Dict, Any, List

from .prompt_scam_analysis import get_scam_analysis_prompt

logger = logging.getLogger(__name__)


class GeminiClient:
    """Wrapper for Google Gemini API interactions"""
    
    def __init__(self, api_key: str, model_name: str = None):
        """
        Initialize Gemini client
        
        Args:
            api_key: Google Gemini API key
            model_name: Optional specific model name to use
        """
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        genai.configure(api_key=api_key)
        self.model = self._initialize_model(model_name)
    
    def _initialize_model(self, model_name: str = None):
        """
        Initialize and test Gemini model
        
        Args:
            model_name: Optional specific model name
            
        Returns:
            Initialized GenerativeModel
        """
        from .config import GEMINI_MODELS
        
        # Try user-specified model first
        if model_name:
            try:
                model = genai.GenerativeModel(model_name)
                test_response = model.generate_content("Test")
                logger.info(f"Successfully initialized model: {model_name}")
                return model
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
        
        # Try to find available models
        try:
            available_models = genai.list_models()
            logger.info("Checking available models...")
            
            for model in available_models:
                if 'generateContent' in model.supported_generation_methods:
                    candidate_name = model.name.replace('models/', '')
                    try:
                        test_model = genai.GenerativeModel(candidate_name)
                        test_response = test_model.generate_content("Test")
                        logger.info(f"Successfully initialized model: {candidate_name}")
                        return test_model
                    except Exception as e:
                        logger.warning(f"Model {candidate_name} failed test: {e}")
                        continue
        except Exception as e:
            logger.warning(f"Error checking available models: {e}")
        
        # Fallback to common model names
        for fallback_name in GEMINI_MODELS:
            try:
                model = genai.GenerativeModel(fallback_name)
                test_response = model.generate_content("Test")
                logger.info(f"Successfully initialized fallback model: {fallback_name}")
                return model
            except Exception as e:
                logger.warning(f"Fallback model {fallback_name} not available: {e}")
                continue
        
        raise ValueError("No compatible Gemini model found. Please check your API key and model availability.")
    
    def analyze_text(self, text: str, complaint_id: str = None) -> Dict[str, Any]:
        """
        Use Gemini AI to analyze text for scam indicators
        
        Args:
            text: Text to analyze
            complaint_id: Optional complaint ID for tracking
            
        Returns:
            Analysis results from Gemini
        """
        # Get prompt from prompts module
        prompt = get_scam_analysis_prompt(text)
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback parsing
                return self._parse_fallback_response(result_text)
                
        except Exception as e:
            logger.error(f"Error analyzing with Gemini: {e}")
            return {
                "scam_probability": 0,
                "red_flags": [],
                "financial_risk": "Unknown",
                "scam_type": "Unknown",
                "victim_profile": "Unknown",
                "recommendations": [],
                "confidence": 0,
                "error": str(e)
            }
    
    def _parse_fallback_response(self, text: str) -> Dict[str, Any]:
        """Parse response when JSON extraction fails"""
        return {
            "scam_probability": self._extract_score(text, "scam_probability"),
            "red_flags": self._extract_list(text, "red_flags"),
            "financial_risk": self._extract_text(text, "financial_risk"),
            "scam_type": self._extract_text(text, "scam_type"),
            "victim_profile": self._extract_text(text, "victim_profile"),
            "recommendations": self._extract_list(text, "recommendations"),
            "confidence": self._extract_score(text, "confidence")
        }
    
    def _extract_score(self, text: str, field: str) -> int:
        """Extract numeric score from text"""
        pattern = rf'{field}["\s]*:?\s*(\d+)'
        match = re.search(pattern, text, re.IGNORECASE)
        return int(match.group(1)) if match else 0
    
    def _extract_text(self, text: str, field: str) -> str:
        """Extract text field from response"""
        pattern = rf'{field}["\s]*:?\s*["\']?([^"\',\n]+)["\']?'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else "Unknown"
    
    def _extract_list(self, text: str, field: str) -> List[str]:
        """Extract list field from response"""
        pattern = rf'{field}["\s]*:?\s*\[(.*?)\]'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            items = re.findall(r'"([^"]+)"', match.group(1))
            return items
        return []

