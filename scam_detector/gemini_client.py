"""
Google Gemini API client wrapper
"""

import re
import json
import logging
import time
import threading
import google.generativeai as genai
from typing import Dict, Any, List, Optional
from collections import deque

from .prompt_scam_analysis import get_scam_analysis_prompt
from .prompt_multi_category import get_multi_category_prompt
from .config import RATE_LIMIT_RPM, RATE_LIMIT_WINDOW, MIN_REQUEST_INTERVAL

logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe rate limiter to ensure we don't exceed API rate limits"""
    
    def __init__(self, max_requests: int, window_seconds: float):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum number of requests allowed
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_times: deque = deque()
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to stay within rate limits"""
        with self.lock:
            now = time.time()
            
            # Remove requests outside the time window
            while self.request_times and now - self.request_times[0] > self.window_seconds:
                self.request_times.popleft()
            
            # If we're at the limit, wait until the oldest request expires
            if len(self.request_times) >= self.max_requests:
                oldest_time = self.request_times[0]
                wait_time = self.window_seconds - (now - oldest_time) + 0.1  # Add small buffer
                if wait_time > 0:
                    logger.debug(f"Rate limit: waiting {wait_time:.2f}s to stay under {self.max_requests} requests/{self.window_seconds}s")
                    time.sleep(wait_time)
                    # Update now after waiting
                    now = time.time()
                    # Clean up again after waiting
                    while self.request_times and now - self.request_times[0] > self.window_seconds:
                        self.request_times.popleft()
            
            # Record this request
            self.request_times.append(time.time())


class GeminiClient:
    """Wrapper for Google Gemini API interactions"""
    
    def __init__(self, api_key: str, model_name: Optional[str] = None):
        """
        Initialize Gemini client
        
        Args:
            api_key: Google Gemini API key
            model_name: Optional specific model name to use
        """
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        genai.configure(api_key=api_key)  # type: ignore[attr-defined]
        self.model, self.model_name = self._initialize_model(model_name)
        
        # Initialize rate limiter (very conservative: 12 RPM for 15 RPM free tier limit)
        # Use rate limiting for models that need it
        if 'flash-lite' in self.model_name or '2.5-flash-lite' in self.model_name:
            self.rate_limiter = RateLimiter(RATE_LIMIT_RPM, RATE_LIMIT_WINDOW)
            logger.info(f"Rate limiter enabled: max {RATE_LIMIT_RPM} requests per {RATE_LIMIT_WINDOW} seconds (free tier limit: 15 RPM)")
        else:
            self.rate_limiter = None
    
    def _initialize_model(self, model_name: Optional[str] = None):
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
                model = genai.GenerativeModel(model_name)  # type: ignore[attr-defined]
                test_response = model.generate_content("Test")
                logger.info(f"Successfully initialized model: {model_name}")
                return model, model_name
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
        
        # First, try prioritized models from config (better rate limits)
        logger.info("Checking prioritized models from config...")
        for prioritized_name in GEMINI_MODELS:
            try:
                model = genai.GenerativeModel(prioritized_name)  # type: ignore[attr-defined]
                test_response = model.generate_content("Test")
                logger.info(f"Successfully initialized prioritized model: {prioritized_name}")
                return model, prioritized_name
            except Exception as e:
                logger.debug(f"Prioritized model {prioritized_name} not available: {e}")
                continue
        
        # If prioritized models fail, try to find any available models
        try:
            available_models = genai.list_models()  # type: ignore[attr-defined]
            logger.info("Checking all available models...")
            
            for model in available_models:
                if 'generateContent' in model.supported_generation_methods:
                    candidate_name = model.name.replace('models/', '')
                    try:
                        test_model = genai.GenerativeModel(candidate_name)  # type: ignore[attr-defined]
                        test_response = test_model.generate_content("Test")
                        logger.info(f"Successfully initialized model: {candidate_name}")
                        return test_model, candidate_name
                    except Exception as e:
                        logger.warning(f"Model {candidate_name} failed test: {e}")
                        continue
        except Exception as e:
            logger.warning(f"Error checking available models: {e}")
        
        # Final fallback to common model names
        for fallback_name in GEMINI_MODELS:
            try:
                model = genai.GenerativeModel(fallback_name)  # type: ignore[attr-defined]
                test_response = model.generate_content("Test")
                logger.info(f"Successfully initialized fallback model: {fallback_name}")
                return model, fallback_name
            except Exception as e:
                logger.warning(f"Fallback model {fallback_name} not available: {e}")
                continue
        
        raise ValueError("No compatible Gemini model found. Please check your API key and model availability.")
    
    def analyze_text(self, text: str, complaint_id: Optional[str] = None, max_retries: int = 3, prompt_mode: str = 'job_scam') -> Dict[str, Any]:
        """
        Use Gemini AI to analyze text for scam indicators with retry logic for rate limits
        
        Args:
            text: Text to analyze
            complaint_id: Optional complaint ID for tracking
            max_retries: Maximum number of retries for rate limit errors (default: 3)
            prompt_mode: Prompt mode - 'job_scam' (detailed job scam analysis) or 'multi_category' (4-category classification)
            
        Returns:
            Analysis results from Gemini
        """
        # Get prompt based on mode
        if prompt_mode == 'multi_category':
            prompt = get_multi_category_prompt(text)
        else:
            prompt = get_scam_analysis_prompt(text)
        
        for attempt in range(max_retries):
            try:
                # Apply rate limiting if enabled
                if self.rate_limiter:
                    self.rate_limiter.wait_if_needed()
                
                response = self.model.generate_content(prompt)
                result_text = response.text
                
                # Try to extract JSON from response - use more robust extraction
                # First try to find JSON block (may be wrapped in markdown code blocks)
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
                if not json_match:
                    # Try without code blocks
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(1) if json_match.lastindex else json_match.group()
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as json_err:
                        logger.warning(f"JSON decode error, trying to fix: {json_err}")
                        # Log a snippet of the problematic JSON for debugging (first 500 chars)
                        error_snippet = json_str[max(0, json_err.pos - 50):json_err.pos + 50] if hasattr(json_err, 'pos') else json_str[:500]
                        logger.debug(f"Problematic JSON snippet: ...{error_snippet}...")
                        
                        # Try to fix common JSON issues
                        fixed_json = self._fix_json_string(json_str)
                        try:
                            return json.loads(fixed_json)
                        except json.JSONDecodeError as fix_err:
                            logger.error(f"Could not parse JSON even after fixing. Original error: {json_err}. Fix attempt error: {fix_err}")
                            # Try one more time with a more aggressive fix
                            try:
                                # Use json5 or try to extract just the essential fields
                                return self._extract_json_fields(json_str)
                            except Exception:
                                return self._parse_fallback_response(result_text)
                else:
                    # Fallback parsing
                    return self._parse_fallback_response(result_text)
                    
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit error (429)
                if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                    # Extract retry delay from error message if available
                    retry_delay = 15  # Default 15 seconds
                    retry_match = re.search(r'retry.*?(\d+)\s*seconds?', error_str, re.IGNORECASE)
                    if retry_match:
                        retry_delay = int(retry_match.group(1)) + 1  # Add 1 second buffer
                    
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limit error (attempt {attempt + 1}/{max_retries}): {error_str[:100]}")
                        logger.info(f"Waiting {retry_delay} seconds before retry...")
                        time.sleep(retry_delay)
                        continue  # Retry
                    else:
                        logger.error(f"Rate limit error after {max_retries} attempts: {error_str[:100]}")
                        return {
                            "scam_probability": 0,
                            "red_flags": {},
                            "financial_risk": {"level": "Unknown"},
                            "scam_type": {"primary_category": "Unknown", "subcategory": "Unknown"},
                            "victim_profile": {"risk_level": "Unknown"},
                            "recommendations": {},
                            "confidence": 0,
                            "error": f"Rate limit exceeded after {max_retries} retries"
                        }
                else:
                    # Non-rate-limit error - log and return error response
                    logger.error(f"Error analyzing with Gemini: {e}")
                    return {
                        "scam_probability": 0,
                        "red_flags": {},
                        "financial_risk": {"level": "Unknown"},
                        "scam_type": {"primary_category": "Unknown", "subcategory": "Unknown"},
                        "victim_profile": {"risk_level": "Unknown"},
                        "recommendations": {},
                        "confidence": 0,
                        "error": str(e)
                    }
        
        # Should not reach here, but just in case
        return {
            "scam_probability": 0,
            "red_flags": {},
            "financial_risk": {"level": "Unknown"},
            "scam_type": {"primary_category": "Unknown", "subcategory": "Unknown"},
            "victim_profile": {"risk_level": "Unknown"},
            "recommendations": {},
            "confidence": 0,
            "error": "Unexpected error in analyze_text"
        }
    
    def _extract_json_fields(self, json_str: str) -> Dict[str, Any]:
        """Extract JSON fields using regex when parsing fails completely"""
        result = {}
        
        # Extract scam_probability
        prob_match = re.search(r'"scam_probability"\s*:\s*(\d+)', json_str)
        if prob_match:
            result["scam_probability"] = int(prob_match.group(1))
        else:
            result["scam_probability"] = self._extract_score(json_str, "scam_probability")
        
        # Extract scam_type
        type_match = re.search(r'"scam_type"\s*:\s*\{[^}]*"primary_category"\s*:\s*"([^"]+)"', json_str)
        sub_match = re.search(r'"subcategory"\s*:\s*"([^"]+)"', json_str)
        result["scam_type"] = {
            "primary_category": type_match.group(1) if type_match else "Unknown",
            "subcategory": sub_match.group(1) if sub_match else "Unknown"
        }
        
        # Extract red_flags (simplified)
        result["red_flags"] = {}
        for category in ["communication", "financial", "job_posting", "hiring_process", "work_activity"]:
            flags = re.findall(rf'"{category}"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
            if flags:
                items = re.findall(r'"([^"]+)"', flags[0])
                result["red_flags"][category] = items
        
        # Extract other fields
        result["financial_risk"] = {"level": self._extract_text(json_str, "level"), "potential_loss": "Unknown", "explanation": ""}
        result["victim_profile"] = {"vulnerability_factors": [], "risk_level": self._extract_text(json_str, "risk_level")}
        result["evidence_strength"] = self._extract_text(json_str, "evidence_strength")
        result["recommendations"] = {"immediate_actions": [], "verification_steps": [], "reporting": []}
        
        conf_match = re.search(r'"confidence"\s*:\s*(\d+)', json_str)
        result["confidence"] = int(conf_match.group(1)) if conf_match else self._extract_score(json_str, "confidence")
        
        return result
    
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
    
    def _fix_json_string(self, json_str: str) -> str:
        """Try to fix common JSON issues"""
        original = json_str
        
        # Remove comments (JSON doesn't support comments)
        json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # Remove trailing commas before closing braces/brackets (most common issue)
        # Need to do this multiple times to handle nested structures
        # Pattern: comma followed by optional whitespace and closing bracket/brace
        # We iterate to handle deeply nested structures
        max_iterations = 20
        for i in range(max_iterations):
            prev_json = json_str
            # Remove trailing commas: ,] or ,} or , ] or , } (with any whitespace)
            json_str = re.sub(r',\s*\]', ']', json_str)
            json_str = re.sub(r',\s*\}', '}', json_str)
            # Also handle cases with newlines: ,\n] or ,\n}
            json_str = re.sub(r',\s*\n\s*([}\]])', r'\n\1', json_str)
            
            if prev_json == json_str:  # No more changes
                break
        
        # Fix missing commas between object properties
        # Pattern: }" or ]" should be }," or ],"
        json_str = re.sub(r'([}\]])"', r'\1, "', json_str)
        json_str = re.sub(r'([}\]])"', r'\1, "', json_str)  # Run twice for nested cases
        
        # Fix single quotes to double quotes for keys and string values
        # But be careful - only fix when it's clearly a JSON delimiter
        # Pattern: 'key': or : 'value'
        json_str = re.sub(r"'(\w+)'\s*:", r'"\1":', json_str)  # Keys
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)  # Values (simple case)
        
        # Try to handle unescaped quotes in string values
        # This is the trickiest part - we'll use a state machine approach
        # Find string values and escape internal quotes
        def fix_string_value(match):
            prefix = match.group(1)  # Everything before the value
            quote = match.group(2)  # Opening quote
            content = match.group(3)  # Content between quotes
            suffix = match.group(4)  # Closing quote and after
            
            # Escape unescaped quotes in content, but preserve escaped ones
            # Replace " that's not preceded by \ with \"
            fixed_content = re.sub(r'(?<!\\)"', r'\\"', content)
            return f'{prefix}{quote}{fixed_content}{quote}{suffix}'
        
        # Pattern: "key": "value with possible "quotes" inside"
        # This regex finds string values and fixes internal quotes
        json_str = re.sub(
            r'("[\w_]+"\s*:\s*)(")((?:[^"\\]|\\.)*)(")',
            fix_string_value,
            json_str
        )
        
        # Remove any control characters that might break JSON (except newlines/tabs)
        json_str = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', json_str)
        
        # If the fix made it worse (too many changes), return original
        # This is a safety check
        if len(json_str) < len(original) * 0.5:  # If we removed more than 50%, something went wrong
            return original
        
        return json_str

