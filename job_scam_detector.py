#!/usr/bin/env python3
"""
Job Scam Detection System using Google Gemini
Analyzes job-related messages and complaints to detect potential scams
"""

import os
import csv
import json
import re
import pandas as pd
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JobScamDetector:
    """
    A comprehensive job scam detection system using Google Gemini AI
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Job Scam Detector
        
        Args:
            api_key: Google Gemini API key. If None, will try to get from environment
        """
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # First, let's check what models are actually available
        try:
            available_models = genai.list_models()
            logger.info("Checking available models...")
            
            # Find a model that supports generateContent
            self.model = None
            for model in available_models:
                if 'generateContent' in model.supported_generation_methods:
                    model_name = model.name.replace('models/', '')
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        # Test the model with a simple request
                        test_response = self.model.generate_content("Test")
                        logger.info(f"Successfully initialized model: {model_name}")
                        break
                    except Exception as e:
                        logger.warning(f"Model {model_name} failed test: {e}")
                        continue
            
            if self.model is None:
                # Fallback to manual model names if list_models fails
                fallback_models = ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-1.0-pro']
                for model_name in fallback_models:
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        test_response = self.model.generate_content("Test")
                        logger.info(f"Successfully initialized fallback model: {model_name}")
                        break
                    except Exception as e:
                        logger.warning(f"Fallback model {model_name} not available: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error checking available models: {e}")
            # Try the most common model names as last resort
            fallback_models = ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-1.0-pro']
            for model_name in fallback_models:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    test_response = self.model.generate_content("Test")
                    logger.info(f"Successfully initialized model: {model_name}")
                    break
                except Exception as e:
                    logger.warning(f"Model {model_name} not available: {e}")
                    continue
        
        if self.model is None:
            raise ValueError("No compatible Gemini model found. Please check your API key and model availability. Try running 'python check_models.py' to see available models.")
        
        # Scam indicators based on analysis of the dataset
        self.scam_indicators = {
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
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for analysis
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned text
        """
        if not text or pd.isna(text):
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', str(text).strip())
        
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
    
    def analyze_with_gemini(self, text: str, complaint_id: str = None) -> Dict[str, Any]:
        """
        Use Gemini AI to analyze text for scam indicators
        
        Args:
            text: Text to analyze
            complaint_id: Optional complaint ID for tracking
            
        Returns:
            Analysis results from Gemini
        """
        prompt = f"""
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
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback parsing
                return {
                    "scam_probability": self._extract_score(result_text, "scam_probability"),
                    "red_flags": self._extract_list(result_text, "red_flags"),
                    "financial_risk": self._extract_text(result_text, "financial_risk"),
                    "scam_type": self._extract_text(result_text, "scam_type"),
                    "victim_profile": self._extract_text(result_text, "victim_profile"),
                    "recommendations": self._extract_list(result_text, "recommendations"),
                    "confidence": self._extract_score(result_text, "confidence")
                }
                
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
    
    def analyze_complaint(self, complaint_text: str, complaint_id: str = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of a single complaint
        
        Args:
            complaint_text: The complaint text to analyze
            complaint_id: Optional complaint ID
            
        Returns:
            Complete analysis results
        """
        # Preprocess text
        cleaned_text = self.preprocess_text(complaint_text)
        
        # Extract rule-based indicators
        indicators = self.extract_scam_indicators(cleaned_text)
        
        # Get Gemini analysis
        gemini_analysis = self.analyze_with_gemini(cleaned_text, complaint_id)
        
        # Combine results
        analysis = {
            'complaint_id': complaint_id,
            'timestamp': datetime.now().isoformat(),
            'text_length': len(cleaned_text),
            'rule_based_indicators': indicators,
            'gemini_analysis': gemini_analysis,
            'overall_risk_score': self._calculate_overall_risk(indicators, gemini_analysis)
        }
        
        return analysis
    
    def _calculate_overall_risk(self, indicators: Dict, gemini_analysis: Dict) -> int:
        """
        Calculate overall risk score combining rule-based and AI analysis
        
        Args:
            indicators: Rule-based indicators found
            gemini_analysis: Gemini AI analysis results
            
        Returns:
            Overall risk score (0-100)
        """
        # Rule-based score (0-50)
        rule_score = 0
        for category, found_indicators in indicators.items():
            rule_score += min(len(found_indicators) * 5, 15)  # Max 15 per category
        
        # Gemini score (0-50)
        gemini_score = gemini_analysis.get('scam_probability', 0) * 0.5
        
        # Combine scores
        total_score = min(rule_score + gemini_score, 100)
        return int(total_score)
    
    def analyze_dataset(self, csv_file_path: str, output_file: str = None) -> pd.DataFrame:
        """
        Analyze a dataset of complaints
        
        Args:
            csv_file_path: Path to CSV file with complaints
            output_file: Optional output file for results
            
        Returns:
            DataFrame with analysis results
        """
        logger.info(f"Loading dataset from {csv_file_path}")
        
        # Load data
        df = pd.read_csv(csv_file_path)
        
        # Ensure we have the required columns
        if 'Consumer complaint narrative' not in df.columns:
            raise ValueError("CSV must contain 'Consumer complaint narrative' column")
        
        complaint_id_col = 'Complaint ID' if 'Complaint ID' in df.columns else None
        
        results = []
        
        logger.info(f"Analyzing {len(df)} complaints...")
        
        for idx, row in df.iterrows():
            try:
                complaint_text = row['Consumer complaint narrative']
                complaint_id = row[complaint_id_col] if complaint_id_col else f"complaint_{idx}"
                
                logger.info(f"Analyzing complaint {idx + 1}/{len(df)}: {complaint_id}")
                
                analysis = self.analyze_complaint(complaint_text, complaint_id)
                results.append(analysis)
                
            except Exception as e:
                logger.error(f"Error analyzing complaint {idx}: {e}")
                continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results if output file specified
        if output_file:
            results_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        
        return results_df
    
    def generate_report(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report
        
        Args:
            results_df: DataFrame with analysis results
            
        Returns:
            Report dictionary
        """
        total_complaints = len(results_df)
        
        # Risk distribution
        risk_distribution = results_df['overall_risk_score'].value_counts().sort_index()
        
        # High risk complaints (score >= 70)
        high_risk = results_df[results_df['overall_risk_score'] >= 70]
        
        # Most common red flags
        all_red_flags = []
        for indicators in results_df['rule_based_indicators']:
            for category, flags in indicators.items():
                all_red_flags.extend(flags)
        
        red_flag_counts = pd.Series(all_red_flags).value_counts()
        
        # Scam type distribution
        scam_types = []
        for analysis in results_df['gemini_analysis']:
            if isinstance(analysis, dict) and 'scam_type' in analysis:
                scam_types.append(analysis['scam_type'])
        
        scam_type_distribution = pd.Series(scam_types).value_counts()
        
        report = {
            'summary': {
                'total_complaints': total_complaints,
                'high_risk_complaints': len(high_risk),
                'high_risk_percentage': (len(high_risk) / total_complaints * 100) if total_complaints > 0 else 0,
                'average_risk_score': results_df['overall_risk_score'].mean(),
                'median_risk_score': results_df['overall_risk_score'].median()
            },
            'risk_distribution': risk_distribution.to_dict(),
            'top_red_flags': red_flag_counts.head(10).to_dict(),
            'scam_type_distribution': scam_type_distribution.to_dict(),
            'high_risk_complaints': high_risk[['complaint_id', 'overall_risk_score']].to_dict('records') if len(high_risk) > 0 else []
        }
        
        return report

def main():
    """Main function to run the job scam detector"""
    
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Please set your GEMINI_API_KEY environment variable")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    # Initialize detector
    detector = JobScamDetector(api_key)
    
    # Analyze the dataset
    csv_file = "/Users/chenwentao/Desktop/Scam Project/data-ashley(Sheet1).csv"
    output_file = "/Users/chenwentao/Desktop/Scam Project/scam_analysis_results.csv"
    
    try:
        # Analyze dataset
        results_df = detector.analyze_dataset(csv_file, output_file)
        
        # Generate report
        report = detector.generate_report(results_df)
        
        # Print summary
        print("\n" + "="*50)
        print("JOB SCAM DETECTION ANALYSIS REPORT")
        print("="*50)
        print(f"Total Complaints Analyzed: {report['summary']['total_complaints']}")
        print(f"High Risk Complaints: {report['summary']['high_risk_complaints']} ({report['summary']['high_risk_percentage']:.1f}%)")
        print(f"Average Risk Score: {report['summary']['average_risk_score']:.1f}")
        print(f"Median Risk Score: {report['summary']['median_risk_score']:.1f}")
        
        print("\nTop Red Flags Found:")
        for flag, count in list(report['top_red_flags'].items())[:5]:
            print(f"  - {flag}: {count} occurrences")
        
        print("\nScam Type Distribution:")
        for scam_type, count in list(report['scam_type_distribution'].items())[:5]:
            print(f"  - {scam_type}: {count} cases")
        
        # Save report
        report_file = "/Users/chenwentao/Desktop/Scam Project/scam_analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_file}")
        print(f"Analysis report saved to: {report_file}")
        
    except Exception as e:
        logger.error(f"Error in main analysis: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
