"""
Main Job Scam Detector class
"""

import os
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from .text_processor import TextProcessor
from .gemini_client import GeminiClient
from .report_generator import ReportGenerator
from .file_handler import FileHandler

logger = logging.getLogger(__name__)


class JobScamDetector:
    """
    A comprehensive job scam detection system using Google Gemini AI
    """
    
    def __init__(self, api_key: str = None, output_dir: Path = None):
        """
        Initialize the Job Scam Detector
        
        Args:
            api_key: Google Gemini API key. If None, will try to get from environment
            output_dir: Directory to save results (defaults to detect_res/)
        """
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize components
        self.text_processor = TextProcessor()
        self.gemini_client = GeminiClient(api_key)
        self.file_handler = FileHandler(output_dir)
        self.report_generator = ReportGenerator()
    
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
        cleaned_text = self.text_processor.preprocess_text(complaint_text)
        
        # Get Gemini analysis - this is the primary and only source of truth
        gemini_analysis = self.gemini_client.analyze_text(cleaned_text, complaint_id)
        
        # Extract scam probability directly from Gemini analysis
        scam_probability = gemini_analysis.get('scam_probability', 0)
        
        # Combine results - use Gemini's scam_probability as the risk score
        analysis = {
            'complaint_id': complaint_id,
            'timestamp': datetime.now().isoformat(),
            'text_length': len(cleaned_text),
            'gemini_analysis': gemini_analysis,
            'risk_score': scam_probability  # Direct use of Gemini's scam_probability
        }
        
        return analysis
    
    def analyze_dataset(self, csv_file_path: str, output_filename: str = None) -> pd.DataFrame:
        """
        Analyze a dataset of complaints
        
        Args:
            csv_file_path: Path to CSV file with complaints
            output_filename: Optional output filename (will be saved to detect_res/)
            
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
        
        # Save results
        output_path = self.file_handler.save_results_csv(results_df, output_filename)
        logger.info(f"Results saved to {output_path}")
        
        return results_df
    
    def generate_report(self, results_df: pd.DataFrame = None, report_filename: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report
        
        Args:
            results_df: Optional DataFrame with analysis results (if None, loads from detect_res/)
            report_filename: Optional report filename
            
        Returns:
            Report dictionary
        """
        if results_df is None:
            # Try to load from default location
            try:
                results_path = self.file_handler.get_results_path()
                results_df = pd.read_csv(results_path)
                logger.info(f"Loaded results from {results_path}")
            except FileNotFoundError:
                raise ValueError("No results DataFrame provided and no saved results found in detect_res/")
        
        report = self.report_generator.generate_report(results_df)
        
        # Save report
        report_path = self.file_handler.save_report_json(report, report_filename)
        logger.info(f"Report saved to {report_path}")
        
        return report

