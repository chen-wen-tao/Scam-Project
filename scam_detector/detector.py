"""
Main Job Scam Detector class
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .text_processor import TextProcessor
from .gemini_client import GeminiClient
from .report_generator import ReportGenerator
from .file_handler import FileHandler

logger = logging.getLogger(__name__)


class JobScamDetector:
    """
    A comprehensive job scam detection system using Google Gemini AI
    """
    
    def __init__(self, api_key: Optional[str] = None, output_dir: Optional[Path] = None):
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
    
    def analyze_complaint(self, complaint_text: str, complaint_id: Optional[str] = None, prompt_mode: str = 'job_scam') -> Dict[str, Any]:
        """
        Comprehensive analysis of a single complaint
        
        Args:
            complaint_text: The complaint text to analyze
            complaint_id: Optional complaint ID
            prompt_mode: Prompt mode - 'job_scam' (detailed analysis) or 'multi_category' (4-category classification)
            
        Returns:
            Complete analysis results
        """
        # Preprocess text
        cleaned_text = self.text_processor.preprocess_text(complaint_text)
        
        # Get Gemini analysis - this is the primary and only source of truth
        gemini_analysis = self.gemini_client.analyze_text(cleaned_text, complaint_id, prompt_mode=prompt_mode)
        
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
    
    def analyze_dataset(self, csv_file_path: str, output_filename: Optional[str] = None, workers: int = 1, prompt_mode: str = 'job_scam') -> pd.DataFrame:
        """
        Analyze a dataset of complaints
        
        Args:
            csv_file_path: Path to CSV file with complaints
            output_filename: Optional output filename (will be saved to detect_res/)
            workers: Number of parallel workers (threads) to use for API calls
            
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
        
        logger.info(f"Analyzing {len(df)} complaints...")
        
        results: List[Dict[str, Any]] = []
        
        def _process_row(idx_row_tuple):
            idx, row = idx_row_tuple
            try:
                complaint_text = row['Consumer complaint narrative']
                complaint_id = row[complaint_id_col] if complaint_id_col else f"complaint_{idx}"
                logger.info(f"Analyzing complaint {idx + 1}/{len(df)}: {complaint_id}")
                return self.analyze_complaint(complaint_text, complaint_id, prompt_mode=prompt_mode)
            except Exception as e:
                logger.error(f"Error analyzing complaint {idx}: {e}")
                return None
        
        if workers and workers > 1:
            logger.info(f"Running with {workers} parallel workers...")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_process_row, (idx, row)): idx for idx, row in df.iterrows()}
                for future in as_completed(futures):
                    res = future.result()
                    if res is not None:
                        results.append(res)
        else:
            for idx, row in df.iterrows():
                res = _process_row((idx, row))
                if res is not None:
                    results.append(res)
        
        # Convert gemini_analysis dict to JSON string for CSV compatibility
        for result in results:
            if 'gemini_analysis' in result and isinstance(result['gemini_analysis'], dict):
                result['gemini_analysis'] = json.dumps(result['gemini_analysis'])
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        output_path = self.file_handler.save_results_csv(results_df, output_filename)
        logger.info(f"Results saved to {output_path}")
        
        return results_df
    
    def generate_report(self, results_df: Optional[pd.DataFrame] = None, report_filename: Optional[str] = None, output_format: str = 'json', model_name: Optional[str] = None, run_time_seconds: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report
        
        Args:
            results_df: Optional DataFrame with analysis results (if None, loads from detect_res/)
            report_filename: Optional report filename
            output_format: Output format - 'json', 'pdf', or 'both' (default: 'json')
            model_name: Optional model name used for analysis
            run_time_seconds: Optional total run time in seconds
            
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
        
        # Add metadata to report
        if model_name:
            report['metadata'] = report.get('metadata', {})
            report['metadata']['model_name'] = model_name
        if run_time_seconds is not None:
            report['metadata'] = report.get('metadata', {})
            report['metadata']['run_time_seconds'] = run_time_seconds
        
        # Save report in requested format(s)
        if output_format in ['json', 'both']:
            report_path = self.file_handler.save_report_json(report, report_filename)
            logger.info(f"Report saved to {report_path}")
        
        if output_format in ['pdf', 'both']:
            pdf_path = self.file_handler.save_report_pdf(report, results_df=results_df, filename=report_filename, model_name=model_name, run_time_seconds=run_time_seconds)
            logger.info(f"PDF report saved to {pdf_path}")
        
        return report

