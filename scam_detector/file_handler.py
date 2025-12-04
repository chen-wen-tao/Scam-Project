"""
File I/O operations for the scam detection system
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime
from .config import DEFAULT_OUTPUT_DIR, DEFAULT_RESULTS_CSV, DEFAULT_REPORT_JSON


class FileHandler:
    """Handles file operations for saving and loading results"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize file handler
        
        Args:
            output_dir: Directory to save results (defaults to detect_res/)
        """
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_results_csv(self, results_df: pd.DataFrame, filename: Optional[str] = None) -> Path:
        """
        Save analysis results to CSV
        
        Args:
            results_df: DataFrame with results
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        base_name = (filename or DEFAULT_RESULTS_CSV).replace('.csv', '')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_name = f"{base_name}_{timestamp}.csv"
        filepath = self.output_dir / final_name
        results_df.to_csv(filepath, index=False)
        return filepath
    
    def save_report_json(self, report: dict, filename: Optional[str] = None) -> Path:
        """
        Save analysis report to JSON
        
        Args:
            report: Report dictionary
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        base_name = (filename or DEFAULT_REPORT_JSON).replace('.json', '')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_name = f"{base_name}_{timestamp}.json"
        filepath = self.output_dir / final_name
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        return filepath
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load CSV file
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Loaded DataFrame
        """
        return pd.read_csv(filepath)
    
    def load_json(self, filepath: str) -> dict:
        """
        Load JSON file
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Loaded dictionary
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def get_results_path(self, filename: Optional[str] = None) -> Path:
        """Get path to results file"""
        filename = filename or DEFAULT_RESULTS_CSV
        return self.output_dir / filename
    
    def get_report_path(self, filename: Optional[str] = None) -> Path:
        """Get path to report file"""
        filename = filename or DEFAULT_REPORT_JSON
        return self.output_dir / filename
    
    def save_report_pdf(self, report: dict, results_df=None, filename: Optional[str] = None, model_name: Optional[str] = None, run_time_seconds: Optional[float] = None) -> Path:
        """
        Save analysis report to PDF
        
        Args:
            report: Report dictionary
            results_df: Optional DataFrame with detailed results (needed for PDF)
            filename: Optional custom filename
            model_name: Optional model name used for analysis
            run_time_seconds: Optional total run time in seconds
            
        Returns:
            Path to saved PDF file
        """
        from .pdf_generator import PDFGenerator
        import pandas as pd
        
        # Save PDF to report_res directory
        report_res_dir = self.output_dir.parent / "report_res"
        pdf_gen = PDFGenerator(output_dir=report_res_dir)
        return pdf_gen.generate_pdf(report, results_df=results_df, filename=filename, model_name=model_name, run_time_seconds=run_time_seconds)

