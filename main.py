#!/usr/bin/env python3
"""
Main entry point for the Job Scam Detection System
"""

import os
import sys
import logging
import argparse
from pathlib import Path

from scam_detector import JobScamDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the job scam detector"""
    
    parser = argparse.ArgumentParser(
        description='Job Scam Detection System using Google Gemini AI'
    )
    parser.add_argument(
        '--input', 
        type=str,
        required=True,
        help='Path to input CSV file with complaints'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: detect_res/)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Google Gemini API key (or set GEMINI_API_KEY env var)'
    )
    parser.add_argument(
        '--results-file',
        type=str,
        default=None,
        help='Custom filename for results CSV (default: scam_analysis_results.csv)'
    )
    parser.add_argument(
        '--report-file',
        type=str,
        default=None,
        help='Custom filename for report JSON (default: scam_analysis_report.json)'
    )
    
    args = parser.parse_args()
    
    # Check for API key
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: Gemini API key is required")
        print("Set GEMINI_API_KEY environment variable or use --api-key argument")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Initialize detector
    output_dir = Path(args.output_dir) if args.output_dir else None
    try:
        detector = JobScamDetector(api_key=api_key, output_dir=output_dir)  # type: ignore
    except Exception as e:
        print(f"Error initializing detector: {e}")
        sys.exit(1)
    
    try:
        # Analyze dataset
        logger.info(f"Starting analysis of {input_path}")
        results_df = detector.analyze_dataset(
            str(input_path),
            output_filename=args.results_file
        )
        
        # Generate report
        logger.info("Generating analysis report...")
        report = detector.generate_report(
            results_df=results_df,
            report_filename=args.report_file
        )
        
        # Print summary
        print("\n" + "="*50)
        print("JOB SCAM DETECTION ANALYSIS REPORT")
        print("="*50)
        print(f"Total Complaints Analyzed: {report['summary']['total_complaints']}")
        print(f"High Risk Complaints (≥{report['summary'].get('high_risk_threshold', 70)}%): {report['summary']['high_risk_complaints']} ({report['summary']['high_risk_percentage']:.1f}%)")
        print(f"Average Risk Score (Gemini scam_probability): {report['summary']['average_risk_score']:.1f}%")
        print(f"Median Risk Score: {report['summary']['median_risk_score']:.1f}%")
        
        print("\nTop Red Flags Found:")
        for flag, count in list(report['top_red_flags'].items())[:5]:
            print(f"  - {flag}: {count} occurrences")
        
        print("\nScam Type Distribution:")
        for scam_type, count in list(report['scam_type_distribution'].items())[:5]:
            print(f"  - {scam_type}: {count} cases")
        
        # Print file locations
        results_path = detector.file_handler.get_results_path(args.results_file)
        report_path = detector.file_handler.get_report_path(args.report_file)
        
        print(f"\nDetailed results saved to: {results_path}")
        print(f"Analysis report saved to: {report_path}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error in main analysis: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

