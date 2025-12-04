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
from scam_detector.config import DEFAULT_WORKERS

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
        required=False,
        default='data/data-merged.csv',
        help='Path to input CSV file with complaints (default: data/data-merged.csv)'
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
    parser.add_argument(
        '--workers',
        type=int,
        default=DEFAULT_WORKERS,
        help=f'Number of parallel workers for analysis (default: {DEFAULT_WORKERS}, increase for faster processing)'
    )
    parser.add_argument(
        '--output-format',
        type=str,
        choices=['json', 'pdf', 'both'],
        default='json',
        help='Output format for report: json, pdf, or both (default: json)'
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
        # Track start time
        import time
        start_time = time.time()
        
        # Analyze dataset
        logger.info(f"Starting analysis of {input_path}")
        logger.info(f"Using {args.workers} worker(s) for parallel processing")
        results_df = detector.analyze_dataset(
            str(input_path),
            output_filename=args.results_file,
            workers=args.workers,
            prompt_mode=args.prompt_mode
        )
        
        # Calculate run time
        end_time = time.time()
        run_time_seconds = end_time - start_time
        run_time_minutes = run_time_seconds / 60
        
        # Get model name from detector
        model_name = detector.gemini_client.model_name
        
        # Generate report
        logger.info("Generating analysis report...")
        report = detector.generate_report(
            results_df=results_df,
            report_filename=args.report_file,
            output_format=args.output_format,
            model_name=model_name,
            run_time_seconds=run_time_seconds
        )
        
        # Print summary
        print("\n" + "="*50)
        print("JOB SCAM DETECTION ANALYSIS REPORT")
        print("="*50)
        print(f"Total Complaints Analyzed: {report['summary']['total_complaints']}")
        print(f"High Risk Complaints (≥{report['summary'].get('high_risk_threshold', 70)}%): {report['summary']['high_risk_complaints']} ({report['summary']['high_risk_percentage']:.1f}%)")
        print(f"Average Risk Score (Gemini scam_probability): {report['summary']['average_risk_score']:.1f}%")
        print(f"Median Risk Score: {report['summary']['median_risk_score']:.1f}%")
        
        # Top Red Flags by Category
        top_red_flags_by_category = report.get('top_red_flags_by_category', {})
        if top_red_flags_by_category:
            print("\nTop Red Flags by Category:")
            category_names = {
                'communication': 'Communication',
                'financial': 'Financial',
                'job_posting': 'Job Posting',
                'hiring_process': 'Hiring Process',
                'work_activity': 'Work Activity'
            }
            for category, flags_dict in top_red_flags_by_category.items():
                if flags_dict:
                    print(f"\n  {category_names.get(category, category)}:")
                    for flag, count in list(flags_dict.items())[:3]:
                        print(f"    - {flag}: {count} occurrences")
        else:
            # Fallback to overall top flags
            print("\nTop Red Flags Found:")
            for flag, count in list(report['top_red_flags'].items())[:5]:
                print(f"  - {flag}: {count} occurrences")
        
        # Top Vulnerability Factors
        vulnerability_factors = report.get('top_vulnerability_factors', {})
        if vulnerability_factors:
            print("\nTop Vulnerability Factors:")
            for factor, count in list(vulnerability_factors.items())[:5]:
                print(f"  - {factor}: {count} occurrences")
        
        print("\nScam Type Distribution:")
        for scam_type, count in list(report['scam_type_distribution'].items())[:5]:
            print(f"  - {scam_type}: {count} cases")
        
        # Print file locations
        results_path = detector.file_handler.get_results_path(args.results_file)
        
        print(f"\nDetailed results saved to: {results_path}")
        
        if args.output_format in ['json', 'both']:
            report_path = detector.file_handler.get_report_path(args.report_file)
            print(f"Analysis report (JSON) saved to: {report_path}")
        
        if args.output_format in ['pdf', 'both']:
            report_res_dir = Path(__file__).parent / "report_res"
            print(f"Analysis report (PDF) saved to: {report_res_dir}/")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error in main analysis: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

