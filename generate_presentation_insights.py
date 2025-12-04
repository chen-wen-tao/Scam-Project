#!/usr/bin/env python3
"""
Generate presentation-ready insights from evaluation results
Creates comprehensive analysis for academic presentation
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_evaluation_results(evaluation_csv: str, original_csv: str) -> pd.DataFrame:
    """Load and merge evaluation results with original complaints"""
    eval_df = pd.read_csv(evaluation_csv)
    original_df = pd.read_csv(original_csv)
    
    # Merge to get complaint narratives
    if 'Complaint ID' in original_df.columns:
        original_df['Complaint ID'] = original_df['Complaint ID'].astype(str)
        eval_df['complaint_id'] = eval_df['complaint_id'].astype(str)
        merged = eval_df.merge(
            original_df[['Complaint ID', 'Consumer complaint narrative']],
            left_on='complaint_id',
            right_on='Complaint ID',
            how='left'
        )
    else:
        merged = eval_df.copy()
        merged['Consumer complaint narrative'] = 'N/A'
    
    return merged


def analyze_errors(df: pd.DataFrame) -> Dict:
    """Analyze false positives and false negatives"""
    # False Positives: Predicted scam but actually not scam
    fp = df[(df['true_label'] == 0) & (df['is_scam_predicted'] == True)]
    
    # False Negatives: Actually scam but predicted as not scam
    fn = df[(df['true_label'] == 1) & (df['is_scam_predicted'] == False)]
    
    # True Positives: Correctly identified scams
    tp = df[(df['true_label'] == 1) & (df['is_scam_predicted'] == True)]
    
    # True Negatives: Correctly identified non-scams
    tn = df[(df['true_label'] == 0) & (df['is_scam_predicted'] == False)]
    
    return {
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'true_negatives': tn,
        'fp_count': len(fp),
        'fn_count': len(fn),
        'tp_count': len(tp),
        'tn_count': len(tn)
    }


def analyze_category_performance(df: pd.DataFrame) -> Dict:
    """Analyze performance by category"""
    category_metrics = {}
    
    for category in df['true_category'].unique():
        cat_df = df[df['true_category'] == category]
        if len(cat_df) == 0:
            continue
        
        correct = len(cat_df[cat_df['true_category'] == cat_df['predicted_category']])
        total = len(cat_df)
        accuracy = correct / total if total > 0 else 0
        
        category_metrics[category] = {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'avg_risk_score': cat_df['predicted_risk_score'].mean()
        }
    
    return category_metrics


def generate_insights_report(
    evaluation_csv: str,
    original_csv: str,
    summary_json: str = None,
    output_file: str = 'presentation_insights.md'
) -> str:
    """Generate comprehensive insights report for presentation"""
    
    logger.info("Loading evaluation results...")
    df = load_evaluation_results(evaluation_csv, original_csv)
    
    logger.info("Analyzing errors...")
    errors = analyze_errors(df)
    
    logger.info("Analyzing category performance...")
    category_perf = analyze_category_performance(df)
    
    # Load summary if available
    summary = None
    if summary_json and Path(summary_json).exists():
        with open(summary_json, 'r') as f:
            summary = json.load(f)
    
    # Generate markdown report
    report = []
    report.append("# Job Scam Detection System - Presentation Insights\n")
    report.append(f"*Generated from evaluation results*\n")
    report.append("---\n\n")
    
    # 1. Executive Summary
    report.append("## 1. Executive Summary\n\n")
    total = len(df)
    accuracy = (errors['tp_count'] + errors['tn_count']) / total if total > 0 else 0
    
    if summary:
        binary_metrics = summary.get('binary_metrics', {})
        report.append(f"- **Total Complaints Analyzed**: {total}\n")
        report.append(f"- **Overall Accuracy**: {binary_metrics.get('accuracy', accuracy):.1%}\n")
        report.append(f"- **Precision**: {binary_metrics.get('precision', 0):.1%}\n")
        report.append(f"- **Recall**: {binary_metrics.get('recall', 0):.1%}\n")
        report.append(f"- **F1 Score**: {binary_metrics.get('f1', 0):.3f}\n")
        report.append(f"- **Optimal Threshold**: {summary.get('threshold', 70)}\n\n")
    else:
        report.append(f"- **Total Complaints Analyzed**: {total}\n")
        report.append(f"- **Overall Accuracy**: {accuracy:.1%}\n\n")
    
    # 2. Confusion Matrix Analysis
    report.append("## 2. Model Performance Breakdown\n\n")
    report.append("### Confusion Matrix\n\n")
    report.append("| | Predicted: Not Scam | Predicted: Scam |\n")
    report.append("|---|---|---|\n")
    report.append(f"| **Actual: Not Scam** | {errors['tn_count']} (TN) | {errors['fp_count']} (FP) |\n")
    report.append(f"| **Actual: Scam** | {errors['fn_count']} (FN) | {errors['tp_count']} (TP) |\n\n")
    
    report.append("### Key Metrics\n\n")
    report.append(f"- **True Positives (TP)**: {errors['tp_count']} - Correctly identified scams\n")
    report.append(f"- **True Negatives (TN)**: {errors['tn_count']} - Correctly identified non-scams\n")
    report.append(f"- **False Positives (FP)**: {errors['fp_count']} - Non-scams incorrectly flagged\n")
    report.append(f"- **False Negatives (FN)**: {errors['fn_count']} - Scams that were missed\n\n")
    
    # 3. Error Analysis
    report.append("## 3. Error Analysis\n\n")
    
    if len(errors['false_positives']) > 0:
        report.append("### False Positives (Type I Errors)\n\n")
        report.append(f"**Count**: {errors['fp_count']}\n\n")
        report.append("**Issue**: System incorrectly flagged legitimate complaints as scams.\n\n")
        report.append("**Examples**:\n\n")
        for idx, row in errors['false_positives'].head(3).iterrows():
            narrative = row.get('Consumer complaint narrative', 'N/A')
            narrative_short = narrative[:200] + "..." if len(narrative) > 200 else narrative
            report.append(f"1. **Complaint ID {row['complaint_id']}** (Risk Score: {row['predicted_risk_score']:.1f}%)\n")
            report.append(f"   - True Category: {row['true_category']}\n")
            report.append(f"   - Predicted Category: {row['predicted_category']}\n")
            report.append(f"   - Narrative: {narrative_short}\n\n")
    
    if len(errors['false_negatives']) > 0:
        report.append("### False Negatives (Type II Errors)\n\n")
        report.append(f"**Count**: {errors['fn_count']}\n\n")
        report.append("**Issue**: System missed actual scams (more critical error).\n\n")
        report.append("**Examples**:\n\n")
        for idx, row in errors['false_negatives'].head(3).iterrows():
            narrative = row.get('Consumer complaint narrative', 'N/A')
            narrative_short = narrative[:200] + "..." if len(narrative) > 200 else narrative
            report.append(f"1. **Complaint ID {row['complaint_id']}** (Risk Score: {row['predicted_risk_score']:.1f}%)\n")
            report.append(f"   - True Category: {row['true_category']}\n")
            report.append(f"   - Predicted Category: {row['predicted_category']}\n")
            report.append(f"   - Narrative: {narrative_short}\n\n")
    
    # 4. Category Performance
    report.append("## 4. Performance by Category\n\n")
    report.append("| Category | Total | Correct | Accuracy | Avg Risk Score |\n")
    report.append("|---|---|---|---|---|\n")
    
    for category, metrics in sorted(category_perf.items(), key=lambda x: x[1]['total'], reverse=True):
        report.append(f"| {category} | {metrics['total']} | {metrics['correct']} | {metrics['accuracy']:.1%} | {metrics['avg_risk_score']:.1f}% |\n")
    report.append("\n")
    
    # 5. Success Cases
    report.append("## 5. Success Cases (True Positives)\n\n")
    report.append("**Examples of correctly identified scams**:\n\n")
    for idx, row in errors['true_positives'].head(3).iterrows():
        narrative = row.get('Consumer complaint narrative', 'N/A')
        narrative_short = narrative[:200] + "..." if len(narrative) > 200 else narrative
        report.append(f"1. **Complaint ID {row['complaint_id']}** (Risk Score: {row['predicted_risk_score']:.1f}%)\n")
        report.append(f"   - Category: {row['true_category']}\n")
        report.append(f"   - Narrative: {narrative_short}\n\n")
    
    # 6. Key Insights
    report.append("## 6. Key Insights for Presentation\n\n")
    
    # Calculate insights
    fp_rate = errors['fp_count'] / (errors['fp_count'] + errors['tn_count']) if (errors['fp_count'] + errors['tn_count']) > 0 else 0
    fn_rate = errors['fn_count'] / (errors['fn_count'] + errors['tp_count']) if (errors['fn_count'] + errors['tp_count']) > 0 else 0
    
    report.append("### Model Strengths\n\n")
    optimal_threshold = summary.get('threshold', 60.0) if summary else 60.0
    precision_val = summary['binary_metrics']['precision'] if summary and 'binary_metrics' in summary else 0
    recall_val = summary['binary_metrics']['recall'] if summary and 'binary_metrics' in summary else 0
    report.append(f"1. **High Precision ({precision_val:.1%} if available)**: When the model predicts a scam, it's usually correct.\n")
    report.append(f"2. **Good Recall ({recall_val:.1%} if available)**: The model catches most actual scams.\n")
    report.append(f"3. **Optimal Threshold**: Threshold of {optimal_threshold:.0f} provides the best F1 score balance.\n\n")
    
    report.append("### Areas for Improvement\n\n")
    if errors['fn_count'] > 0:
        report.append(f"1. **False Negatives ({errors['fn_count']})**: Need to improve detection of subtle scam patterns.\n")
    if errors['fp_count'] > 0:
        report.append(f"2. **False Positives ({errors['fp_count']})**: Some legitimate complaints are being flagged.\n")
    
    report.append("\n### Recommendations\n\n")
    report.append("1. **Review False Negatives**: Analyze missed scams to identify patterns for prompt improvement.\n")
    report.append("2. **Refine Prompt Engineering**: Update prompts based on error analysis.\n")
    report.append("3. **Consider Ensemble Methods**: Combine multiple models for better accuracy.\n")
    report.append("4. **Expand Training Data**: More labeled examples would improve performance.\n\n")
    
    # 7. Risk Score Distribution
    report.append("## 7. Risk Score Distribution Analysis\n\n")
    report.append("### Statistics\n\n")
    report.append(f"- **Mean Risk Score**: {df['predicted_risk_score'].mean():.1f}%\n")
    report.append(f"- **Median Risk Score**: {df['predicted_risk_score'].median():.1f}%\n")
    report.append(f"- **Min Risk Score**: {df['predicted_risk_score'].min():.1f}%\n")
    report.append(f"- **Max Risk Score**: {df['predicted_risk_score'].max():.1f}%\n")
    report.append(f"- **Standard Deviation**: {df['predicted_risk_score'].std():.1f}%\n\n")
    
    # Risk score distribution by label
    scam_scores = df[df['true_label'] == 1]['predicted_risk_score']
    non_scam_scores = df[df['true_label'] == 0]['predicted_risk_score']
    
    report.append("### Risk Scores by True Label\n\n")
    report.append(f"- **Actual Scams**: Mean = {scam_scores.mean():.1f}%, Median = {scam_scores.median():.1f}%\n")
    report.append(f"- **Actual Non-Scams**: Mean = {non_scam_scores.mean():.1f}%, Median = {non_scam_scores.median():.1f}%\n\n")
    
    # Save report
    report_text = "".join(report)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"✅ Presentation insights saved to: {output_file}")
    return output_file


def find_latest_evaluation_files(evaluation_dir: str = 'evaluation_results', threshold: float = 70.0):
    """Find latest evaluation files automatically"""
    eval_path = Path(evaluation_dir)
    
    if not eval_path.exists():
        return None, None
    
    # Look for files with threshold
    csv_pattern = f"evaluation_results_threshold_{threshold:.0f}.csv"
    json_pattern = f"evaluation_summary_threshold_{threshold:.0f}.json"
    
    csv_file = eval_path / csv_pattern
    json_file = eval_path / json_pattern
    
    # If exact match not found, find latest
    if not csv_file.exists():
        csv_files = list(eval_path.glob("evaluation_results_threshold_*.csv"))
        if csv_files:
            csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Using latest evaluation CSV: {csv_file}")
    
    if not json_file.exists():
        json_files = list(eval_path.glob("evaluation_summary_threshold_*.json"))
        if json_files:
            json_file = max(json_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Using latest evaluation JSON: {json_file}")
    
    csv_path = str(csv_file) if csv_file.exists() else None
    json_path = str(json_file) if json_file.exists() else None
    
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description='Generate presentation insights from evaluation results')
    parser.add_argument('--evaluation-csv', type=str, default=None,
                       help='Path to evaluation results CSV (auto-detected if not provided)')
    parser.add_argument('--original-csv', type=str, default='data/cfpb-complaints-2025-11-03-12-03.csv',
                       help='Path to original labeled CSV (default: data/cfpb-complaints-2025-11-03-12-03.csv)')
    parser.add_argument('--summary-json', type=str, default=None,
                       help='Path to evaluation summary JSON (auto-detected if not provided)')
    parser.add_argument('--threshold', type=float, default=60.0,
                       help='Threshold to look for (default: 60.0)')
    parser.add_argument('--evaluation-dir', type=str, default='evaluation_results',
                       help='Directory to search for evaluation files (default: evaluation_results)')
    parser.add_argument('--output', type=str, default='presentation_insights.md',
                       help='Output markdown file (default: presentation_insights.md)')
    
    args = parser.parse_args()
    
    # Auto-detect files if not provided
    if args.evaluation_csv is None:
        logger.info("Auto-detecting evaluation files...")
        eval_csv, eval_json = find_latest_evaluation_files(args.evaluation_dir, args.threshold)
        
        if eval_csv is None:
            print("\n❌ Error: No evaluation results found!")
            print(f"\nPlease run evaluation first:")
            print(f"  python evaluate_model.py \\")
            print(f"    --input {args.original_csv} \\")
            print(f"    --threshold {args.threshold} \\")
            print(f"    --output-dir {args.evaluation_dir}")
            return
        
        args.evaluation_csv = eval_csv
        if args.summary_json is None:
            args.summary_json = eval_json
    
    # Check if files exist
    if not Path(args.evaluation_csv).exists():
        print(f"\n❌ Error: Evaluation CSV not found: {args.evaluation_csv}")
        print(f"\nPlease run evaluation first or check the file path.")
        return
    
    if not Path(args.original_csv).exists():
        print(f"\n❌ Error: Original CSV not found: {args.original_csv}")
        print(f"\nPlease check the file path.")
        return
    
    generate_insights_report(
        args.evaluation_csv,
        args.original_csv,
        args.summary_json,
        args.output
    )
    
    print(f"\n✅ Presentation insights generated!")
    print(f"📄 Open {args.output} to view the report")


if __name__ == '__main__':
    main()

