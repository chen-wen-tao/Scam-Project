#!/usr/bin/env python3
"""
Evaluation script for Job Scam Detection System
Calculates F1 score, precision, recall, and confusion matrix
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse
import logging
import json
import tempfile
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score, 
    precision_score, 
    recall_score,
    accuracy_score
)

from scam_detector import JobScamDetector
from scam_detector.config import DEFAULT_WORKERS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def map_prediction_to_category(risk_score: float, gemini_analysis: Dict, threshold: float = 70.0) -> str:
    """
    Map prediction (risk_score + gemini_analysis) to category
    
    Args:
        risk_score: Scam probability (0-100)
        gemini_analysis: Full Gemini analysis dict
        threshold: Threshold for binary classification
        
    Returns:
        Category: 'scam_job', 'scam_other', 'not_scam_irrelevant', 'not_scam_job_relevant'
    """
    # NEW: If using multi_category prompt, check for explicit category field first
    explicit_category = gemini_analysis.get('category', '').lower()
    if explicit_category in ['scam_job', 'scam_other', 'not_scam_job_relevant', 'not_scam_irrelevant']:
        return explicit_category
    
    # LEGACY: Fallback logic for job_scam prompt mode
    # Check if it's a scam (risk_score >= threshold)
    is_scam = risk_score >= threshold
    
    if not is_scam:
        # Not a scam - need to determine if job-relevant
        is_job_related = gemini_analysis.get('is_job_related', None)
        if is_job_related is not None:
            return 'not_scam_job_relevant' if is_job_related else 'not_scam_irrelevant'
        
        # Fallback: check scam_type for job-related terms
        scam_type = gemini_analysis.get('scam_type', {})
        if isinstance(scam_type, dict):
            primary = scam_type.get('primary_category', '').lower()
            subcategory = scam_type.get('subcategory', '').lower()
            job_keywords = ['job', 'employment', 'work', 'hire', 'position', 'career', 'recruit']
            text_combined = f"{primary} {subcategory}".lower()
            if any(keyword in text_combined for keyword in job_keywords):
                return 'not_scam_job_relevant'
        
        return 'not_scam_irrelevant'
    
    # It's a scam - determine if job-related
    # First check explicit is_job_related field
    is_job_related = gemini_analysis.get('is_job_related', None)
    if is_job_related is not None:
        return 'scam_job' if is_job_related else 'scam_other'
    
    # Fallback: check scam_type for job-related indicators
    scam_type = gemini_analysis.get('scam_type', {})
    if isinstance(scam_type, dict):
        primary = scam_type.get('primary_category', '').lower()
        subcategory = scam_type.get('subcategory', '').lower()
        
        # Check if it's job-related based on framework categories
        job_related_categories = [
            'financial mark scams',
            'scapegoat scams', 
            'pyramid/mlm schemes',
            'fake placement/staffing scams',
            'corporate identity theft'
        ]
        
        # Also check subcategory for job-related terms
        job_keywords = ['job', 'employment', 'work', 'hire', 'position', 'career', 'recruit', 'remote work']
        text_combined = f"{primary} {subcategory}".lower()
        
        # Check if primary category matches job scam categories
        if any(cat in primary for cat in job_related_categories):
            return 'scam_job'
        
        # Check if subcategory contains job-related keywords
        if any(keyword in text_combined for keyword in job_keywords):
            return 'scam_job'
        
        # Check for non-job scam indicators
        non_job_indicators = ['wire fraud', 'romance scam', 'business fraud', 'service scam', 'n/a', 'not a scam']
        if any(indicator in text_combined for indicator in non_job_indicators):
            return 'scam_other'
    
    # Default: assume non-job scam if unclear
    return 'scam_other'


def evaluate_model(
    csv_path: str,
    threshold: float = 70.0,
    workers: int = DEFAULT_WORKERS,
    api_key: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Evaluate the model on labeled data
    
    Args:
        csv_path: Path to CSV with True_Label and Category columns
        threshold: Risk score threshold for binary classification
        workers: Number of parallel workers
        api_key: Optional Gemini API key
        output_dir: Optional output directory
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Loading labeled data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['Consumer complaint narrative', 'True_Label', 'Category']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter out rows with missing narratives
    df = df.dropna(subset=['Consumer complaint narrative'])
    logger.info(f"Evaluating on {len(df)} complaints")
    
    # Initialize detector
    from pathlib import Path
    output_path = Path(output_dir) if output_dir else None
    detector = JobScamDetector(api_key=api_key, output_dir=output_path)
    
    # Run predictions using batch processing for efficiency
    logger.info("Running predictions...")
    
    # Prepare data for batch processing
    if 'Complaint ID' in df.columns:
        df['Complaint ID'] = df['Complaint ID'].astype(str)
    else:
        df['Complaint ID'] = df.index.astype(str)
    
    # Create temporary CSV for batch processing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        temp_csv = tmp_file.name
        df[['Complaint ID', 'Consumer complaint narrative']].to_csv(temp_csv, index=False)
    
    try:
        # Use batch processing with multi_category prompt mode
        results_df = detector.analyze_dataset(temp_csv, workers=workers, prompt_mode='multi_category')
        
        # The results_df uses 'complaint_id' (lowercase with underscore)
        # We need to normalize column names for merging
        if 'complaint_id' in results_df.columns:
            results_df['Complaint ID'] = results_df['complaint_id'].astype(str)
        elif 'Complaint ID' not in results_df.columns:
            # If neither exists, create from index
            results_df['Complaint ID'] = results_df.index.astype(str)
        
        # Ensure Complaint ID exists in df
        if 'Complaint ID' not in df.columns:
            df['Complaint ID'] = df.index.astype(str)
        
        # Merge predictions with true labels
        results_df['Complaint ID'] = results_df['Complaint ID'].astype(str)
        df['Complaint ID'] = df['Complaint ID'].astype(str)
        merged = results_df.merge(
            df[['Complaint ID', 'True_Label', 'Category']], 
            on='Complaint ID', 
            how='left'
        )
        
        # Extract predictions
        predictions = []
        for _, row in merged.iterrows():
            risk_score = row.get('risk_score', 0) or 0
            if pd.isna(risk_score):
                risk_score = 0
            
            # Parse gemini_analysis if it's a string
            gemini_analysis = row.get('gemini_analysis', {})
            if isinstance(gemini_analysis, str):
                try:
                    gemini_analysis = json.loads(gemini_analysis)
                except:
                    gemini_analysis = {}
            if not isinstance(gemini_analysis, dict):
                gemini_analysis = {}
            
            # Map to category
            predicted_category = map_prediction_to_category(float(risk_score), gemini_analysis, threshold)
            
            complaint_id_val = row.get('Complaint ID', 'unknown')
            complaint_id = str(complaint_id_val) if complaint_id_val is not None and pd.notna(complaint_id_val) else 'unknown'
            
            true_label_val = row.get('True_Label', 0)
            true_label = int(true_label_val) if true_label_val is not None and pd.notna(true_label_val) else 0
            
            true_category_val = row.get('Category', 'not_scam_irrelevant')
            true_category = str(true_category_val) if true_category_val is not None and pd.notna(true_category_val) else 'not_scam_irrelevant'
            
            predictions.append({
                'complaint_id': complaint_id,
                'true_label': true_label,
                'true_category': true_category,
                'predicted_risk_score': float(risk_score),
                'predicted_category': predicted_category,
                'is_scam_predicted': float(risk_score) >= threshold
            })
            
            logger.info(f"Complaint {row['Complaint ID']}: Risk={risk_score:.1f}%, "
                       f"True={row.get('Category', 'N/A')}, Predicted={predicted_category}")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_csv):
            os.unlink(temp_csv)
    
    pred_df = pd.DataFrame(predictions)
    
    # Calculate metrics
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    
    # Binary classification (scam vs not_scam)
    y_true_binary = pred_df['true_label'].values
    y_pred_binary = pred_df['is_scam_predicted'].astype(int).values
    
    binary_metrics = {
        'accuracy': accuracy_score(y_true_binary, y_pred_binary),
        'precision': precision_score(y_true_binary, y_pred_binary, zero_division='warn'),
        'recall': recall_score(y_true_binary, y_pred_binary, zero_division='warn'),
        'f1': f1_score(y_true_binary, y_pred_binary, zero_division='warn')
    }
    
    logger.info("\n--- Binary Classification (Scam vs Not Scam) ---")
    logger.info(f"Threshold: {threshold}")
    logger.info(f"Accuracy:  {binary_metrics['accuracy']:.3f}")
    logger.info(f"Precision: {binary_metrics['precision']:.3f}")
    logger.info(f"Recall:    {binary_metrics['recall']:.3f}")
    logger.info(f"F1 Score:  {binary_metrics['f1']:.3f}")
    
    # Confusion matrix for binary
    cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
    logger.info("\nConfusion Matrix (Binary):")
    logger.info("                Predicted")
    logger.info("              Not Scam  Scam")
    logger.info(f"Actual Not Scam  {cm_binary[0,0]:4d}    {cm_binary[0,1]:4d}")
    logger.info(f"Actual Scam      {cm_binary[1,0]:4d}    {cm_binary[1,1]:4d}")
    
    # Multi-class classification (4 categories)
    categories = ['scam_job', 'scam_other', 'not_scam_irrelevant', 'not_scam_job_relevant']
    y_true_cat = pred_df['true_category'].values
    y_pred_cat = pred_df['predicted_category'].values
    
    # Filter to only include categories present in both true and predicted
    present_categories = sorted(set(y_true_cat) | set(y_pred_cat))
    
    logger.info("\n--- Multi-Class Classification (4 Categories) ---")
    logger.info("\nClassification Report:")
    logger.info(classification_report(
        y_true_cat, 
        y_pred_cat, 
        labels=present_categories,
        zero_division='warn'
    ))
    
    # Per-category F1 scores
    category_f1 = {}
    for cat in present_categories:
        y_true_cat_binary = (y_true_cat == cat).astype(int)
        y_pred_cat_binary = (y_pred_cat == cat).astype(int)
        
        if y_true_cat_binary.sum() > 0 or y_pred_cat_binary.sum() > 0:
            f1 = f1_score(y_true_cat_binary, y_pred_cat_binary, zero_division='warn')
            precision = precision_score(y_true_cat_binary, y_pred_cat_binary, zero_division='warn')
            recall = recall_score(y_true_cat_binary, y_pred_cat_binary, zero_division='warn')
            
            category_f1[cat] = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'support': int(y_true_cat_binary.sum())
            }
    
    logger.info("\n--- Per-Category Metrics ---")
    for cat, metrics in sorted(category_f1.items()):
        logger.info(f"\n{cat}:")
        logger.info(f"  Precision: {metrics['precision']:.3f}")
        logger.info(f"  Recall:    {metrics['recall']:.3f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.3f}")
        logger.info(f"  Support:   {metrics['support']}")
    
    # Confusion matrix for multi-class
    cm_multi = confusion_matrix(y_true_cat, y_pred_cat, labels=present_categories)
    logger.info("\nConfusion Matrix (Multi-Class):")
    logger.info("Categories: " + ", ".join(present_categories))
    logger.info("\n" + str(cm_multi))
    
    # Save detailed results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        results_file = output_path / f"evaluation_results_threshold_{threshold}.csv"
        pred_df.to_csv(results_file, index=False)
        logger.info(f"\nDetailed results saved to: {results_file}")
        
        # Save summary metrics
        summary = {
            'threshold': threshold,
            'binary_metrics': binary_metrics,
            'category_metrics': category_f1,
            'confusion_matrix_binary': cm_binary.tolist(),
            'confusion_matrix_multi': cm_multi.tolist(),
            'categories': present_categories
        }
        
        summary_file = output_path / f"evaluation_summary_threshold_{threshold}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary metrics saved to: {summary_file}")
    
    return {
        'binary_metrics': binary_metrics,
        'category_metrics': category_f1,
        'predictions': pred_df,
        'confusion_matrix_binary': cm_binary,
        'confusion_matrix_multi': cm_multi
    }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Evaluate Job Scam Detection System on labeled data'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to CSV file with True_Label and Category columns'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=70.0,
        help='Risk score threshold for binary classification (default: 70.0)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=DEFAULT_WORKERS,
        help=f'Number of parallel workers (default: {DEFAULT_WORKERS})'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Google Gemini API key (or set GEMINI_API_KEY env var)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Output directory for evaluation results (default: evaluation_results/)'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_model(
        csv_path=args.input,
        threshold=args.threshold,
        workers=args.workers,
        api_key=args.api_key,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

