#!/usr/bin/env python3
"""
Test different thresholds to find optimal F1 score
"""

import argparse
from evaluate_model import evaluate_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_thresholds(csv_path: str, thresholds: list = None, workers: int = 1, api_key: str = None, prompt_mode: str = 'multi_category'):
    """
    Test multiple thresholds and find the best one
    
    Args:
        csv_path: Path to labeled CSV
        thresholds: List of thresholds to test (default: [50, 60, 70, 80, 90])
        workers: Number of parallel workers
        api_key: Optional API key
    """
    if thresholds is None:
        thresholds = [50, 60, 70, 80, 90]
    
    logger.info(f"Testing thresholds: {thresholds}")
    logger.info("="*60)
    
    best_f1 = 0
    best_threshold = None
    results = []
    
    for threshold in thresholds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing threshold: {threshold}")
        logger.info(f"{'='*60}\n")
        
        try:
            # Note: evaluate_model uses multi_category prompt mode by default
            # But we can't pass prompt_mode directly, so it's hardcoded in evaluate_model
            metrics = evaluate_model(
                csv_path=csv_path,
                threshold=threshold,
                workers=workers,
                api_key=api_key,
                output_dir=None  # Don't save intermediate results
            )
            
            f1 = metrics['binary_metrics']['f1']
            precision = metrics['binary_metrics']['precision']
            recall = metrics['binary_metrics']['recall']
            accuracy = metrics['binary_metrics']['accuracy']
            
            results.append({
                'threshold': threshold,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
            
            logger.info(f"\nThreshold {threshold}: F1={f1:.3f}, Precision={precision:.3f}, "
                       f"Recall={recall:.3f}, Accuracy={accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error testing threshold {threshold}: {e}")
            continue
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("THRESHOLD COMPARISON SUMMARY")
    logger.info("="*60)
    logger.info(f"{'Threshold':<12} {'F1':<8} {'Precision':<12} {'Recall':<12} {'Accuracy':<12}")
    logger.info("-"*60)
    
    for r in results:
        logger.info(f"{r['threshold']:<12.1f} {r['f1']:<8.3f} {r['precision']:<12.3f} "
                   f"{r['recall']:<12.3f} {r['accuracy']:<12.3f}")
    
    logger.info("\n" + "="*60)
    logger.info(f"Best threshold: {best_threshold} (F1={best_f1:.3f})")
    logger.info("="*60)
    
    return best_threshold, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test different thresholds for optimal F1')
    parser.add_argument('--input', type=str, required=True, help='Path to labeled CSV')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[50, 60, 70, 80, 90],
                       help='Thresholds to test (default: 50 60 70 80 90)')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--api-key', type=str, default=None, help='Gemini API key')
    
    args = parser.parse_args()
    
    test_thresholds(args.input, args.thresholds, args.workers, args.api_key)

