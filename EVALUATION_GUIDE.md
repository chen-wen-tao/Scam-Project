# Evaluation Guide - F1 Testing

This guide explains how to verify your system and conduct F1 testing on labeled data.

## Prerequisites

1. **Labeled CSV file** with columns:
   - `Consumer complaint narrative` - The complaint text
   - `True_Label` - Binary label (0 = not scam, 1 = scam)
   - `Category` - Category label (`scam_job`, `scam_other`, `not_scam_irrelevant`, `not_scam_job_relevant`)
   - `Complaint ID` - Optional, but recommended for tracking

2. **Python packages** (already in requirements.txt):
   - `pandas`
   - `scikit-learn` (for metrics)
   - `numpy`

## Quick Start

### 1. Basic Evaluation (Single Threshold)

Run evaluation with default threshold (70):

```bash
python evaluate_model.py \
    --input data/cfpb-complaints-2025-11-03-12-03.csv \
    --threshold 70 \
    --workers 2 \
    --output-dir evaluation_results
```

**Output:**
- Console: Metrics, confusion matrices, classification report
- CSV: `evaluation_results/evaluation_results_threshold_70.csv` (detailed predictions)
- JSON: `evaluation_results/evaluation_summary_threshold_70.json` (summary metrics)

### 2. Test Multiple Thresholds

Find the optimal threshold by testing multiple values:

```bash
python test_thresholds.py \
    --input data/cfpb-complaints-2025-11-03-12-03.csv \
    --thresholds 50 60 70 80 90 \
    --workers 2
```

This will:
- Test each threshold
- Compare F1 scores
- Report the best threshold

## Understanding the Output

### Binary Classification Metrics

- **Accuracy**: Overall correctness
- **Precision**: Of predicted scams, how many are actually scams?
- **Recall**: Of actual scams, how many did we catch?
- **F1 Score**: Harmonic mean of precision and recall

### Multi-Class Classification

The system predicts one of 4 categories:
- `scam_job` - Job-related scams
- `scam_other` - Other scams (not job-related)
- `not_scam_irrelevant` - Not a scam, not job-related
- `not_scam_job_relevant` - Not a scam, but job-related

### Confusion Matrix

Shows:
- **True Positives (TP)**: Correctly identified scams
- **False Positives (FP)**: Non-scams incorrectly flagged as scams
- **False Negatives (FN)**: Scams missed!
- **True Negatives (TN)**: Correctly identified non-scams

## Category Mapping Logic

The system maps predictions to categories using:

1. **Risk Score Threshold**: If `risk_score >= threshold` → scam, else → not scam
2. **Job Relevance**: Checks Gemini's `scam_type` for job-related keywords:
   - Job-related categories from framework
   - Keywords: "job", "employment", "work", "hire", etc.

## Example Output

```
--- Binary Classification (Scam vs Not Scam) ---
Threshold: 70.0
Accuracy:  0.857
Precision: 0.900
Recall:    0.750
F1 Score:  0.818

Confusion Matrix (Binary):
                Predicted
              Not Scam  Scam
Actual Not Scam    12      2
Actual Scam          3      9

--- Multi-Class Classification (4 Categories) ---
Classification Report:
              precision    recall  f1-score   support

scam_job           0.85      0.80      0.82        10
scam_other         0.90      0.95      0.92        20
not_scam_irrelevant 0.88      0.85      0.86        13
not_scam_job_relevant 0.75      0.60      0.67         5
```

## Tips

1. **Start with default threshold (70)**: Good balance for most cases
2. **Adjust threshold based on use case**:
   - Lower threshold (50-60): Catch more scams (higher recall, more false positives)
   - Higher threshold (80-90): Fewer false positives (higher precision, may miss some scams)
3. **Review false positives/negatives**: Check `evaluation_results_threshold_*.csv` to see which complaints were misclassified
4. **Use parallel workers**: Set `--workers 2` or `3` for faster processing (watch rate limits!)

## Troubleshooting

**Issue**: "Missing required columns"
- **Solution**: Ensure CSV has `Consumer complaint narrative`, `True_Label`, and `Category` columns

**Issue**: Rate limit errors
- **Solution**: Reduce `--workers` to 1, or wait between runs

**Issue**: Low F1 score
- **Solution**: 
  - Review misclassified cases in the results CSV
  - Adjust threshold using `test_thresholds.py`
  - Check if labels are correct
  - Consider prompt refinement if patterns emerge

## Next Steps

1. **Analyze errors**: Review false positives/negatives to identify patterns
2. **Refine prompts**: Update `scam_detector/prompt_scam_analysis.py` based on findings
3. **Optimize threshold**: Use `test_thresholds.py` to find optimal value
4. **Expand dataset**: Label more complaints for better statistical significance

