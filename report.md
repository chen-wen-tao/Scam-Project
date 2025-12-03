# Job Scam Detection Project - Progress Report

## Report Visualization
![](images/scam_report_0.png)

---

## What Has Been Completed

### 1. **Data Preparation & Merging**
- Created `data_merge.py` to merge multiple data sources (`data-ashley.csv`, `data-jessica.csv`)
- Implemented duplicate removal based on Complaint ID
- Extracted only essential columns: `Complaint ID` and `Consumer complaint narrative`
- Result: 108 unique complaints from merged dataset

### 2. **Prompt Engineering for Job Scams**
- Developed job-scam-specific prompt template based on research frameworks (Ravenelle et al., 2022; FTC guidelines)
- Created comprehensive classification framework in JSON (`scam_classification_framework.json`)
- Implemented RAG-style approach: moved classification details to external JSON file
- Optimized prompt length (~80% reduction) for faster LLM responses
- Categories include:
  - Financial Mark Scams
  - Scapegoat Scams
  - Pyramid/MLM Schemes
  - Fake Placement/Staffing Scams
  - Corporate Identity Theft

### 3. **Error Handling & Robustness**
- Improved JSON parsing with fallback mechanisms
- Fixed DataFrame serialization issues (dict as key errors)
- Enhanced error handling for malformed LLM responses
- Proper handling of both dict and string representations of analysis data

### 4. **Concurrent run**
- add worker. For satisfying gemini rate limiting, default set to be 2.

---

## Future Work & Planned Enhancements

### 1. **PDF Report Generation** (High Priority)
**Goal**: Generate professional PDF reports instead of console output

**Implementation Approach**:
- Use libraries like `reportlab`, `weasyprint`, or `matplotlib` + `fpdf`
- Create template-based PDF generation
- Include:
  - Executive summary with key statistics
  - Risk distribution charts
  - Top red flags visualization
  - Scam type breakdown
  - Detailed complaint analysis
  - Recommendations section

**Files to Create/Modify**:
- `scam_detector/pdf_generator.py` - New module for PDF generation
- Update `report_generator.py` to support PDF output format
- Add `--output-format pdf` option to `main.py`

### 2. **Prompt Validation with Recent CFPB Data** (In Progress)

**Current Validation Dataset**:
- **File**: `data/complaints-2025-11-03-12-03.csv`
- **Size**: 28 complaints (latest one month from CFPB)
- **Preliminary Review**: Approximately 1-2 job scams, rest are other scam types or non-scam complaints

**Step 1: Create Labels File**

Create a new file `data/labels.csv` with the following columns:

| Column Name | Description | Possible Values |
|------------|-------------|----------------|
| `Complaint ID` | The complaint ID from original CSV | e.g., 17186864 |
| `True_Label` | Binary label for job scam detection | `1` = job scam, `0` = not job scam |
| `Category` | Detailed category classification | See values below |
| `Notes` | Optional notes about the complaint | Any text |

**Category Values** (use exactly these):
- `scam_job`: Job-related scams (fake job offers, employment scams)
- `scam_other`: Other types of scams (romance, investment, wire fraud, etc.)
- `not_scam_irrelevant`: Legitimate complaints unrelated to employment (banking issues, etc.)
- `not_scam_job_relevant`: Legitimate job-related complaints (wage disputes, discrimination, etc.)

**Example `data/labels.csv` structure**:
```csv
Complaint ID,True_Label,Category,Notes
17186864,0,scam_other,Home repair scam - not job related
16108072,0,scam_other,Car purchase wire fraud
15792287,0,scam_other,Puppy scam via payment app
17224078,0,scam_other,Bank impersonation scam
...
```

**Step 2: Labeling Process**
1. Read each complaint narrative from `complaints-2025-11-03-12-03.csv`
2. Determine the category based on the complaint content
3. Set `True_Label` = `1` only if it's a job scam (`scam_job`), otherwise `0`
4. Fill in the `Category` column with one of the four values above
5. Add any helpful notes in the `Notes` column

**Step 3: Run Detection**
```bash
python main.py --input "data/complaints-2025-11-03-12-03.csv"
```

**Step 4: Compare Predictions vs. Labels**
- Load predictions from `detect_res/scam_analysis_results_*.csv`
- Load ground truth from `data/labels.csv`
- Compare `risk_score` (prediction) vs `True_Label` (ground truth)
- Identify false positives (predicted job scam but not) and false negatives (missed job scams)

**Step 5: Analysis**
- Calculate basic accuracy: How many were correctly classified?
- For job scams (True_Label=1): Did the model identify them? (Check risk_score >= 70)
- For non-job scams (True_Label=0): Did the model correctly give low scores?
- Document specific cases where the model failed

**Limitations**:
- Small sample size (28) - focus on qualitative insights, not statistical significance
- Imbalanced (1-2 job scams) - cannot reliably measure recall
- Purpose: Identify prompt weaknesses and make targeted improvements

### 3. **Expanded Data Categories & F1 Analysis** (Future Work)
**Goal**: Build comprehensive evaluation dataset and measure model performance

**Data Categories to Add**:
1. **Scam (but not job scam)**: Other types of scams (e.g., romance scams, investment scams)
2. **Not scam (irrelevant to job)**: Legitimate complaints unrelated to employment
3. **Not scam (job relevant)**: Legitimate job-related complaints (e.g., wage disputes, discrimination)

**F1 Analysis Implementation** (For future larger dataset):

**Step 1: Data Labeling**
- Manually label a subset of complaints with ground truth:
  - `scam_job` (1) vs `not_scam_job` (0) for binary classification
  - Multi-class: `scam_job`, `scam_other`, `not_scam_irrelevant`, `not_scam_job_relevant`
- Create `data/labels.csv` with columns: `Complaint ID`, `True Label`, `Category`

**Step 2: Prediction Collection**
- Run detection on labeled dataset
- Save predictions: `Complaint ID`, `Predicted Risk Score`, `Predicted Category`

**Step 3: Threshold Optimization**
- Convert risk scores to binary predictions using threshold (e.g., ≥70 = scam)
- Test multiple thresholds (50, 60, 70, 80, 90) to find optimal F1 score
- Calculate metrics:
  - **Precision**: TP / (TP + FP)
  - **Recall**: TP / (TP + FN)
  - **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
  - **Accuracy**: (TP + TN) / (TP + TN + FP + FN)

**Step 4: Multi-class Evaluation** (if using expanded categories)
- Confusion matrix for all categories
- Per-class precision, recall, F1
- Macro-averaged and weighted F1 scores

**Implementation Files**:
- `evaluate_model.py` - New script for F1 analysis
- `data/labels.csv` - Ground truth labels
- `scam_detector/metrics.py` - Metrics calculation utilities

**Example Code Structure**:
```python
# evaluate_model.py
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

# Load ground truth and predictions
labels = pd.read_csv('data/labels.csv')
predictions = pd.read_csv('detect_res/predictions.csv')

# Binary classification (scam vs not scam)
y_true = labels['is_scam']
y_pred = (predictions['risk_score'] >= threshold).astype(int)

# Calculate metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Multi-class evaluation
# ... (similar approach for expanded categories)
```

### 4. **Additional Future Enhancements**

**Chain-of-Thought (CoT) Reasoning**:
- Enhance prompts to include explicit reasoning steps
- Improve accuracy by making LLM reasoning process transparent
- Better handling of edge cases

**LIME Integration**:
- Implement Local Interpretable Model-Agnostic Explanations
- Identify which words/phrases trigger scam classifications
- Reduce "black box" effect

**RAG System**:
- Build vector database from LLM reasoning traces
- Enable retrieval of similar past cases during detection
- Dynamic knowledge base updates

**Model Comparison**:
- Compare detection accuracy across different LLMs (Gemini, GPT-4, Claude)
- Benchmark performance metrics

---

## Current Validation Workflow (28 Sample Dataset)

**Timeline**: 1 day quick validation

**Step 1: Create Labels File** (30-60 minutes)
- Create `data/labels.csv` with columns: `Complaint ID`, `True_Label`, `Category`, `Notes`
- Manually review all 28 complaints and assign labels
- Use categories: `scam_job`, `scam_other`, `not_scam_irrelevant`, `not_scam_job_relevant`
- Set `True_Label` = 1 for job scams, 0 for everything else

**Step 2: Run Detection** (30-60 minutes)
```bash
python main.py --input "data/complaints-2025-11-03-12-03.csv"
```
- Results saved to `detect_res/scam_analysis_results_*.csv`

**Step 3: Compare & Analyze** (1-2 hours)
- Load predictions and labels
- Identify false positives (high risk_score but True_Label=0)
- Identify false negatives (low risk_score but True_Label=1)
- Calculate basic accuracy: (correct predictions) / 28

**Step 4: Prompt Refinement** (1-2 hours)
- Review error cases
- Identify common patterns in misclassifications
- Update prompt in `scam_detector/prompt_scam_analysis.py`
- Re-test if time permits

## Recommended F1 Analysis Workflow (For Future Larger Dataset)

1. **Data Collection Phase** (1-2 weeks)
   - Label 200-500 complaints across all categories
   - Ensure balanced distribution if possible
   - Create `data/labels.csv` with ground truth

2. **Baseline Evaluation** (1 week)
   - Run current model on labeled dataset
   - Calculate baseline F1 score
   - Identify common failure cases

3. **Threshold Optimization** (3-5 days)
   - Test different risk score thresholds
   - Find optimal threshold for F1 score
   - Document precision-recall trade-offs

4. **Multi-class Evaluation** (1 week)
   - If implementing expanded categories
   - Calculate per-class and macro-averaged metrics
   - Analyze confusion matrix

5. **Iterative Improvement** (Ongoing)
   - Use F1 results to refine prompts
   - Test improvements on validation set
   - Repeat until satisfactory performance

---

## Key Questions to Answer (28-Sample Validation)

1. **Does the prompt correctly identify the 1-2 job scams?**
   - Check if risk_score >= 70 for job scam complaints
   - If missed, why? What keywords/patterns were missing?

2. **Does the prompt avoid false positives?**
   - Check if other scams (romance, investment, wire fraud) get low scores
   - If they get high scores, the prompt may be too broad

3. **Can the prompt distinguish job scams from legitimate job complaints?**
   - If you have any legitimate job-related complaints, do they get low scores?
   - This tests the prompt's specificity

4. **What patterns cause misclassification?**
   - Document specific phrases or scenarios that confuse the model
   - Use these to refine the prompt

**Expected Outcome**: Identify 2-3 specific prompt improvements that address the most common error patterns

## Suggestions for Future F1 Analysis (Larger Dataset)

1. **Start with Binary Classification**: Begin with simple scam vs. not-scam before expanding to multi-class
2. **Use Stratified Sampling**: Ensure balanced representation of all categories in test set
3. **Cross-Validation**: Use k-fold cross-validation for more robust metrics
4. **Error Analysis**: Deep dive into false positives/negatives to improve prompts
5. **Baseline Comparison**: Compare against simple keyword-based or rule-based baselines

---

## Notes

- Current system optimized for speed while maintaining quality
- Modular architecture makes future enhancements straightforward
- PDF generation and F1 analysis are natural next steps for production readiness
