# Job Scam Detection System

A comprehensive AI-powered system for detecting job scams using Google Gemini LLM. This system analyzes consumer complaints and job-related messages to identify potential scam patterns and provide risk assessments.

## Read TO_DO.md for the unsolved tasks.

## Features

- **AI-Powered Analysis**: Uses Google Gemini LLM to analyze text for scam indicators
- **Dual Prompt Modes**: Job scam analysis or multi-category classification (scam_job/scam_other/not_scam_job_relevant/not_scam_irrelevant)
- **Risk Scoring**: Provides 0-100 risk scores based on Gemini's scam probability
- **Comprehensive Reporting**: Generates detailed JSON and PDF reports with insights
- **Model Evaluation**: F1 score, precision, recall, accuracy metrics with threshold optimization
- **Rate Limiting**: Automatic request throttling to stay within API limits
- **Batch Processing**: Parallel processing with configurable workers

## Installation

1. **Clone or download the project files**

2. **Create and activate a virtual environment (can skip but recommended):**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate it (macOS/Linux)
   source venv/bin/activate
   
   # Or on Windows
   venv\Scripts\activate
   ```
   
   You should see `(venv)` in your terminal prompt when activated.

3. **Install Python dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```

4. **Get Google Gemini API Key:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Set it as an environment variable:
     ```bash
     export GEMINI_API_KEY="your_api_key_here"
     ```

## Usage

### Basic Analysis

**Job Scam Analysis** (default mode):
```bash
python3 main.py --input "data/data-merged.csv" --output-format both
```

**Multi-Category Classification** (for evaluation datasets):
```bash
python3 main.py --input "data/cfpb-complaints-2025-11-03-12-03.csv" --prompt-mode multi_category --output-format both
```

**Options:**
- `--input`: Path to CSV file (required)
- `--output-format`: `json`, `pdf`, or `both` (default: `json`)
- `--prompt-mode`: `job_scam` (default) or `multi_category`
- `--workers`: Number of parallel workers (default: 1, use 2-3 for faster processing)
- `--output-dir`: Output directory (default: `detect_res/`)

**Output:**
- `scam_analysis_results.csv`: Detailed analysis per complaint
- `scam_analysis_report.json`: Summary statistics
- `scam_analysis_report.pdf`: PDF report (if `--output-format pdf` or `both`)

### Generate Visualizations

After running the analysis, generate visualizations:

```bash
python visualize_results.py
```

This creates charts in the `visualizations/` folder showing:
- Risk score distributions
- Red flag patterns (from Gemini analysis)
- Scam type breakdowns
- Risk score vs text length correlation
- Comprehensive dashboard

**Important**: You must run the analysis first (`main.py`) before generating visualizations, as the visualization script reads from the results files.

### Custom Analysis

You can also use the detector programmatically:

```python
# Recommended: Use the new modular package
from scam_detector import JobScamDetector

# Or use the legacy import (backward compatible)
# from job_scam_detector import JobScamDetector

# Initialize detector
detector = JobScamDetector(api_key="your_api_key")

# Analyze a single complaint
result = detector.analyze_complaint("Your complaint text here")
print(f"Risk Score: {result['risk_score']}")

# Analyze a dataset (results saved to detect_res/ by default)
results = detector.analyze_dataset("your_data.csv")

# Generate report (saved to detect_res/ by default)
report = detector.generate_report(results)
```

## Data Format

The system expects CSV files with the following columns:
- `Consumer complaint narrative`: The main complaint text
- `Complaint ID`: (Optional) Unique identifier for each complaint

## Scam Detection Features

### AI-Powered Analysis

The system uses Google Gemini LLM with sophisticated prompt engineering to analyze complaints. Gemini provides:

- **Scam Probability (0-100)**: Direct assessment of scam likelihood
- **Red Flags**: Contextual indicators identified by AI analysis
- **Financial Risk Assessment**: Low/Medium/High risk levels
- **Scam Type Classification**: Specific scam categories (e.g., fake check, advance fee, impersonation)
- **Victim Vulnerability Analysis**: Profile of victim characteristics
- **Prevention Recommendations**: Actionable advice for avoiding similar scams

The system relies entirely on AI analysis rather than simple keyword matching, providing more accurate and nuanced detection.

## Output Files

- `scam_analysis_results.csv`: Detailed analysis of each complaint
- `scam_analysis_report.json`: Summary statistics and insights
- `visualizations/`: Charts and graphs (if visualization script is run)

## Risk Scoring & Thresholds

The risk score is based on Gemini's `scam_probability` (0-100):

- **0-59**: Low risk (predicted as not scam)
- **60-100**: High risk (predicted as scam)

**Optimal Threshold**: 60 (F1=0.837) for multi-category classification
- Threshold can be optimized using `test_thresholds.py`
- Lower threshold = higher recall, lower precision
- Higher threshold = higher precision, lower recall

## Example Results

```json
{
  "scam_probability": 85,
  "red_flags": ["fake check", "send money", "urgent"],
  "financial_risk": "High",
  "scam_type": "Fake check scam",
  "victim_profile": "Unemployed job seeker",
  "recommendations": ["Never send money to employers", "Verify company legitimacy"]
}
```

### Model Evaluation

Evaluate model performance on labeled data:

```bash
# Test multiple thresholds to find optimal
python3 test_thresholds.py --input data/cfpb-complaints-2025-11-03-12-03.csv --thresholds 50 60 70 80 90

# Run full evaluation with optimal threshold (60)
python3 evaluate_model.py --input data/cfpb-complaints-2025-11-03-12-03.csv --threshold 60 --workers 2

# Generate presentation insights
python3 generate_presentation_insights.py --threshold 60
```

**Evaluation Metrics:**
- Binary classification: Accuracy, Precision, Recall, F1 Score
- Multi-class classification: Per-category F1 scores
- Optimal threshold: 60 (F1=0.837) for multi-category prompt

## API Usage & Rate Limiting

The system uses Google Gemini API with automatic rate limiting:
- **Model**: `gemini-2.5-flash-lite` (prioritized for speed and rate limits)
- **Rate Limits**: 15 RPM (requests per minute), automatically throttled
- **Retry Logic**: Automatic retry with exponential backoff for 429 errors
- **Free Tier**: 15 RPM limit, use `--workers 1` to stay within limits
- **Rate Limiter**: Thread-safe throttling ensures requests stay under limits

## Testing

### Quick Test

Test with the included sample data:

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Set API key
export GEMINI_API_KEY="your_api_key_here"

# Step 1: Run analysis
python main.py --input "data/data-ashley(Sheet1).csv"

# Step 2: Generate visualizations (after analysis completes)
python visualize_results.py
```

The visualizations will be saved in the `visualizations/` folder.

### Programmatic Testing

You can also test programmatically:

```python
from scam_detector import JobScamDetector
import os

# Set API key (or use environment variable)
api_key = os.getenv('GEMINI_API_KEY')

# Initialize detector
detector = JobScamDetector(api_key=api_key)

# Analyze dataset
results = detector.analyze_dataset("data/data-ashley(Sheet1).csv")

# Generate report
report = detector.generate_report(results)

# Check results location
print(f"Results saved to: {detector.file_handler.get_results_path()}")
print(f"Report saved to: {detector.file_handler.get_report_path()}")
```

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `GEMINI_API_KEY` environment variable is set
   ```bash
   echo $GEMINI_API_KEY  # Should show your key
   ```

2. **Module Not Found Error**: Make sure virtual environment is activated and dependencies are installed
   ```bash
   source venv/bin/activate  # Activate venv first
   pip install -r requirements.txt
   ```

3. **Model Not Found Error**: Run `python check_models.py` to see available models

4. **CSV Format Error**: Check that your CSV has the required columns:
   - `Consumer complaint narrative` (required)
   - `Complaint ID` (optional)

5. **File Not Found**: Check the path to your CSV file
   ```bash
   ls -la "data/data-ashley(Sheet1).csv"
   ```

6. **Memory Issues**: For large datasets, consider processing in batches

### Model Availability Issues

If you get a "model not found" error:

```bash
# Check available models
python check_models.py

# The system automatically tries models in priority order:
# - gemini-2.5-flash-lite (preferred: 15 RPM, fastest)
# - gemini-2.0-flash-lite (fallback: 30 RPM)
# - gemini-1.5-flash (legacy fallback)
```

### Performance Tips

- Use smaller batches for very large datasets
- Monitor API usage to avoid rate limits
- Consider caching results for repeated analysis

## Project Structure

The project has been refactored into a modular structure:

```
scam_detector/              # Main package
├── __init__.py            # Package initialization
├── config.py              # Configuration and constants
├── detector.py            # Main detector class
├── text_processor.py      # Text preprocessing
├── gemini_client.py       # Gemini API wrapper with rate limiting
├── prompt_scam_analysis.py      # Job scam analysis prompt
├── prompt_multi_category.py      # Multi-category classification prompt
├── report_generator.py    # JSON report generation
├── pdf_generator.py       # PDF report generation
└── file_handler.py        # File I/O operations

evaluation_results/        # Model evaluation outputs
detect_res/                # Analysis results (CSV, JSON, PDF)
visualizations/            # Generated charts
data/                      # Input data files
main.py                    # CLI entry point
evaluate_model.py          # Model evaluation script
test_thresholds.py         # Threshold optimization
generate_presentation_insights.py  # Presentation report generator
```

## Key Improvements

- **JSON RAG Framework**: Classification framework cached for 80% token reduction
- **Rate Limiting**: Automatic throttling prevents API quota errors
- **Dual Prompts**: Job scam analysis and multi-category classification
- **PDF Reports**: Professional reports with model metadata and run time
- **Model Evaluation**: Comprehensive metrics (F1, precision, recall, accuracy)
- **Threshold Optimization**: Automated testing to find optimal threshold
- **Parallel Processing**: Configurable workers for faster batch analysis

## License

This project is for educational and research purposes. Please ensure compliance with Google's API terms of service.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the error logs
3. Ensure all dependencies are installed correctly

## Future Enhancements

- Real-time scam detection API
- Integration with job boards
- Machine learning model training
- Multi-language support
- Advanced NLP preprocessing
