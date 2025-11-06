# Job Scam Detection System

A comprehensive AI-powered system for detecting job scams using Google Gemini LLM. This system analyzes consumer complaints and job-related messages to identify potential scam patterns and provide risk assessments.

## Read TO_DO.md for the unsolved tasks.

## Features

- **AI-Powered Analysis**: Uses Google Gemini LLM to analyze text for scam indicators
- **Prompt Engineering**: Sophisticated prompt design for accurate scam detection
- **Risk Scoring**: Provides 0-100 risk scores based on Gemini's scam probability
- **Comprehensive Reporting**: Generates detailed analysis reports with insights
- **Visualization**: Creates charts and graphs to visualize scam patterns
- **Batch Processing**: Analyzes large datasets of complaints efficiently

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
   pip install -r requirements.txt
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

Run the main analysis script on your dataset:

```bash
python main.py --input "data/data-ashley(Sheet1).csv"
```

**With custom options:**
```bash
# Custom output directory
python main.py --input "data/data-ashley(Sheet1).csv" --output-dir detect_res

# Custom filenames
python main.py --input "data/data-ashley(Sheet1).csv" --results-file ashley_results.csv --report-file ashley_report.json
```

This will:
- Analyze all complaints in the CSV file using Gemini AI
- Generate risk scores (0-100) based on scam probability
- Save results to `detect_res/scam_analysis_results.csv`
- Create a summary report in `detect_res/scam_analysis_report.json`

**Note**: All analysis results are saved in the `detect_res/` folder by default.

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

## Risk Scoring

The risk score is directly based on Gemini's `scam_probability` (0-100), providing a clear assessment:

- **0-30**: Low risk (likely legitimate)
- **31-60**: Medium risk (suspicious, needs review)
- **61-80**: High risk (likely scam)
- **81-100**: Very high risk (definite scam)

The score reflects Gemini's confidence in identifying scams, making it a reliable indicator of risk level.

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

## API Usage

The system uses Google Gemini API. Monitor your usage:
- Free tier: 15 requests per minute
- Paid tier: Higher limits available

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

# The system will automatically try different model names:
# - gemini-1.5-pro (preferred)
# - gemini-1.5-flash (fallback)
# - gemini-pro (legacy)
```

### Performance Tips

- Use smaller batches for very large datasets
- Monitor API usage to avoid rate limits
- Consider caching results for repeated analysis

## Project Structure

The project has been refactored into a modular structure:

```
scam_detector/          # Main package
├── __init__.py        # Package initialization
├── config.py          # Configuration and constants
├── detector.py        # Main detector class
├── text_processor.py  # Text preprocessing and indicators
├── gemini_client.py   # Gemini API wrapper
├── prompt_scam_analysis.py  # AI prompt templates
├── report_generator.py # Report generation
└── file_handler.py    # File I/O operations

detect_res/            # Output directory for analysis results
visualizations/        # Generated visualization charts
data/                  # Input data files
main.py               # New entry point (CLI)
job_scam_detector.py  # Legacy wrapper (backward compatible)
visualize_results.py  # Visualization script
```

## Contributing

To improve the system:

1. **Enhance Prompt Engineering**: Improve the Gemini prompt in `scam_detector/prompt_scam_analysis.py` for better detection accuracy
2. **Add Visualizations**: Create new visualization types in `visualize_results.py`
3. **Improve Text Processing**: Enhance preprocessing in `scam_detector/text_processor.py`
4. **Expand Analysis**: Add new analysis dimensions in `scam_detector/report_generator.py`

The system is designed to rely on AI analysis rather than rule-based heuristics, so focus improvements on prompt engineering and model optimization.

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
