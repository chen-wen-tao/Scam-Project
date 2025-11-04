# Job Scam Detection System

A comprehensive AI-powered system for detecting job scams using Google Gemini LLM. This system analyzes consumer complaints and job-related messages to identify potential scam patterns and provide risk assessments.

## Read TO_DO.md for the unsolved tasks.

## Features

- **AI-Powered Analysis**: Uses Google Gemini to analyze text for scam indicators
- **Rule-Based Detection**: Implements pattern matching for common scam red flags
- **Risk Scoring**: Provides 0-100 risk scores for each complaint
- **Comprehensive Reporting**: Generates detailed analysis reports
- **Visualization**: Creates charts and graphs to visualize scam patterns
- **Batch Processing**: Analyzes large datasets of complaints

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get Google Gemini API Key:**
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
python job_scam_detector.py
```

This will:
- Analyze all complaints in the CSV file
- Generate risk scores and scam classifications
- Save results to `scam_analysis_results.csv`
- Create a summary report in `scam_analysis_report.json`

### Visualization

Generate visualizations of the analysis results:

```bash
python visualize_results.py
```

This creates charts showing:
- Risk score distributions
- Red flag patterns
- Scam type breakdowns
- Correlation analysis

### Custom Analysis

You can also use the detector programmatically:

```python
from job_scam_detector import JobScamDetector

# Initialize detector
detector = JobScamDetector(api_key="your_api_key")

# Analyze a single complaint
result = detector.analyze_complaint("Your complaint text here")
print(f"Risk Score: {result['overall_risk_score']}")

# Analyze a dataset
results = detector.analyze_dataset("your_data.csv", "output.csv")
```

## Data Format

The system expects CSV files with the following columns:
- `Consumer complaint narrative`: The main complaint text
- `Complaint ID`: (Optional) Unique identifier for each complaint

## Scam Detection Features

### Rule-Based Indicators

The system looks for these red flags:

**Financial Red Flags:**
- Fake check mentions
- Money transfer requests
- Equipment purchase requirements
- Advance payment demands

**Urgency Indicators:**
- Immediate action required
- Time pressure tactics
- Deadline threats

**Communication Red Flags:**
- Email-only communication
- No phone interviews
- Messaging app usage

**Job Red Flags:**
- No experience required
- Unrealistic promises
- Vague job descriptions

### AI Analysis

Gemini AI provides:
- Scam probability (0-100)
- Financial risk assessment
- Scam type classification
- Victim vulnerability analysis
- Prevention recommendations

## Output Files

- `scam_analysis_results.csv`: Detailed analysis of each complaint
- `scam_analysis_report.json`: Summary statistics and insights
- `visualizations/`: Charts and graphs (if visualization script is run)

## Risk Scoring

The system combines rule-based and AI analysis to generate risk scores:

- **0-30**: Low risk (likely legitimate)
- **31-60**: Medium risk (suspicious, needs review)
- **61-80**: High risk (likely scam)
- **81-100**: Very high risk (definite scam)

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

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `GEMINI_API_KEY` environment variable is set
2. **Model Not Found Error**: Run `python check_models.py` to see available models
3. **CSV Format Error**: Check that your CSV has the required columns
4. **Memory Issues**: For large datasets, consider processing in batches

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

## Contributing

To improve the system:

1. Add new scam indicators to `scam_indicators` dictionary
2. Enhance the Gemini prompt for better analysis
3. Add new visualization types
4. Implement additional data preprocessing

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
