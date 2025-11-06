#!/usr/bin/env python3
"""
Example usage of the Job Scam Detection System
Demonstrates how to use the system for various scenarios
"""

import os
import json
from scam_detector import JobScamDetector

def example_single_analysis():
    """Example: Analyze a single complaint"""
    print("="*50)
    print("EXAMPLE 1: Single Complaint Analysis")
    print("="*50)
    
    # Initialize detector (you'll need to set your API key)
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        return
    
    detector = JobScamDetector(api_key)
    
    # Example complaint text
    complaint_text = """
    I was contacted by a company claiming to be a legitimate employer offering remote work. 
    The job involved completing tasks and reviewing materials. I performed the tasks requested 
    and communicated with the alleged representatives throughout the process. After completing 
    the work, I expected compensation as agreed, but I was never paid. They sent me a check 
    for $5000 to buy equipment, but when I deposited it, the bank said it was fake. I lost 
    $3000 in this scam.
    """
    
    # Analyze the complaint
    result = detector.analyze_complaint(complaint_text, "example_001")
    
    # Print results
    print(f"Complaint ID: {result['complaint_id']}")
    print(f"Overall Risk Score: {result['overall_risk_score']}/100")
    print(f"Text Length: {result['text_length']} characters")
    
    print("\nRule-based Indicators Found:")
    for category, indicators in result['rule_based_indicators'].items():
        print(f"  {category}: {indicators}")
    
    print("\nGemini AI Analysis:")
    gemini = result['gemini_analysis']
    print(f"  Scam Probability: {gemini.get('scam_probability', 'N/A')}%")
    print(f"  Financial Risk: {gemini.get('financial_risk', 'N/A')}")
    print(f"  Scam Type: {gemini.get('scam_type', 'N/A')}")
    print(f"  Red Flags: {gemini.get('red_flags', [])}")
    print(f"  Recommendations: {gemini.get('recommendations', [])}")

def example_batch_analysis():
    """Example: Analyze multiple complaints"""
    print("\n" + "="*50)
    print("EXAMPLE 2: Batch Analysis")
    print("="*50)
    
    # Create sample data
    sample_complaints = [
        {
            "Complaint ID": "sample_001",
            "Consumer complaint narrative": "I got a job offer via email. They sent me a check for $2000 to buy equipment. I deposited it and sent the money to their vendor. The check bounced and I lost $2000."
        },
        {
            "Complaint ID": "sample_002", 
            "Consumer complaint narrative": "I applied for a legitimate marketing position at a well-known company. The interview process was professional and I was hired based on my qualifications. I started work and received regular payments."
        },
        {
            "Complaint ID": "sample_003",
            "Consumer complaint narrative": "Someone contacted me about a work-from-home opportunity. They said I needed to send $500 upfront for training materials. I sent the money but never received any materials or work."
        }
    ]
    
    # Save sample data to CSV
    import pandas as pd
    df = pd.DataFrame(sample_complaints)
    df.to_csv("sample_complaints.csv", index=False)
    
    print("Created sample_complaints.csv with 3 test complaints")
    print("Run the main detector to analyze this file:")
    print("python job_scam_detector.py")

def example_custom_indicators():
    """Example: Add custom scam indicators"""
    print("\n" + "="*50)
    print("EXAMPLE 3: Custom Indicators")
    print("="*50)
    
    # You can extend the detector with custom indicators
    custom_indicators = {
        'crypto_red_flags': [
            'bitcoin', 'cryptocurrency', 'crypto payment', 'blockchain',
            'digital currency', 'wallet', 'mining'
        ],
        'recruitment_red_flags': [
            'no interview', 'hired immediately', 'start today',
            'no experience needed', 'easy money', 'quick cash'
        ]
    }
    
    print("Custom indicators you could add:")
    for category, indicators in custom_indicators.items():
        print(f"  {category}: {indicators}")

def example_risk_categories():
    """Example: Understanding risk categories"""
    print("\n" + "="*50)
    print("EXAMPLE 4: Risk Categories")
    print("="*50)
    
    risk_categories = {
        "0-30": "Low Risk - Likely legitimate job opportunity",
        "31-60": "Medium Risk - Suspicious, needs careful review", 
        "61-80": "High Risk - Likely scam, avoid",
        "81-100": "Very High Risk - Definite scam, report immediately"
    }
    
    print("Risk Score Interpretation:")
    for score_range, description in risk_categories.items():
        print(f"  {score_range}: {description}")

def example_prevention_tips():
    """Example: Scam prevention tips"""
    print("\n" + "="*50)
    print("EXAMPLE 5: Prevention Tips")
    print("="*50)
    
    prevention_tips = [
        "Never send money to employers",
        "Verify company legitimacy through official channels",
        "Be suspicious of jobs that require upfront payments",
        "Avoid jobs that only communicate via email or messaging apps",
        "Research companies thoroughly before accepting positions",
        "Never deposit checks from unknown sources",
        "Be wary of jobs that seem too good to be true",
        "Ask for detailed job descriptions and contracts",
        "Verify the identity of hiring managers",
        "Trust your instincts - if something feels wrong, it probably is"
    ]
    
    print("Key Prevention Tips:")
    for i, tip in enumerate(prevention_tips, 1):
        print(f"  {i}. {tip}")

def main():
    """Run all examples"""
    print("JOB SCAM DETECTION SYSTEM - USAGE EXAMPLES")
    print("="*60)
    
    example_single_analysis()
    example_batch_analysis()
    example_custom_indicators()
    example_risk_categories()
    example_prevention_tips()
    
    print("\n" + "="*60)
    print("To run the actual analysis:")
    print("1. Set your GEMINI_API_KEY environment variable")
    print("2. Run: python job_scam_detector.py")
    print("3. Run: python visualize_results.py")
    print("="*60)

if __name__ == "__main__":
    main()
