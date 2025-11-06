#!/usr/bin/env python3
"""
Setup script for Job Scam Detection System
Installs dependencies and sets up the environment
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        return False

def check_api_key():
    """Check if Gemini API key is set"""
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print("✓ GEMINI_API_KEY is set")
        return True
    else:
        print("✗ GEMINI_API_KEY not found")
        print("Please set your API key:")
        print("export GEMINI_API_KEY='your_api_key_here'")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['visualizations', 'output', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        import pandas
        import numpy
        import matplotlib
        import seaborn
        import google.generativeai
        print("✓ All required modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("Job Scam Detection System - Setup")
    print("=" * 40)
    
    # Install requirements
    if not install_requirements():
        print("Setup failed at dependency installation")
        return False
    
    # Check API key
    api_key_ok = check_api_key()
    
    # Create directories
    create_directories()
    
    # Test imports
    imports_ok = test_imports()
    
    print("\n" + "=" * 40)
    if api_key_ok and imports_ok:
        print("✓ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python job_scam_detector.py")
        print("2. Run: python visualize_results.py")
    else:
        print("✗ Setup completed with warnings")
        if not api_key_ok:
            print("- Set your GEMINI_API_KEY environment variable")
        if not imports_ok:
            print("- Check that all dependencies are installed")
    
    return api_key_ok and imports_ok

if __name__ == "__main__":
    main()
