#!/usr/bin/env python3
"""
Check available Gemini models
"""

import os
import google.generativeai as genai

def check_available_models():
    """Check which Gemini models are available"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        return
    
    genai.configure(api_key=api_key)
    
    print("Checking available Gemini models...")
    
    # List all available models
    try:
        models = genai.list_models()
        print("\nAvailable models:")
        working_models = []
        
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                model_name = model.name.replace('models/', '')
                print(f"  ✓ {model.name}")
                working_models.append(model_name)
            else:
                print(f"  ✗ {model.name} (no generateContent support)")
        
        print(f"\nFound {len(working_models)} models that support generateContent")
        
        # Test each working model
        print(f"\nTesting working models:")
        for model_name in working_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Hello")
                print(f"  ✓ {model_name} - Working")
            except Exception as e:
                print(f"  ✗ {model_name} - Error: {e}")
                
    except Exception as e:
        print(f"Error listing models: {e}")
        print("This might be due to API permissions or network issues.")
    
    # Test specific models as fallback
    test_models = ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-1.0-pro', 'gemini-pro']
    
    print(f"\nTesting fallback models:")
    for model_name in test_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Hello")
            print(f"  ✓ {model_name} - Working")
        except Exception as e:
            print(f"  ✗ {model_name} - Error: {e}")

if __name__ == "__main__":
    check_available_models()
