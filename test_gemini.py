#!/usr/bin/env python3
"""
Simple test script to debug Gemini API issues
"""

import os
import google.generativeai as genai

def test_gemini_connection():
    """Test basic Gemini connection and model availability"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        print("Set it with: export GEMINI_API_KEY='your_key_here'")
        return False
    
    print("✅ API key found")
    
    try:
        genai.configure(api_key=api_key)
        print("✅ Gemini configured successfully")
    except Exception as e:
        print(f"❌ Error configuring Gemini: {e}")
        return False
    
    # Test 1: List models
    print("\n🔍 Testing model listing...")
    try:
        models = genai.list_models()
        print(f"✅ Found {len(models)} models")
        
        for model in models:
            print(f"  - {model.name}")
            print(f"    Methods: {model.supported_generation_methods}")
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return False
    
    # Test 2: Try specific models
    print("\n🧪 Testing specific models...")
    test_models = [
        'gemini-1.5-pro',
        'gemini-1.5-flash', 
        'gemini-1.0-pro',
        'gemini-pro'
    ]
    
    working_model = None
    for model_name in test_models:
        try:
            print(f"Testing {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Hello, are you working?")
            print(f"✅ {model_name} - SUCCESS")
            print(f"   Response: {response.text[:50]}...")
            working_model = model_name
            break
        except Exception as e:
            print(f"❌ {model_name} - FAILED: {e}")
    
    if working_model:
        print(f"\n🎉 Working model found: {working_model}")
        return True
    else:
        print("\n❌ No working models found")
        return False

if __name__ == "__main__":
    print("Gemini API Test")
    print("=" * 30)
    test_gemini_connection()
