#!/usr/bin/env python3
"""Simple test script to verify Ollama Llama3.2 connectivity"""

import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

def test_llm():
    """Test Ollama Llama3.2 LLM configuration"""
    
    # Test Ollama Llama3.2 only
    llama_model = os.getenv("OLLAMA_LLAMA3", "llama3.2")
    print(f"🔧 Testing Ollama Llama3.2: {llama_model}")
    
    try:
        llm = LLM(
            model=f"ollama/{llama_model.replace('ollama/', '')}",
            base_url="http://localhost:11434",
            stream=False
        )
        
        # Simple test call
        response = llm.call("Say hello in one word.")
        print(f"✅ Ollama Llama3.2 Success! Response: {response}")
        return llm
        
    except Exception as e:
        print(f"❌ Ollama Llama3.2 failed: {str(e)}")
        print("Make sure Ollama is running and llama3.2 model is available.")
        print("Run: ollama pull llama3.2")
        return None

if __name__ == "__main__":
    working_llm = test_llm()
    if working_llm:
        print("\n🎉 Ollama Llama3.2 is working!")
    else:
        print("\n💔 Ollama Llama3.2 is not working. Check Ollama installation and model availability.")
