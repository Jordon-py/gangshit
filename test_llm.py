#!/usr/bin/env python3
"""Simple test script to verify LLM connectivity"""

import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

def test_llm():
    """Test different LLM configurations"""
    
    # Test Ollama Deepseek (Primary)
    deepseek_model = os.getenv("OLLAMA_DEEPSEEK", "deepseek-r1:8b")
    if deepseek_model:
        print(f"🔧 Testing Ollama Deepseek: {deepseek_model}")
        try:
            llm = LLM(
                model=f"ollama/{deepseek_model.replace('ollama/', '')}",
                base_url="http://localhost:11434",
                stream=False
            )
            response = llm.call("Say hello in one word.")
            print(f"✅ Ollama Deepseek Success! Response: {response}")
            return llm
        except Exception as e:
            print(f"❌ Ollama Deepseek failed: {str(e)[:100]}...")
    
    # Test OpenRouter
    openrouter_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
    if openrouter_key:
        print(f"🔧 Testing OpenRouter with key: {openrouter_key[:10]}...")
        
        models_to_test = [
            "qwen/qwen-2.5-7b-instruct:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free"
        ]
        
        for model in models_to_test:
            try:
                print(f"   Testing: {model}")
                llm = LLM(
                    model=f"openrouter/{model}",
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_key,
                    stream=False
                )
                
                # Simple test call
                response = llm.call("Say hello in one word.")
                print(f"   ✅ Success! Response: {response}")
                return llm
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)[:100]}...")
                continue
    
    # Test Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print(f"🔧 Testing Groq with key: {groq_key[:10]}...")
        try:
            llm = LLM(
                model="groq/llama-3.1-8b-instant",
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_key,
                stream=False
            )
            response = llm.call("Say hello in one word.")
            print(f"✅ Groq Success! Response: {response}")
            return llm
        except Exception as e:
            print(f"❌ Groq failed: {e}")
    
    print("❌ No working LLM found!")
    return None

if __name__ == "__main__":
    working_llm = test_llm()
    if working_llm:
        print("\n🎉 Found a working LLM configuration!")
    else:
        print("\n💔 No working LLM found. Check your API keys in .env file.")
