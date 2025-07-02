#!/usr/bin/env python
import sys
from typing import Dict
import warnings
import os
import asyncio
from datetime import datetime
from crewai_tools import SerperDevTool
from gangshit.crew import Gangshit, Crew


warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

# src/vegas_crew/main.py

import os
from dotenv import load_dotenv

# Import your CrewBase class
from gangshit.crew import Gangshit
import asyncio

async def async_main():
    """
    Async entrypoint for running the CrewAI workflow.
    Loads environment, instantiates crew, and runs kickoff with robust error handling.
    """
    load_dotenv(override=True)
    crew = Gangshit()
    topic = "Vegas Nightlife Analytics"
    requirements = "Real-time trend detection, social sentiment mapping, and high-throughput live web integration"

    print("🚀 Kicking off CrewAI multi-agent workflow...")
    print(f"  • Topic:        {topic}")
    print(f"  • Requirements: {requirements}")
    print("-" * 40)

    try:
        # Await the async kickoff method
        result = await crew.gangshit_crew().kickoff(
            inputs={
                "topic": topic,
                "requirements": requirements
            }
        )
    except Exception as e:
        import traceback
        print("\n❌ Exception during CrewAI kickoff:")
        traceback.print_exc()
        return None

    print("\n✅ CrewAI Orchestration complete!\n")
    if isinstance(result, dict):
        for k, v in result.items():
            print(f"\n--- Output: {k} ---\n")
            print(v if isinstance(v, str) else str(v))
    else:
        print(result)
    return result



def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year),
    }
    try:
        Gangshit().gangshit_crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")



def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Gangshit().gangshit_crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    
    try:
        Gangshit().gangshit_crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    # Run the async main function using asyncio
    asyncio.run(async_main())
    