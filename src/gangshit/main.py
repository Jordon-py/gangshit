#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from gangshit.crew import Gangshit

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """

    inputs = {
        'topic':"""Build a cross-platform desktop application that trains powerful ML and RL models to predict forex currency pair movements (e.g. EUR/USD, GBP/JPY), visualizes training and backtesting metrics, supports backtesting on historical data, provides explanatory tooltips suitable for both beginners and experts, and enables model export for real-world deployment.""",
        'requirements': [
            "Train both supervised ML (e.g. LSTM, random forest) and reinforcement learning (e.g. DQN, PPO) models using at least 10 years of historical forex data.",
            "Visualize training metrics (loss, accuracy, reward) in interactive charts (using Plotly/Matplotlib).",
            "Implement backtesting on multiple forex pairs with downloadable summary reports and equity curves.",
            "Offer user-friendly tooltips that explain model parameters, metrics, and actions in both beginner and expert modes.",
            "Provide model export functionality in standard formats (e.g. ONNX, Pickle) for external usage.",
            "Ensure GUI is intuitive and cross-platform (Electron, Tauri, or similar).",
            "Modular backend/frontend separation; code quality enforced via formatting and security audits.",
            "Include sample configurations/widgets for demoing forex model workflows."
        ],
        'current_year': str(datetime.now().year)
    }
    
    try:
        print("🚀 Starting Gangshit crew...")
        result = Gangshit().gangshit_crew().kickoff(inputs=inputs)
        print("✅ Crew execution completed!")
        print(f"📊 Result: {result}")
        return result
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
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
    run()
