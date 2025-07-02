# =====================================================================
# Enhanced CrewAI Crew Class (2024+ Best Practices)
# =====================================================================

from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew, before_kickoff, after_kickoff
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

import os
import sys
from dotenv import load_dotenv
from crewai_tools import SerperDevTool

# Load environment variables from .env
load_dotenv(override=True)

# =========================
# 1. ENVIRONMENT LOADING & PATHS
# =========================
print("[CrewAI] Loading environment variables...")
load_dotenv(override=True)

# Validate required environment variables for Ollama
def assert_env_var(var, example):
    if os.getenv(var) is None:
        print(f"[ERROR] Required environment variable '{var}' not set! Example: {example}")
        sys.exit(1)

required_env = {
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "OLLAMA_LLAMA3": "llama3.2:latest",
    "OLLAMA_DEEPSEEK": "deepseek-r1:1.5b-qwen-distill-q8_0",
    "OLLAMA_GEMMA3": "gemma3:latest",
    # Add any others (SERPER_API_KEY, etc) as needed
}
for k, v in required_env.items():
    assert_env_var(k, v)

# Project file locations (relative to this file)
AGENTS_YAML = "config/agents.yaml"
TASKS_YAML = "config/tasks.yaml"

def full_path(rel_path):
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, rel_path)

if not os.path.exists(full_path(AGENTS_YAML)):
    print(f"[ERROR] Agents YAML not found at {AGENTS_YAML}")
    sys.exit(1)
if not os.path.exists(full_path(TASKS_YAML)):
    print(f"[ERROR] Tasks YAML not found at {TASKS_YAML}")
    sys.exit(1)

# =========================
# 2. CREWAI BASE CREW DEFINITION
# =========================
@CrewBase
class Gangshit:
    """
    CrewAI 2024+ Enhanced Orchestration Class.
    Loads agents & tasks from YAML, assigns LLMs/tools, enables verbose logging, and adds robust error handling.
    """

    # Specify YAML config files (CrewAI auto-loads by key)
    agents_config = AGENTS_YAML
    tasks_config = TASKS_YAML

    # =========================
    # 3. LLM MODEL CONFIGURATION
    # =========================
    @staticmethod
    def llm_models():
        """
        Loads LLM model names from environment variables, defaults provided.
        Returns: (LLAMA3, DEEPSEEK, GEMMA3)
        """
        # Print which models are being used
        LLAMA3 = os.getenv("OLLAMA_LLAMA3", "llama3.2:latest")
        DEEPSEEK = os.getenv("OLLAMA_DEEPSEEK", "deepseek-r1:1.5b-qwen-distill-q8_0")
        GEMMA3 = os.getenv("OLLAMA_GEMMA3", "gemma3:latest")
        print(f"[CrewAI] Using LLMs:")
        print(f"  LLAMA3:   {LLAMA3}")
        print(f"  DEEPSEEK: {DEEPSEEK}")
        print(f"  GEMMA3:   {GEMMA3}")
        return LLAMA3, DEEPSEEK, GEMMA3

    # Set class attributes (static)
    OLLAMA_LLAMA3, OLLAMA_DEEPSEEK, OLLAMA_GEMMA3 = llm_models.__func__()

    # =========================
    # 4. AGENT FACTORIES
    # =========================
    @agent
    def researcher(self) -> Agent:
        """
        Research specialist, uses GEMMA3 and web search tool.
        Config injected from YAML by CrewAI.
        """
        print("[CrewAI] Instantiating researcher agent...")
        try:
            return Agent(
                config=self.agents_config['researcher'],
                llm=self.OLLAMA_GEMMA3,
                verbose=True,
                tools=[SerperDevTool()]  # Add more tools if needed
            )
        except Exception as e:
            print(f"[ERROR] Failed to initialize 'researcher' agent: {e}")
            raise

    @agent
    def analyst(self) -> Agent:
        """
        Technical analyst, uses GEMMA3 and web search tool.
        """
        print("[CrewAI] Instantiating analyst agent...")
        try:
            return Agent(
                config=self.agents_config['analyst'],
                llm=self.OLLAMA_GEMMA3,
                verbose=True,
                tools=[SerperDevTool()]
            )
        except Exception as e:
            print(f"[ERROR] Failed to initialize 'analyst' agent: {e}")
            raise

    @agent
    def coding_agent(self) -> Agent:
        """
        ML/Full-stack coding agent, uses DEEPSEEK model.
        """
        print("[CrewAI] Instantiating coding_agent agent...")
        try:
            return Agent(
                config=self.agents_config['coding_agent'],
                llm=self.OLLAMA_DEEPSEEK,
                verbose=True
            )
        except Exception as e:
            print(f"[ERROR] Failed to initialize 'coding_agent' agent: {e}")
            raise

    @agent
    def overlord(self) -> Agent:
        """
        Orchestrator/QA agent, uses LLAMA3.
        """
        print("[CrewAI] Instantiating overlord agent...")
        try:
            return Agent(
                config=self.agents_config['overlord'],
                llm=self.OLLAMA_LLAMA3,
                verbose=True
            )
        except Exception as e:
            print(f"[ERROR] Failed to initialize 'overlord' agent: {e}")
            raise

    # =========================
    # 5. TASK FACTORIES
    # =========================
    @task
    def research_task(self) -> Task:
        """
        Task: Research with data curation and CSV/Markdown output.
        """
        print("[CrewAI] Configuring research_task...")
        try:
            return Task(
                config=self.tasks_config['research_task'],
                output_file='results/research_report.md'
            )
        except Exception as e:
            print(f"[ERROR] Failed to configure 'research_task': {e}")
            raise

    @task
    def analyst_task(self) -> Task:
        """
        Task: Analyze research output, create architecture spec.
        """
        print("[CrewAI] Configuring analyst_task...")
        try:
            return Task(
                config=self.tasks_config['analyst_task'],
                output_file='results/analyst_report.md'
            )
        except Exception as e:
            print(f"[ERROR] Failed to configure 'analyst_task': {e}")
            raise

    @task
    def coding_task(self) -> Task:
        """
        Task: Build app/codebase for topic, ensure compliance.
        """
        print("[CrewAI] Configuring coding_task...")
        try:
            return Task(
                config=self.tasks_config['coding_task'],
                output_file='results/coding_report.md'
            )
        except Exception as e:
            print(f"[ERROR] Failed to configure 'coding_task': {e}")
            raise

    @task
    def overlord_task(self) -> Task:
        """
        Task: Final orchestration, QA, compliance reporting.
        """
        print("[CrewAI] Configuring overlord_task...")
        try:
            return Task(
                config=self.tasks_config['overlord_task'],
                output_file='results/overlord_report.md'
            )
        except Exception as e:
            print(f"[ERROR] Failed to configure 'overlord_task': {e}")
            raise

    # =========================
    # 6. CREW ASSEMBLY
    # =========================
    @crew
    def gangshit_crew(self) -> Crew:
        """
        Assemble all agents and tasks into a hierarchical crew.
        Includes manager LLM and live tools.
        """
        print("[CrewAI] Assembling crew (hierarchical process)...")
        try:
            # Make sure results folder exists
            results_path = full_path("../../results")
            os.makedirs(results_path, exist_ok=True)

            assembled_crew = Crew(
                agents=self.agents,         # Injected agent instances
                tasks=self.tasks,           # Injected task instances
                process=Process.hierarchical,    # Multi-agent delegation
                manager_llm=self.OLLAMA_LLAMA3,  # Main LLM for orchestration
                verbose=True,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                tools=[SerperDevTool()],    # Global tools (add more as needed)
            )
            print("[CrewAI] Crew assembled successfully!")
            return assembled_crew
        except Exception as e:
            print(f"[ERROR] Failed to assemble crew: {e}")
            raise

# END OF FILE
# =========================