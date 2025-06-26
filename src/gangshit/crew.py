# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from crewai.agents.agent_builder.base_agent import BaseAgent
from gangshit.tools import MyCustomListener
from typing import List
import os
from dotenv import load_dotenv

load_dotenv(override=True)  # Load environment variables from .env file

# Comment out the listener setup for now
# my_listener = MyCustomListener()
# my_listener.setup_listeners(crewai_event_bus)

@CrewBase
class Gangshit():
    """Gangshit crew for web application development"""

    agents_config_path = 'config/agents.yaml'
    tasks_config_path = 'config/tasks.yaml'

    # LLM Configuration with Ollama primary, fallbacks
    def _get_llm(self):
        """Get the best available LLM configuration"""
        
        # Primary: Ollama Deepseek R1
        deepseek_model = os.getenv("OLLAMA_DEEPSEEK", "deepseek-r1:8b")
        if deepseek_model:
            try:
                print(f"🔧 Using Ollama Deepseek: {deepseek_model}")
                return LLM(
                    model=f"ollama/{deepseek_model.replace('ollama/', '')}",
                    base_url="http://localhost:11434",
                    stream=False,  # Disable streaming for stability
                )
            except Exception as e:
                print(f"⚠️  Ollama Deepseek failed: {e}")
        
        # Fallback 1: OpenRouter
        openrouter_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
        if openrouter_key:
            print("🔧 Falling back to OpenRouter...")
            return LLM(
                model="openrouter/qwen/qwen-2.5-7b-instruct:free",
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
                stream=False,
            )
        
        # Fallback 2: Groq
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            print("🔧 Falling back to Groq...")
            return LLM(
                model="groq/llama-3.1-8b-instant",
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_key,
                stream=False,
            )
        
        # Last resort - try Ollama without API key
        print("⚠️  Warning: Using basic Ollama configuration")
        return LLM(
            model="ollama/deepseek-r1:8b",
            base_url="http://localhost:11434",
            stream=False,
        )

    @property
    def llm(self):
        if not hasattr(self, '_llm'):
            self._llm = self._get_llm()
        return self._llm

    @before_kickoff
    def before_kickoff(self, inputs):
        print("🚀 Crew is about to start!")
        print(f"📝 Topic: {inputs.get('topic', 'Not specified')}")

    @after_kickoff
    def after_kickoff(self, output):
        print("✅ Crew has completed execution!")
        print(f"📊 Generated {len(str(output))} characters of analysis")

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            llm=self.llm,
        )

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['analyst'],
            verbose=True,
            llm=self.llm,
        )
        
    @agent
    def coding_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['coding_agent'],
            verbose=True,
            llm=self.llm,
        )
        
    @agent
    def overlord(self) -> Agent:
        return Agent(
            config=self.agents_config['overlord'],
            verbose=True,
            llm=self.llm,
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def researcher_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
            output_file='research_report.md',
        )

    @task
    def analyst_task(self) -> Task:
        return Task(
            config=self.tasks_config['analysis_task'],
            output_file='analyst_report.md'
        )
        
    @task
    def coding_task(self) -> Task:
        return Task(
            config=self.tasks_config['coding_task'],
            output_file='coding_report.md'
        )
        
    @task
    def overlord_task(self) -> Task:
        return Task(
            config=self.tasks_config['overlord_task'],
            output_file='overlord_report.md'
        )

    @crew
    def gangshit_crew(self) -> Crew:
        """Creates the Gangshit crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
