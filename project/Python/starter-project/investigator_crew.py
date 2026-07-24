import os
import certifi
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Crew, Task, Process, LLM
from crewai.project import CrewBase, agent, task, crew
from rpt1_sklearn_tool import call_rpt1
from grounding_tool import call_grounding_service

# Fix SSL certificate verification for corporate/enterprise environments
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["HTTPX_SSL_VERIFY"] = certifi.where()

# Load .env from the same directory as this script
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Shared LLM instance for all agents
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=api_key,
)

@CrewBase
class InvestigatorCrew():
    """InvestigatorCrew crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def appraiser_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['appraiser_agent'],
            llm=gemini_llm,
            verbose=True,
            tools=[call_rpt1]
        )

    @task
    def appraise_loss_task(self) -> Task:
        return Task(
            config=self.tasks_config['appraise_loss_task']
        )

    @agent
    def evidence_analyst_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['evidence_analyst_agent'],
            llm=gemini_llm,
            verbose=True,
            tools=[call_grounding_service]
        )

    @task
    def analyze_evidence_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_evidence_task']
        )

    @agent
    def lead_detective_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['lead_detective_agent'],
            llm=gemini_llm,
            verbose=True
            # No tools — the detective reasons from the other agents' outputs
        )

    @task
    def solve_crime(self) -> Task:
        return Task(
            config=self.tasks_config['solve_crime'],
            context=[self.appraise_loss_task(), self.analyze_evidence_task()]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
