import os
import ssl
import certifi
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai import LLM
from rpt1_sklearn_tool import call_rpt1
from payload import payload

# Fix SSL certificate verification for corporate/enterprise environments
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["HTTPX_SSL_VERIFY"] = certifi.where()

# Load .env from the same directory as this script
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Confirm key is loaded
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")
print(f"GEMINI_API_KEY loaded: {api_key[:8]}...")

# Explicitly configure the LLM with the API key
llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=api_key,
)

# Create a Loss Appraiser Agent
appraiser_agent = Agent(
    role="Loss Appraiser",
    goal=f"Predict the missing values of stolen items using the RPT-1 model via the call_rpt1 tool. Use this payload {payload} as input.",
    backstory="You are an expert insurance appraiser specializing in fine art valuation and theft assessment.",
    llm=llm,
    tools=[call_rpt1],
    verbose=True
)

# Create a task for the appraiser
appraise_loss_task = Task(
    description=f"Analyze the theft crime scene and predict the missing values of stolen items using the RPT-1 model via the call_rpt1 tool. Use this dict {payload} as input.",
    expected_output="JSON with predicted values for the stolen items.",
    agent=appraiser_agent
)

# Create a crew with the appraiser agent
crew = Crew(
    agents=[appraiser_agent],
    tasks=[appraise_loss_task],
    verbose=True
)

# Execute the crew
def main():
    result = crew.kickoff(inputs={'payload': payload})
    print("\n" + "="*50)
    print("Insurance Appraiser Report:")
    print("="*50)
    print(result)

if __name__ == "__main__":
    main()
