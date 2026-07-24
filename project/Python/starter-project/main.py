from investigator_crew import InvestigatorCrew
from payload import payload

def main():
    result = InvestigatorCrew().crew().kickoff(inputs={
        'payload': payload,
        'suspect_names': "Sophie Dubois, Marcus Chen, Viktor Petrov"
    })
    print("\n📘 Result:\n", result)

if __name__ == "__main__":
    main()