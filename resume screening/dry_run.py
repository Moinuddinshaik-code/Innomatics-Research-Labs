import sys
import os
from unittest.mock import MagicMock

# Add current directory to path
sys.path.append(os.getcwd())

# Define a simple mock for a LangChain chain
class MockChain:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0
    
    def invoke(self, inputs):
        res = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return res

def test_modular_logic():
    print("Starting Modular Logic Test (Dry Run)...")
    print("-" * 45)

    # 1. Mock the chains
    # Each chain normally returns a string (thanks to StrOutputParser)
    mock_chains = {
        "extract": MockChain(["SKILLS: Python, SQL\nTOOLS: MySQL\nEXPERIENCE: 2\nEDUCATION: B.Tech"]),
        "score":   MockChain(["MATCHED_SKILLS: Python, SQL\nFIT_SCORE: 85"]),
        "explain": MockChain(["Candidate is a strong fit. VERDICT: Recommend"])
    }

    # 2. Import the screening function from our new structure
    try:
        from chains.pipeline import run_screening
        print("Successfully imported run_screening from chains.pipeline")
    except ImportError as e:
        print(f"Import Error: {e}")
        return

    # 3. Simulate a run
    resume = "Name: Rahul, Skills: Python, SQL"
    jd = "WANTED: Python Developer"
    
    print("Processing candidate: Rahul...")
    result = run_screening(resume, jd, mock_chains)

    # 4. Verify results
    print("-" * 45)
    print(f"Extraction Result: {result['extracted'].replace('\n', ' ')}")
    print(f"Score Value: {result['score']}")
    print(f"Final Recommendation: {result['explanation']}")
    print("-" * 45)
    
    if result['score'] == '85' and 'Recommend' in result['explanation']:
        print("SUCCESS: The modular pipeline logic is working correctly.")
        print("The system is ready for real API keys in the .env file.")
    else:
        print("FAILED: Unexpected result from the pipeline logic.")

if __name__ == "__main__":
    test_modular_logic()
