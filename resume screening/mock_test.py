import sys
import os
from unittest.mock import MagicMock, patch

# Add current directory to path so we can import our modules
sys.path.append(os.getcwd())

from langchain_core.messages import AIMessage

# Mock the OpenAI related calls before they are even used
with patch('langchain_openai.ChatOpenAI') as MockChat:
    # Set up the mock instance
    mock_llm = MagicMock()
    MockChat.return_value = mock_llm
    
    # Define mock responses for the 3 steps in the pipeline (as AIMessages)
    mock_llm.invoke.side_effect = [
        # Candidate 1: Priya (Strong)
         AIMessage(content="SKILLS: Python, PyTorch\nTOOLS: AWS\nEXPERIENCE: 3\nEDUCATION: M.Sc.\nPROJECTS: NLP Sentiment Analyzer"),
         AIMessage(content="MATCHED_SKILLS: Python, PyTorch\nMISSING_SKILLS: None\nEXPERIENCE_FIT: Yes\nFIT_SCORE: 95"),
         AIMessage(content="Excellent candidate for the role. VERDICT: Strongly Recommend"),
         
        # Candidate 2: Rahul (Average)
         AIMessage(content="SKILLS: Python, SQL\nTOOLS: MySQL\nEXPERIENCE: 2\nEDUCATION: B.Tech\nPROJECTS: Sales Forecasting"),
         AIMessage(content="MATCHED_SKILLS: Python, SQL\nMISSING_SKILLS: PyTorch, Cloud\nEXPERIENCE_FIT: Partial\nFIT_SCORE: 55"),
         AIMessage(content="Meets basic requirements but lacks specialization. VERDICT: Maybe"),
         
        # Candidate 3: Amit (Weak)
         AIMessage(content="SKILLS: JavaScript, Django\nTOOLS: Git\nEXPERIENCE: 3\nEDUCATION: B.Sc.\nPROJECTS: Online food delivery"),
         AIMessage(content="MATCHED_SKILLS: Python (basic)\nMISSING_SKILLS: ML, Statistics, SQL (production)\nEXPERIENCE_FIT: No\nFIT_SCORE: 20"),
         AIMessage(content="Background is in web development, not Data Science. VERDICT: Do Not Recommend")
    ]

    # Now import and run the screening logic from pipeline
    from chains.pipeline import build_chains, run_screening
    
    # Mock data (mimics main.py)
    job_description = "Data Scientist with Python and PyTorch experience."
    resume_text = "Priya Sharma, 3 years exp, Python, PyTorch, AWS."
    
    print("Starting Mock Pipeline Test...")
    print("-" * 40)
    
    chains = build_chains(mock_llm)
    result = run_screening(resume_text, job_description, chains)
    
    print(f"Candidate: Priya Sharma")
    print(f"Score: {result['score']}")
    print(f"Verdict: {result['explanation']}")
    print("-" * 40)
    print("Logic Check: Pipeline connected successfully and processed mock data.")
