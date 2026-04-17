# main.py
# Run this from the terminal: python main.py
# Or just open the notebook if you prefer cell-by-cell execution.

import os
import re
import json
from dotenv import load_dotenv

from chains.pipeline import build_llm, build_chains, run_screening


# ─── Setup ────────────────────────────────────────────────────────────────────

load_dotenv()

# Enable LangSmith tracing — every run below will appear in your dashboard
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "resume-screener")

print("✅ LangSmith tracing ON")
print(f"   Project: {os.environ['LANGCHAIN_PROJECT']}")
print(f"   Dashboard: https://smith.langchain.com\n")


# ─── Build the pipeline ───────────────────────────────────────────────────────

llm    = build_llm()
chains = build_chains(llm)


# ─── Job Description ──────────────────────────────────────────────────────────

job_description = """
Position: Data Scientist
Company: DataFlow Analytics

Requirements:
- 3+ years of experience in data science or machine learning
- Strong Python skills (pandas, numpy, scikit-learn)
- Experience with deep learning frameworks (TensorFlow or PyTorch)
- Proficiency in SQL and data wrangling
- Familiarity with cloud platforms (AWS, GCP, or Azure)
- Experience deploying ML models to production
- Good communication — ability to explain findings to non-technical stakeholders
- Master's or PhD in CS, Statistics, or related field preferred
"""


# ─── Candidates ───────────────────────────────────────────────────────────────

resume_strong = """
Priya Sharma — Data Scientist
Email: priya@email.com | LinkedIn: linkedin.com/in/priyasharma

EXPERIENCE
Senior Data Scientist — TechCorp (2021–2024, 3 years)
  - Built customer churn prediction model (PyTorch) with 91% accuracy, deployed on AWS SageMaker
  - Reduced model inference time by 40% using ONNX optimization
  - Presented monthly insights to C-suite, simplifying complex ML results into business language

Data Analyst — InfoSys (2019–2021, 2 years)
  - Created automated ETL pipelines using Python (pandas, numpy) and PostgreSQL
  - Managed 50M+ row datasets on GCP BigQuery

EDUCATION
M.Sc. in Data Science — IIT Bombay (2019)

SKILLS
Python, pandas, numpy, scikit-learn, PyTorch, TensorFlow (basic), SQL, PostgreSQL

TOOLS
AWS SageMaker, GCP BigQuery, Docker, Git, Jupyter, MLflow

PROJECTS
NLP Sentiment Analyzer — fine-tuned BERT model for product review classification
"""

resume_average = """
Rahul Nair — Junior Data Scientist
Email: rahul@email.com

EXPERIENCE
Data Analyst — StartupXYZ (2022–2024, 2 years)
  - Built dashboards using Python and matplotlib to track product KPIs
  - Wrote SQL queries for data extraction and reporting
  - Trained basic classification models with scikit-learn

EDUCATION
B.Tech in Computer Science — VIT University (2022)

SKILLS
Python, pandas, scikit-learn, SQL, data visualization

TOOLS
MySQL, Excel, Tableau, Jupyter Notebook, Git

PROJECTS
Sales Forecasting using Linear Regression (scikit-learn) — personal project
"""

resume_weak = """
Amit Verma — Software Developer
Email: amit@email.com

EXPERIENCE
Web Developer — Freelance (2021–2024, 3 years)
  - Built e-commerce websites using Django and React
  - Wrote REST APIs in Python (Flask)
  - Used MySQL for database management

EDUCATION
B.Sc. in Information Technology — Mumbai University (2021)

SKILLS
Python, JavaScript, HTML/CSS, Django, Flask, REST APIs

TOOLS
MySQL, Git, Postman, VS Code, Linux

PROJECTS
Online food delivery platform — full-stack web application
"""

candidates = [
    {"name": "Priya Sharma (Strong)",  "resume": resume_strong},
    {"name": "Rahul Nair (Average)",   "resume": resume_average},
    {"name": "Amit Verma (Weak)",      "resume": resume_weak},
]


# ─── Run Screening ────────────────────────────────────────────────────────────

results = {}

for candidate in candidates:
    name = candidate["name"]
    print(f"\n{'='*60}")
    print(f"🔍 Screening: {name}")
    print(f"{'='*60}")

    result = run_screening(candidate["resume"], job_description, chains)
    results[name] = result

    print(f"\n📋 EXTRACTED PROFILE:\n{result['extracted']}")
    print(f"\n📊 SCORING BREAKDOWN:\n{result['scored']}")
    print(f"\n💬 RECOMMENDATION:\n{result['explanation']}")
    print(f"\n✅ Final Score: {result['score']} / 100")


# ─── Summary Table ────────────────────────────────────────────────────────────

def extract_verdict(text: str) -> str:
    match = re.search(r"VERDICT:\s*(.+)", text)
    return match.group(1).strip() if match else "—"

print("\n" + "═"*65)
print("  CANDIDATE SCREENING SUMMARY")
print("═"*65)
print(f"{'Candidate':<30} {'Score':>7}   {'Verdict'}")
print("-"*65)

for name, result in results.items():
    verdict = extract_verdict(result["explanation"])
    print(f"{name:<30} {result['score']:>5}/100   {verdict}")

print("═"*65)
