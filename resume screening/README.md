# AI Resume Screening System 🤖

An AI-powered recruitment tool built with LangChain and LangSmith to automate resume evaluation against job descriptions.

## 📁 Project Structure

This project follows a modular architecture as required for the Data Science Internship task:

- `main.py`: The entry point script to run the screening for all candidates.
- `prompts/`: Contains structured prompt templates.
    - `prompts.py`: Defines Skill Extraction, Scoring, and Explanation prompts.
- `chains/`: Contains the logic for building LangChain pipelines.
    - `pipeline.py`: Defines the LCEL chains and the core screening logic.
- `AI_Resume_Screener.ipynb`: A Jupyter Notebook version for interactive execution and visualization.
- `.env`: (To be created) Stores your API keys (OpenAI and LangSmith).
- `requirements.txt`: Lists all necessary Python dependencies.

## 🚀 How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set Up API Keys**:
    Create a `.env` file in the root directory and add your keys:
    ```env
    OPENAI_API_KEY=your_openai_key_here
    LANGCHAIN_API_KEY=your_langsmith_key_here
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_PROJECT=resume-screener
    ```

3.  **Run the Screening**:
    ```bash
    python main.py
    ```

## 📊 Pipeline Flow

The system processes resumes through four distinct steps:
1.  **Skill Extraction**: Pulls technical skills, tools, and experience from the raw resume text.
2.  **Matching**: Compares extracted data against the Job Description requirements.
3.  **Scoring**: Assigns a numeric fit score (0–100) based on objective matches.
4.  **Explanation**: Generates a plain-English recommendation for a recruiter.

## 🔍 Tracing and Debugging

This project is integrated with **LangSmith**. Every run is traced, allowing you to:
- Monitor pipeline performance.
- Debug incorrect outputs.
- View step-by-step reasoning of the LLM.

---
*Built for: Data Science Internship Assignment — February 2026*
