# chains/pipeline.py
# Wires up the prompts into runnable chains.
# Nothing in here touches the actual resume data — that stays in main.py.

import re
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from prompts.prompts import extraction_prompt, scoring_prompt, explanation_prompt


def build_llm() -> ChatOpenAI:
    """
    Creates the language model instance.
    temperature=0 keeps outputs consistent and deterministic —
    which is what you want for a scoring system.
    """
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )


def build_chains(llm: ChatOpenAI) -> dict:
    """
    Builds and returns the three LCEL chains.
    Each chain is: prompt | llm | output_parser
    """
    parser = StrOutputParser()

    return {
        "extract": extraction_prompt | llm | parser,
        "score":   scoring_prompt   | llm | parser,
        "explain": explanation_prompt | llm | parser,
    }


def parse_score(scoring_text: str) -> str:
    """Extracts the integer score from the scoring step's text output."""
    match = re.search(r"FIT_SCORE:\s*(\d+)", scoring_text)
    return match.group(1) if match else "N/A"


def run_screening(resume: str, job_description: str, chains: dict) -> dict:
    """
    Runs the full 3-step pipeline for one candidate.

    Args:
        resume:          Raw resume text
        job_description: The job posting text
        chains:          Dict of chains returned by build_chains()

    Returns:
        Dict with keys: extracted, scored, score, explanation
    """
    # Step 1 — What does this person actually bring to the table?
    extracted = chains["extract"].invoke({"resume": resume})

    # Step 2 — How well does it line up with what the job needs?
    scored = chains["score"].invoke({
        "extracted_profile": extracted,
        "job_description": job_description
    })

    # Step 3 — Turn the numbers into something a recruiter can act on
    score_value = parse_score(scored)
    explanation = chains["explain"].invoke({
        "scoring_result": scored,
        "fit_score": score_value
    })

    return {
        "extracted":   extracted,
        "scored":      scored,
        "score":       score_value,
        "explanation": explanation,
    }
