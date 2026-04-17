# prompts/prompts.py
# All LangChain PromptTemplates live here.
# Keeping prompts separate makes them easy to tweak without touching pipeline logic.

from langchain_core.prompts import PromptTemplate


# ─── Prompt 1: Skill Extraction ───────────────────────────────────────────────
# First step in the pipeline. Reads the raw resume and pulls out structured info.
# The strict "DO NOT" rules are what prevent hallucinations here.

extraction_prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
You are a technical recruiter reading a resume.

Extract ONLY what is explicitly written in the resume below.
DO NOT infer or assume skills that are not clearly mentioned.
DO NOT add generic skills like 'communication' unless stated.

Resume:
{resume}

Return your answer in this exact format:

SKILLS: <comma-separated list of technical skills>
TOOLS: <comma-separated list of tools/frameworks/libraries>
EXPERIENCE: <total years of relevant experience as a number>
EDUCATION: <highest degree and field>
PROJECTS: <one-line summary of notable projects, or 'None mentioned'>
"""
)


# ─── Prompt 2: Matching + Scoring ─────────────────────────────────────────────
# Takes the structured extraction output (not the raw resume) and the JD,
# then figures out what matches, what's missing, and assigns a 0–100 score.

scoring_prompt = PromptTemplate(
    input_variables=["extracted_profile", "job_description"],
    template="""
You are an unbiased hiring evaluator. Score the candidate strictly based on facts.

CANDIDATE PROFILE (extracted from resume):
{extracted_profile}

JOB DESCRIPTION:
{job_description}

Scoring rules:
- Compare skills/tools mentioned in the profile against requirements in the JD
- Award points only for matches that are explicitly present
- Do NOT give benefit of the doubt for missing information
- Score range: 0 (no match) to 100 (perfect match)

Respond in this format:

MATCHED_SKILLS: <skills from profile that appear in JD requirements>
MISSING_SKILLS: <skills required by JD but absent from profile>
EXPERIENCE_FIT: <Yes/No/Partial — with one sentence why>
FIT_SCORE: <integer 0–100>
"""
)


# ─── Prompt 3: Human-Readable Explanation ─────────────────────────────────────
# Turns the raw score into actionable hiring advice.
# Written for a recruiter who may not know what PyTorch is.

explanation_prompt = PromptTemplate(
    input_variables=["scoring_result", "fit_score"],
    template="""
You are writing a hiring recommendation for a recruiter who isn't technical.
Be direct, honest, and use plain language. No fluff.

Scoring Result:
{scoring_result}

Fit Score: {fit_score} / 100

Write a short recommendation (3–5 sentences) that covers:
1. Whether to proceed with this candidate and why
2. Their strongest relevant strengths
3. The biggest gap(s) that might be a concern
4. A clear hiring suggestion (Strongly Recommend / Recommend / Maybe / Do Not Recommend)

End with: VERDICT: <one of the four options above>
"""
)


# ─── Bonus: JSON Output Prompt ─────────────────────────────────────────────────
# Same as scoring_prompt but returns structured JSON instead of text.
# Useful if you want to feed results into a database or UI.
# Includes few-shot examples so the model knows exactly what format to use.

json_scoring_prompt = PromptTemplate(
    input_variables=["extracted_profile", "job_description"],
    template="""
You are a hiring evaluator. Score the candidate and return ONLY valid JSON.
Do not include any text before or after the JSON object.

CANDIDATE PROFILE:
{extracted_profile}

JOB DESCRIPTION:
{job_description}

Few-shot examples of the expected format:

Example 1 (strong match):
{{"fit_score": 88, "matched_skills": ["Python", "PyTorch", "AWS"], "missing_skills": ["Azure"], "verdict": "Strongly Recommend", "summary": "Excellent technical fit with proven production ML experience."}}

Example 2 (weak match):
{{"fit_score": 22, "matched_skills": ["Python"], "missing_skills": ["PyTorch", "TensorFlow", "SQL", "Cloud"], "verdict": "Do Not Recommend", "summary": "Background is in web development, not data science."}}

Now evaluate the actual candidate above and return JSON in the exact same format:
"""
)
