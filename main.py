from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
import chromadb
from chromadb.config import Settings
import os
import json
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------
# Load Environment
# ---------------------------------------------------
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found.")

# ---------------------------------------------------
# FastAPI Setup
# ---------------------------------------------------
app = FastAPI(title="BharatAI – Scheme & Job Eligibility Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Gemini Client
# ---------------------------------------------------
client = genai.Client(api_key=api_key)

# ---------------------------------------------------
# Chroma Setup
# ---------------------------------------------------
chroma_client = chromadb.Client(
    Settings(
        persist_directory="./chroma_db",
        is_persistent=True
    )
)

collection = chroma_client.get_or_create_collection(name="gov_schemes")

# ---------------------------------------------------
# Sample Government Job Dataset (Mock Live Data)
# ---------------------------------------------------
government_jobs = [
    {
        "job_title": "SSC Junior Clerk",
        "department": "Staff Selection Commission",
        "min_age": 18,
        "max_age": 27,
        "required_qualification": "Graduate",
        "required_documents": ["Aadhaar", "Educational Certificate", "Caste Certificate (if applicable)"]
    },
    {
        "job_title": "Railway Group D",
        "department": "Indian Railways",
        "min_age": 18,
        "max_age": 33,
        "required_qualification": "10th Pass",
        "required_documents": ["Aadhaar", "10th Marksheet", "Income Certificate"]
    }
]

# ---------------------------------------------------
# Pydantic Model
# ---------------------------------------------------
class UserProfile(BaseModel):
    age: int
    income: int
    occupation: str
    state: str

# ---------------------------------------------------
# Main Endpoint
# ---------------------------------------------------
@app.post("/check")
def check_eligibility(user: UserProfile):

    # -------------------------------
    # 1️⃣ Create Query for Scheme RAG
    # -------------------------------
    query_text = f"""
    Age: {user.age}
    Income: {user.income}
    Occupation: {user.occupation}
    State: {user.state}
    """

    query_embedding_response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[query_text]
    )

    query_vector = query_embedding_response.embeddings[0].values

    search_results = collection.query(
        query_embeddings=[query_vector],
        n_results=4
    )

    relevant_docs = search_results.get("documents", [[]])[0]

    if not relevant_docs:
        return {
            "status": "No Match",
            "message": "No relevant schemes found."
        }

    # -------------------------------
    # 2️⃣ Gemini Combined Prompt
    # -------------------------------
    prompt = f"""
    You are an AI government advisory system.

    User Profile:
    {user.model_dump()}

    Government Schemes:
    {relevant_docs}

    Government Job Vacancies:
    {government_jobs}

    Tasks:
    1. Evaluate eligibility for schemes.
    2. Evaluate eligibility for government job vacancies.
    3. Provide structured reasoning.
    4. Do NOT assume data outside what is provided.

    Return ONLY valid JSON in this format:

    {{
        "schemes": [
            {{
                "scheme_name": "string",
                "eligible": true/false,
                "reason": "string",
                "required_documents": ["list"],
                "eligibility_score": 0-100
            }}
        ],
        "jobs": [
            {{
                "job_title": "string",
                "department": "string",
                "eligible": true/false,
                "reason": "string",
                "required_documents": ["list"]
            }}
        ]
    }}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )

    try:
        parsed = json.loads(response.text)

        schemes = parsed.get("schemes", [])
        jobs = parsed.get("jobs", [])

        # Rank schemes
        schemes_sorted = sorted(
            schemes,
            key=lambda x: x.get("eligibility_score", 0),
            reverse=True
        )

        best_scheme = schemes_sorted[0] if schemes_sorted else None
        others = schemes_sorted[1:] if len(schemes_sorted) > 1 else []

        confidence = "Low"
        if best_scheme:
            score = best_scheme.get("eligibility_score", 0)
            if score >= 75:
                confidence = "High"
            elif score >= 40:
                confidence = "Medium"

        return {
            "status": "Success",
            "best_scheme": best_scheme,
            "other_schemes": others,
            "job_opportunities": jobs,
            "confidence": confidence
        }

    except Exception:
        return {
            "status": "Error",
            "message": "Model did not return valid JSON.",
            "raw_output": response.text
        }