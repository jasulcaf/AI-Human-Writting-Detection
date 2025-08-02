from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import string
import joblib
import os
import requests

# === Model loading with fallback download ===

MODEL_PATH = os.getenv("MODEL_PATH", "models/naive_bayes.joblib")
MODEL_URL = os.getenv("MODEL_URL")

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print(f"🔄 Model not found at {MODEL_PATH}, downloading...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            print("✅ Model downloaded successfully.")
        else:
            raise RuntimeError(f"Failed to download model: {response.status_code}")

download_model_if_needed()
model = joblib.load(MODEL_PATH)

# === FastAPI App Setup ===

app = FastAPI(title="AI vs Human Text Classifier")

origins = [
    "https://outlook.office.com",
    "https://outlook.office365.com",
    "http://localhost:8000",   # Local testing
    "http://localhost:3000",   # If testing add-in locally via web
    "https://jasulcaf.github.io/AI-Human-Writting-Detection/"  # GitHub Pages
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Input Schema ===

class EmailInput(BaseModel):
    text: str

# === Prediction Logic ===

def get_confidence_score(text):
    proba = model.predict_proba([text])[0][1]
    return round(proba * 100, 2)

def get_label(score):
    if score > 75:
        return "Likely AI"
    elif score < 30:
        return "Likely Human"
    else:
        return "Some AI assistance used"

# === Text Cleaning Functions ===

def remove_tags(text):
    tags = ['\n', '\'']
    for tag in tags:
        text = text.replace(tag, '')
    return text

def remove_punc(text):
    return ''.join([char for char in text if char not in string.punctuation])

def lowercase(text):
    return text.lower()

def clean_text(text):
    text = remove_tags(text)
    text = remove_punc(text)
    return lowercase(text)

# === API Endpoint ===

@app.post("/predict")
def predict(input: EmailInput):
    clean_input = clean_text(input.text)
    score = get_confidence_score(clean_input)
    label = get_label(score)
    return {"confidence": score, "label": label}