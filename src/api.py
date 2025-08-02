from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import string
import joblib
import os
import requests

# === Google Drive Large File Download Helpers ===

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

# === Model loading with fallback download ===

MODEL_PATH = os.getenv("MODEL_PATH", "models/naive_bayes.joblib")
MODEL_URL = os.getenv("MODEL_URL")

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print(f"🔄 Model not found at {MODEL_PATH}, downloading...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        # Extract Google Drive file ID from the URL
        if "id=" in MODEL_URL:
            file_id = MODEL_URL.split("id=")[-1]
        else:
            raise RuntimeError("MODEL_URL is missing 'id=' parameter for Google Drive file ID extraction.")
        download_file_from_google_drive(file_id, MODEL_PATH)
        print(f"✅ Model downloaded successfully from Google Drive. File size: {os.path.getsize(MODEL_PATH)} bytes")

download_model_if_needed()
model = joblib.load(MODEL_PATH)

# === FastAPI App Setup ===

app = FastAPI(title="AI vs Human Text Classifier")

origins = [
    "https://outlook.office.com",
    "https://outlook.office365.com",
    "http://localhost:8000",
    "http://localhost:3000",
    "https://jasulcaf.github.io/AI-Human-Writting-Detection/"
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
