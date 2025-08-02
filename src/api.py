from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import string
import joblib
import os
import requests

# === Google Drive download helper ===

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

def save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)

# === Model loading with fallback download ===

MODEL_PATH = os.getenv("MODEL_PATH", "models/naive_bayes.joblib")
MODEL_URL = os.getenv("MODEL_URL")

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print(f"🔄 Model not found at {MODEL_PATH}, downloading...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        if MODEL_URL and "drive.google.com" in MODEL_URL:
            import re
            file_id_match = re.search(r'id=([a-zA-Z0-9_-]+)', MODEL_URL)
            if not file_id_match:
                raise RuntimeError("Could not extract Google Drive file ID from MODEL_URL")
            file_id = file_id_match.group(1)

            download_file_from_google_drive(file_id, MODEL_PATH)
            print("✅ Model downloaded successfully from Google Drive.")
        else:
            if not MODEL_URL:
                raise RuntimeError("MODEL_URL is not set.")
            response = requests.get(MODEL_URL)
            if response.status_code == 200:
                with open(MODEL_PATH, 'wb') as f:
                    f.write(response.content)
                print("✅ Model downloaded successfully from URL.")
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
