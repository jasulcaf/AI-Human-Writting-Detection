from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import string
import joblib
import os

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/naive_bayes.joblib")
model = joblib.load(DEFAULT_MODEL_PATH)

app = FastAPI(title="AI vs Human Text Classifier")

# Allow Outlook and local dev origins
origins = [
    "https://outlook.office.com",
    "https://outlook.office365.com",
    "http://localhost:8000"  # For your local testing (optional)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # Domains allowed to access API
    allow_credentials=True,
    allow_methods=["*"],             # Allow all HTTP methods
    allow_headers=["*"],             # Allow all headers
)

class EmailInput(BaseModel):
    text: str

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
    
def remove_tags(text):
    tags = ['\n', '\'']
    for tag in tags:
        # Replace with an empty string instead of the tag
        text = text.replace(tag, '')
    return text

# Remove punctuation
def remove_punc(text):
    # Filter out all punctuation chars
    new_text = ''.join([char for char in text if char not in string.punctuation])
    return new_text

# Lowercase
def lowercase(text):
    new_text = text.lower()
    return new_text

def clean_text(text):
    clean_input = remove_tags(text)
    clean_input = remove_punc(clean_input)
    clean_input = lowercase(clean_input)
    return clean_input

@app.post("/predict")
def predict(input: EmailInput):
    clean_input = clean_text(input.text)
    score = get_confidence_score(clean_input)
    label = get_label(score)
    return {"confidence": score, "label": label}
