# AI-Human-Writting-Detection

Detect whether email text is **AI-generated** or **human-written**.  
This project includes:
- A FastAPI backend that loads a trained classifier (`.joblib`) and exposes `/predict`.
- A local UI (`local_ui.html`) for copy/paste testing in any browser.
- (WIP) A full Outlook add-in implementation (HTML/JS/manifest) that calls the same API.

---

## Why?

Generative AI has made it easy to produce polished, convincing emails at scale. While this has improved user efficiency (converting "informal" emails into "ready-to-send" versions), this has inadvertidely helped a) scams look cleaner & more credible, b) generated spams skipping past older keyword-only filters, and c) impacted admins, analysts, or educators who may want a signal for potential/likely AI involvement. Importantly, this tool is not a "proof" - use results with judgement and context. 

---

## Quick-start (using pre-trained models)
1. Run API
```
uvicorn src.api:app --reload
```
- API base: http://127.0.0.1:8000 (append /docs for visually inspecting and being able to pass through sanity tests through -Curl)

2. Open the Local UI
```
cd src
python -m http.server 3000
```
- Then open http://localhost:3000/local_ui.html
- Can then paste text, click Predict, and view the confidence + label output.

3. Choose a different model file

Default in api.py:
```
MODEL_PATH = os.getenv("MODEL_PATH", "models/naive_bayes.joblib")
```

4. (Optional) Remote model download

If hosting .joblib remotely, set:
```
export MODEL_URL="https://your-url/model.joblib"
export MODEL_PATH="models/naive_bayes.joblib"
```
The app will download it automatically if missing.

## Train your own models (produces .joblib files)
1. Prepare data

Place your raw CSV training data in data/raw/AI_Human.csv with the following set-up:
```
text – the content
generated – 0 for human, 1 for AI
```

2. Clean data
```
python src/data_cleaning.py
```
- Output: data/cleaned/AI_Human_cleaned.csv

3. Train models
```
python src/model.py
```
- Trains multiple models (Naive Bayes, Logistic Regression, Linear SVC, Random Forest, Gradient Boosting) saved at models/*.joblib
- Writes performance results to results/model_results.csv

## Outlook add-in (WIP)

The outlook_addin/ folder contains:
```
editor.html – task pane for reading messages

editor.js – retrieves message body and calls /predict

permissions.xml – manifest
```

To use:

1. Run the API as above.

2. Serve the Outlook add-in files (or adjust manifest URL).

3. Sideload the add-in into Outlook (per Microsoft docs).

4. Open an email → the add-in analyzes it through /predict.
