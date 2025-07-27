import os
import joblib
import time
import pandas as pd
from tqdm import tqdm

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


# Model definitions
MODELS = {
    "naive_bayes": MultinomialNB(),
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "linear_svc": LinearSVC(max_iter=1000)
}


def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df["text"]
    y = df["generated"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_pipeline(model_name):
    model = MODELS[model_name]

    # Use CountVectorizer for Naive Bayes; TF-IDF for others
    if model_name == "naive_bayes":
        vectorizer = CountVectorizer(ngram_range=(1, 2))
    else:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    return Pipeline([
        ('vectorizer', vectorizer),
        ('clf', model)
    ])


def train_and_evaluate_model(model_name, X_train, X_test, y_train, y_test, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.joblib")

    retrained = False
    start_time = time.time()

    if os.path.exists(model_path):
        print(f"[INFO] Skipping training for '{model_name}' (model exists). Loading model...")
        pipeline = joblib.load(model_path)
    else:
        print(f"[INFO] Training model: {model_name}")
        pipeline = build_pipeline(model_name)

        for _ in tqdm(range(1), desc=f"Training {model_name}"):
            pipeline.fit(X_train, y_train)

        joblib.dump(pipeline, model_path)
        print(f"[INFO] Saved model to {model_path}")
        retrained = True

    latency = time.time() - start_time

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n[INFO] {model_name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Latency: {latency:.2f} seconds")

    return {
        "model": model_name,
        "accuracy": acc,
        "f1_score": f1,
        "latency_sec": round(latency, 2),
        "retrained": retrained
    }


def main():
    # Absolute path to current script directory
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths relative to script
    data_path = os.path.join(this_dir, "..", "data", "cleaned", "AI_Human_cleaned.csv")
    model_dir = os.path.join(this_dir, "..", "models")
    results_dir = os.path.join(this_dir, "..", "results")
    results_path = os.path.join(results_dir, "model_results.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"[INFO] Loading dataset from {data_path}")
    X_train, X_test, y_train, y_test = load_data(data_path)

    results = []

    for model_name in MODELS:
        result = train_and_evaluate_model(
            model_name,
            X_train, X_test,
            y_train, y_test,
            model_dir=model_dir
        )
        results.append(result)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)
    print(f"\n[INFO] Model evaluation results saved to {results_path}")


if __name__ == "__main__":
    main()
