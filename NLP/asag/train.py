"""Training CLI for ASAG prototype"""
import argparse
import os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, cohen_kappa_score

try:
    import lightgbm as lgb
except Exception:
    lgb = None

from .data import load_data
from .features import fit_tfidf, build_sbert_features, save_artifact


def compute_qwk(y_true, y_pred, min_rating=None, max_rating=None):
    if min_rating is None:
        min_rating = int(np.min(y_true))
    if max_rating is None:
        max_rating = int(np.max(y_true))
    y_pred_round = np.clip(np.rint(y_pred), min_rating, max_rating).astype(int)
    return cohen_kappa_score(y_true.astype(int), y_pred_round, weights="quadratic")


def train_baseline(df, model_dir='models'):
    corpus = (df['student_answer'].fillna('') + ' ' + df['model_answer'].fillna('')).tolist()
    y = df['score'].astype(float).values
    tf, X = fit_tfidf(corpus)
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Ridge(alpha=1.0)
    model.fit(Xtr, ytr)
    preds = model.predict(Xv)
    qwk = compute_qwk(yv, preds)
    mse = mean_squared_error(yv, preds)
    print(f"TF-IDF Ridge: QWK={qwk:.4f}, MSE={mse:.4f}")
    os.makedirs(model_dir, exist_ok=True)
    save_artifact(tf, os.path.join(model_dir, 'tfidf_vectorizer.joblib'))
    save_artifact(model, os.path.join(model_dir, 'ridge_model.joblib'))


def train_sbert(df, model_dir='models', sbert_name='all-MiniLM-L6-v2'):
    # Import inside function to avoid heavy import on module load
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(sbert_name)
    student_texts = df['student_answer'].fillna('').tolist()
    model_texts = df['model_answer'].fillna('').tolist()
    y = df['score'].astype(float).values
    encode_fn = lambda texts: sbert.encode(texts, show_progress_bar=False)
    X = build_sbert_features(encode_fn, student_texts, model_texts)
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42)
    if lgb is not None:
        reg = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05)
    else:
        reg = Ridge(alpha=1.0)
    reg.fit(Xtr, ytr)
    preds = reg.predict(Xv)
    qwk = compute_qwk(yv, preds)
    mse = mean_squared_error(yv, preds)
    print(f"SBERT + {reg.__class__.__name__}: QWK={qwk:.4f}, MSE={mse:.4f}")
    os.makedirs(model_dir, exist_ok=True)
    save_artifact({'model': reg, 'sbert_name': sbert_name}, os.path.join(model_dir, 'sbert_model.joblib'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/train.csv')
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--train-baseline', action='store_true')
    parser.add_argument('--train-sbert', action='store_true')
    args = parser.parse_args()
    df = load_data(args.data)
    if args.train_baseline:
        train_baseline(df, model_dir=args.model_dir)
    if args.train_sbert:
        train_sbert(df, model_dir=args.model_dir)


if __name__ == '__main__':
    main()
