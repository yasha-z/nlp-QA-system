"""Retrain SBERT model with better scaling for low-score dataset"""
import sys
sys.path.append('..')

from asag.data import load_data
from asag.features import build_sbert_features, save_artifact
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, cohen_kappa_score
import numpy as np
import os

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def compute_qwk(y_true, y_pred, min_rating=0, max_rating=3):
    y_pred_round = np.clip(np.rint(y_pred), min_rating, max_rating).astype(int)
    return cohen_kappa_score(y_true.astype(int), y_pred_round, weights="quadratic")


def retrain_sbert_scaled():
    """Retrain SBERT with scaled targets to handle low-score dataset"""
    
    print("Loading data...")
    df = load_data('../data/train.csv')
    
    print(f"Dataset size: {len(df)}")
    print(f"Score distribution: {df['score'].value_counts().sort_index()}")
    
    from sentence_transformers import SentenceTransformer
    
    sbert_name = 'all-MiniLM-L6-v2'
    print(f"\nLoading SBERT: {sbert_name}")
    sbert = SentenceTransformer(sbert_name)
    
    student_texts = df['student_answer'].fillna('').tolist()
    model_texts = df['model_answer'].fillna('').tolist()
    y = df['score'].astype(float).values
    
    print("Building SBERT features...")
    encode_fn = lambda texts: sbert.encode(texts, show_progress_bar=True)
    X = build_sbert_features(encode_fn, student_texts, model_texts)
    
    print(f"Feature shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train with better LightGBM parameters optimized for this data
    if lgb is not None:
        print("\nTraining LightGBM with optimized parameters...")
        
        model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=8,
            num_leaves=50,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='rmse'
        )
    else:
        print("\nLightGBM not available, using Ridge...")
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=0.5)
        model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    qwk = compute_qwk(y_test, preds, min_rating=0, max_rating=3)
    mse = mean_squared_error(y_test, preds)
    
    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  QWK: {qwk:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  Prediction range: [{preds.min():.2f}, {preds.max():.2f}]")
    print(f"  Mean prediction: {preds.mean():.2f}")
    print(f"{'='*50}")
    
    # Save model
    model_dir = '../models'
    os.makedirs(model_dir, exist_ok=True)
    artifact = {'model': model, 'sbert_name': sbert_name}
    save_artifact(artifact, os.path.join(model_dir, 'sbert_model.joblib'))
    
    print(f"\n✓ Saved retrained SBERT model to {model_dir}")
    
    return model, qwk


if __name__ == '__main__':
    retrain_sbert_scaled()
