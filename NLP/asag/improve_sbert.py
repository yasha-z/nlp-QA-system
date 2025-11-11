"""Improved SBERT training with better hyperparameters"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, cohen_kappa_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import joblib
import os

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def compute_qwk(y_true, y_pred, min_rating=0, max_rating=4):
    """Compute Quadratic Weighted Kappa"""
    y_pred_round = np.clip(np.rint(y_pred), min_rating, max_rating).astype(int)
    return cohen_kappa_score(y_true.astype(int), y_pred_round, weights="quadratic")


def build_improved_sbert_features(encode_fn, student_texts, model_texts):
    """Build enhanced SBERT features with additional linguistic features"""
    
    # Get embeddings
    student_embs = encode_fn(student_texts)
    model_embs = encode_fn(model_texts)
    
    # Basic features: embeddings + difference + cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    cos_sims = []
    for i in range(len(student_embs)):
        cos_sim = cosine_similarity([student_embs[i]], [model_embs[i]])[0][0]
        cos_sims.append(cos_sim)
    
    cos_sims = np.array(cos_sims).reshape(-1, 1)
    differences = student_embs - model_embs
    
    # Additional features
    additional_features = []
    for i in range(len(student_texts)):
        student_text = student_texts[i] or ''
        model_text = model_texts[i] or ''
        
        # Text length features
        student_words = len(student_text.split())
        model_words = len(model_text.split())
        
        # Word overlap features
        student_word_set = set(student_text.lower().split())
        model_word_set = set(model_text.lower().split())
        overlap = len(student_word_set.intersection(model_word_set))
        
        features = [
            student_words,  # Student answer length
            model_words,    # Model answer length
            student_words / max(model_words, 1),  # Length ratio
            overlap,        # Word overlap count
            overlap / max(len(model_word_set), 1),  # Word overlap ratio
            len(student_text),  # Character length
            abs(student_words - model_words),  # Length difference
        ]
        additional_features.append(features)
    
    additional_features = np.array(additional_features)
    
    # Combine all features
    X = np.hstack([student_embs, model_embs, differences, cos_sims, additional_features])
    return X


def train_improved_sbert(df, model_dir='models', sbert_name='all-MiniLM-L6-v2'):
    """Train improved SBERT model with better features and hyperparameters"""
    
    from sentence_transformers import SentenceTransformer
    
    print(f"Loading SBERT model: {sbert_name}")
    sbert = SentenceTransformer(sbert_name)
    
    student_texts = df['student_answer'].fillna('').tolist()
    model_texts = df['model_answer'].fillna('').tolist()
    y = df['score'].astype(float).values
    
    print("Building enhanced SBERT features...")
    encode_fn = lambda texts: sbert.encode(texts, show_progress_bar=True)
    X = build_improved_sbert_features(encode_fn, student_texts, model_texts)
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Try multiple models with different hyperparameters
    models_to_try = []
    
    # LightGBM with better parameters
    if lgb is not None:
        models_to_try.append(('LightGBM_v1', lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=6,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            random_state=42
        )))
        
        models_to_try.append(('LightGBM_v2', lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            random_state=42
        )))
    
    # Ridge with different alphas
    models_to_try.extend([
        ('Ridge_0.01', Ridge(alpha=0.01)),
        ('Ridge_0.1', Ridge(alpha=0.1)),
        ('Ridge_1.0', Ridge(alpha=1.0)),
        ('Ridge_10.0', Ridge(alpha=10.0)),
    ])
    
    # Random Forest
    models_to_try.append(('RandomForest', RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )))
    
    best_qwk = -1
    best_model = None
    best_name = None
    
    print("\nTesting different models:")
    print("=" * 50)
    
    for name, model in models_to_try:
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            qwk = compute_qwk(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            
            print(f"{name:15s}: QWK={qwk:.4f}, MSE={mse:.4f}")
            
            if qwk > best_qwk:
                best_qwk = qwk
                best_model = model
                best_name = name
                
        except Exception as e:
            print(f"{name:15s}: ERROR - {str(e)}")
    
    print("=" * 50)
    print(f"Best model: {best_name} with QWK={best_qwk:.4f}")
    
    # Save the best model
    os.makedirs(model_dir, exist_ok=True)
    artifact = {'model': best_model, 'sbert_name': sbert_name}
    joblib.dump(artifact, os.path.join(model_dir, 'sbert_model.joblib'))
    
    print(f"Saved improved SBERT model to {model_dir}")
    return best_model, best_qwk


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from asag.data import load_data
    
    df = load_data('../data/train.csv')
    print("Training improved SBERT model...")
    train_improved_sbert(df)