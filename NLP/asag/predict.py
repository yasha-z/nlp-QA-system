import os
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_artifacts(model_dir='models'):
    arts = {}
    tf_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')
    ridge_path = os.path.join(model_dir, 'ridge_model.joblib')
    sbert_path = os.path.join(model_dir, 'sbert_model.joblib')
    if os.path.exists(tf_path):
        arts['tf'] = joblib.load(tf_path)
    if os.path.exists(ridge_path):
        arts['ridge'] = joblib.load(ridge_path)
    if os.path.exists(sbert_path):
        arts['sbert_art'] = joblib.load(sbert_path)
    return arts


def improved_score_mapping(raw_pred):
    """Better mapping from continuous predictions to discrete scores
    
    Training data has scores 0-3 ONLY (no score 4!)
    Distribution: 39% score 0, 33% score 1, 24% score 2, 4% score 3
    Mean score: 0.93
    
    UPDATED: More lenient thresholds to reduce false negatives
    """
    # Map predictions to 0-3 range with RELAXED boundaries
    if raw_pred < 0.3:       # Very poor (was < 0)
        return 0
    elif raw_pred < 0.9:     # Below average (was <= 0.5)
        return 1
    elif raw_pred < 1.6:     # Good (was <= 1.0)
        return 2
    else:                    # Excellent (was <= 1.8)
        return 3


def predict_tfidf_ridge(arts, student_answer, model_answer):
    if 'tf' not in arts or 'ridge' not in arts:
        raise RuntimeError('TF-IDF or Ridge artifact missing; run training first')
    tf = arts['tf']
    ridge = arts['ridge']
    text = (student_answer or '') + ' ' + (model_answer or '')
    X = tf.transform([text])
    raw_pred = float(ridge.predict(X)[0])
    
    # Apply improved score mapping
    mapped_score = improved_score_mapping(raw_pred)
    
    return raw_pred, mapped_score


def predict_sbert(arts, student_answer, model_answer):
    if 'sbert_art' not in arts:
        raise RuntimeError('SBERT artifact missing; run training first')
    sbert_art = arts['sbert_art']
    reg = sbert_art['model']
    sbert_name = sbert_art.get('sbert_name', 'all-MiniLM-L6-v2')
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(sbert_name)
    
    # Use original feature format to match trained model
    stu_emb = sbert.encode([student_answer or ''])[0]
    mod_emb = sbert.encode([model_answer or ''])[0]
    cos_sim = cosine_similarity([stu_emb], [mod_emb])[0][0]
    X = np.hstack([stu_emb, mod_emb, (stu_emb - mod_emb), [cos_sim]])
    
    raw_pred = float(reg.predict([X])[0])
    
    # Apply improved score mapping
    mapped_score = improved_score_mapping(raw_pred)
    
    return raw_pred, mapped_score, float(cos_sim)


if __name__ == '__main__':
    print('Artifacts loader test:')
    arts = load_artifacts()
    print('Found artifacts:', list(arts.keys()))
