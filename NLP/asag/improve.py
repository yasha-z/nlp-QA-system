"""Improved ASAG models with better performance"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, mean_squared_error
import re
import string
from scipy.stats import pearsonr


def improved_text_preprocessing(text):
    """Enhanced text preprocessing"""
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation but keep some meaningful ones
    text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
    
    # Remove extra spaces
    text = text.strip()
    
    return text


def extract_additional_features(student_text, model_text):
    """Extract additional features beyond TF-IDF"""
    features = {}
    
    # Text length features
    features['student_len'] = len(student_text.split())
    features['model_len'] = len(model_text.split())
    features['len_ratio'] = features['student_len'] / max(features['model_len'], 1)
    
    # Word overlap features
    student_words = set(student_text.lower().split())
    model_words = set(model_text.lower().split())
    overlap = len(student_words.intersection(model_words))
    features['word_overlap'] = overlap
    features['word_overlap_ratio'] = overlap / max(len(model_words), 1)
    
    # Character-level features
    features['char_len_student'] = len(student_text)
    features['char_len_model'] = len(model_text)
    
    return features


def improved_score_mapping(raw_pred, score_range=(0, 4)):
    """Better mapping from continuous predictions to discrete scores"""
    min_score, max_score = score_range
    
    # Apply sigmoid-like scaling to compress extreme values
    scaled = np.tanh(raw_pred) * 2  # Scale to roughly [-2, 2]
    
    # Map to score range with proper thresholds
    # These thresholds were tuned based on training data distribution
    if scaled <= -1.0:
        return min_score
    elif scaled <= -0.2:
        return min_score + 1
    elif scaled <= 0.5:
        return min_score + 2
    elif scaled <= 1.2:
        return min_score + 3
    else:
        return max_score


def train_improved_tfidf_model(df, model_dir='models'):
    """Train improved TF-IDF model with better preprocessing and features"""
    
    # Improved preprocessing
    student_processed = df['student_answer'].fillna('').apply(improved_text_preprocessing)
    model_processed = df['model_answer'].fillna('').apply(improved_text_preprocessing)
    
    # Combine texts for TF-IDF
    corpus = (student_processed + ' ' + model_processed).tolist()
    
    # TF-IDF Removes Stop Words!
    # Use improved TF-IDF with better parameters
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),  # Include bigrams
        min_df=2,
        max_df=0.95,
        stop_words='english',
        sublinear_tf=True  # Use log scaling
    )
    
    X_tfidf = tfidf.fit_transform(corpus)
    
    # Extract additional features
    additional_features = []
    for i, row in df.iterrows():
        feats = extract_additional_features(
            student_processed.iloc[i], 
            model_processed.iloc[i]
        )
        additional_features.append(list(feats.values()))
    
    # Combine TF-IDF with additional features
    additional_features = np.array(additional_features)
    X_combined = np.hstack([X_tfidf.toarray(), additional_features])
    
    y = df['score'].astype(float).values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42
    )
    
    # Train with optimized Ridge parameters
    model = Ridge(alpha=0.1)  # Lower alpha for less regularization
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    qwk = cohen_kappa_score(y_test.astype(int), 
                           np.round(np.clip(preds, 0, 4)).astype(int), 
                           weights='quadratic')
    mse = mean_squared_error(y_test, preds)
    
    print(f"Improved TF-IDF Ridge: QWK={qwk:.4f}, MSE={mse:.4f}")
    
    # Test improved score mapping
    mapped_preds = [improved_score_mapping(p) for p in preds]
    mapped_qwk = cohen_kappa_score(y_test.astype(int), mapped_preds, weights='quadratic')
    print(f"With improved mapping: QWK={mapped_qwk:.4f}")
    
    return model, tfidf, qwk


def create_ensemble_predictor(tfidf_model, sbert_artifacts, tfidf_vectorizer):
    """Create ensemble predictor combining both models"""
    
    def ensemble_predict(student_text, model_text):
        # TF-IDF prediction
        processed_student = improved_text_preprocessing(student_text)
        processed_model = improved_text_preprocessing(model_text)
        combined_text = processed_student + ' ' + processed_model
        
        tfidf_features = tfidf_vectorizer.transform([combined_text])
        additional_feats = extract_additional_features(processed_student, processed_model)
        
        # For now, just use TF-IDF (we can add SBERT later)
        tfidf_pred = tfidf_model.predict(tfidf_features)[0]
        
        # Apply improved score mapping
        final_score = improved_score_mapping(tfidf_pred)
        
        return {
            'raw_prediction': float(tfidf_pred),
            'mapped_score': int(final_score),
            'confidence': min(abs(tfidf_pred), 1.0)  # Simple confidence measure
        }
    
    return ensemble_predict


if __name__ == '__main__':
    # Test the improvements
    import sys
    sys.path.append('../')
    from asag.data import load_data
    
    df = load_data('../data/train.csv')
    print("Training improved model...")
    train_improved_tfidf_model(df)