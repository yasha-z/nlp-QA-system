import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def fit_tfidf(corpus, max_features=5000, ngram_range=(1,2)):
    tf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = tf.fit_transform(corpus)
    return tf, X


def transform_tfidf(tf, corpus):
    return tf.transform(corpus)


def build_sbert_features(encode_fn, student_texts, model_texts):
    # encode_fn should accept a list[str] and return a 2D numpy array
    student_emb = np.array(encode_fn(student_texts))
    model_emb = np.array(encode_fn(model_texts))
    cos_sim = np.array([cosine_similarity([student_emb[i]], [model_emb[i]])[0][0] for i in range(len(student_emb))])
    X = np.hstack([student_emb, model_emb, (student_emb - model_emb), cos_sim.reshape(-1,1)])
    return X


def save_artifact(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path):
    return joblib.load(path)
