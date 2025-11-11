import os
import pandas as pd


def load_data(path="data/train.csv"):
    """Load dataset from CSV. If not present, create a tiny synthetic dataset for demo.

    Expected columns: question_text, model_answer, student_answer, score
    """
    if os.path.exists(path):
        return pd.read_csv(path)

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    rows = [
        {
            "question_text": "What is photosynthesis?",
            "model_answer": "Photosynthesis is the process by which plants convert light energy into chemical energy (glucose) using CO2 and water.",
            "student_answer": "Plants make food using light, CO2 and water.",
            "score": 4,
        },
        {
            "question_text": "What is photosynthesis?",
            "model_answer": "Photosynthesis is the process by which plants convert light energy into chemical energy (glucose) using CO2 and water.",
            "student_answer": "It is how plants convert sunlight into sugar.",
            "score": 4,
        },
        {
            "question_text": "Define acceleration.",
            "model_answer": "Acceleration is the rate of change of velocity per unit time.",
            "student_answer": "It is the change in velocity over time.",
            "score": 5,
        },
        {
            "question_text": "What causes tides?",
            "model_answer": "Tides are caused by the gravitational pull of the moon and the sun on Earth's oceans.",
            "student_answer": "The moon's gravity causes the ocean to bulge, making tides.",
            "score": 3,
        },
        {
            "question_text": "What causes tides?",
            "model_answer": "Tides are caused by the gravitational pull of the moon and the sun on Earth's oceans.",
            "student_answer": "Wind and boats cause tides.",
            "score": 0,
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


if __name__ == '__main__':
    print('Creating or loading data/train.csv')
    df = load_data()
    print(df.head())
