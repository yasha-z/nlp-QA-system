# NLP-QA System: Automated Short Answer Grading

A Flask-based web application for automated grading of short answer questions using Natural Language Processing and machine learning models.

## Project Overview

This system implements **Automatic Short Answer Grading (ASAG)** using two complementary machine learning approaches:
- **TF-IDF + Ridge Regression**: Fast, interpretable baseline model
- **Sentence-BERT (SBERT)**: Deep learning semantic similarity model

The system supports both individual answer grading and batch processing of multiple student submissions, with a user-friendly web interface and teacher dashboard.

## Group Members

| Name | ID |
|------|-----|
| Yumna Irfan | CT-22004 |
| Aleeshba | CT-22005 |
| Hadiya Kashif | CT-22008 |
| Yasha Ali | CT-22010 |

## Features

### Core Functionality
- **Individual Grading**: Grade single student answers in real-time with immediate feedback
- **Batch Grading**: Process multiple student files (CSV/Excel) simultaneously
- **Multi-Model Predictions**: Compare predictions from TF-IDF, SBERT, and ensemble methods
- **Semantic Similarity Scoring**: Calculate cosine similarity between answers using SBERT embeddings
- **Score Mapping**: Map raw model predictions to discrete scores (0-3)

### User Interface
- **Student Portal** (`/`): Simple interface for submitting answers and viewing scores
- **Teacher Dashboard** (`/teacher`): Batch processing interface for managing multiple student submissions
- **Results Management**: Download and track grading results

### File Support
- CSV files
- Excel files (.xlsx, .xls)
- Maximum file size: 16MB

## Technical Stack

### Dependencies
```
pandas          - Data manipulation and analysis
numpy           - Numerical computing
scikit-learn    - Machine learning (TF-IDF, Ridge Regression)
sentence-transformers - SBERT models for semantic similarity
lightgbm        - Gradient boosting (optional)
joblib          - Model serialization
Flask           - Web framework
openpyxl        - Excel file handling
xlrd            - Excel file reading
```

### Models
- **Ridge Regression Model** (`models/ridge_model.joblib`): Trained on TF-IDF features
- **SBERT Model** (`models/sbert_model.joblib`): Pre-trained Sentence-BERT model
- **TF-IDF Vectorizer** (`models/tfidf_vectorizer.joblib`): Fitted vectorizer for text transformation

## Project Structure

```
nlp-QA-system/
├── NLP/
│   ├── app.py                 # Flask application & API endpoints
│   ├── requirements.txt       # Python dependencies
│   ├── asag/                  # Core ASAG package
│   │   ├── __init__.py
│   │   ├── batch.py          # Batch processing module
│   │   ├── data.py           # Data loading utilities
│   │   ├── features.py       # Feature extraction
│   │   ├── predict.py        # Prediction functions
│   │   ├── train.py          # Model training
│   │   ├── tuning.py         # Hyperparameter tuning
│   │   ├── improve.py        # Model improvements
│   │   ├── improve_sbert.py  # SBERT-specific improvements
│   ├── models/               # Serialized models
│   │   ├── ridge_model.joblib
│   │   ├── sbert_model.joblib
│   │   └── tfidf_vectorizer.joblib
│   ├── data/                 # Training data
│   │   ├── train.csv
│   │   └── asap/
│   ├── scripts/              # Utility scripts
│   │   ├── normalize_asap.py
│   │   └── retrain_sbert.py
│   ├── asap-sas/            # ASAP dataset files
│   ├── static/              # Web UI files
│   │   ├── index.html       # Student portal
│   │   ├── teacher.html     # Teacher dashboard
│   │   └── styles.css       # Styling
│   ├── results/             # Generated grading results
│   └── uploads/             # Temporary uploaded files
├── README.md
└── student*.csv             # Sample student submission files
```

## Installation & Setup

### Prerequisites
- Python 3.7+
- pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yasha-z/nlp-QA-system.git
   cd nlp-QA-system/NLP
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model files exist**
   Ensure the following files are present in the `models/` directory:
   - `ridge_model.joblib`
   - `sbert_model.joblib`
   - `tfidf_vectorizer.joblib`

## Usage

### Starting the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### API Endpoints

#### 1. **Health Check**
```
GET /health
```
Returns the status and loaded models.

#### 2. **Individual Answer Grading**
```
POST /predict
Content-Type: application/json

{
  "student_answer": "Your answer here",
  "model_answer": "Expected answer here",
  "mode": "both"  # Options: "tfidf", "sbert", "both"
}
```

**Response:**
```json
{
  "ok": true,
  "result": {
    "tfidf_pred": 2.5,
    "tfidf_pred_mapped": 3,
    "sbert_pred": 2.1,
    "sbert_pred_mapped": 2,
    "sbert_cosine": 0.87,
    "cosine_based_score": 3,
    "ensemble_score": 2,
    "better_ensemble": 2
  }
}
```

#### 3. **Batch Grading**
```
POST /batch-grade
Content-Type: multipart/form-data

- files[]: CSV/Excel files to process
- model_answers: JSON string with model answers
```

**Model Answers Format:**
```json
{
  "Q1": "Expected answer for Q1",
  "Q2": "Expected answer for Q2",
  ...
}
```

#### 4. **Download Results**
```
GET /download/<filename>
```

#### 5. **List Results**
```
GET /results
```

### Web Interfaces

#### Teacher Dashboard (`/teacher`)
- Upload multiple student files
- Input model answers
- Process batch grading
- Download individual and summary results

## Scoring System

The system uses a **0-3 scale** scoring system:
- **0**: Poor match (incorrect or minimal relevance)
- **1**: Fair match (partial understanding)
- **2**: Good match (mostly correct with minor issues)
- **3**: Excellent match (correct and complete answer)

### Score Mapping
Raw model predictions are mapped to discrete scores using the following thresholds:
```python
raw_pred < 0.3   → score 0
0.3 ≤ raw_pred < 0.9   → score 1
0.9 ≤ raw_pred < 1.6   → score 2
raw_pred ≥ 1.6   → score 3
```

## Model Predictions

### TF-IDF + Ridge Regression
- **Speed**: Fast
- **Interpretability**: High
- **Features**: Bag-of-words based
- **Weight in Ensemble**: 60%

### Sentence-BERT (SBERT)
- **Semantic Understanding**: Captures semantic similarity
- **Cosine-Based Score**: Alternative scoring using embeddings
- **Weight in Ensemble**: 40% (cosine-based)

### Ensemble Methods
1. **Weighted Average**: Combines TF-IDF (60%) + SBERT (40%)
2. **Better Ensemble**: Combines TF-IDF (50%) + Cosine-based (50%)

## Batch Processing

### Input File Format

**CSV/Excel expected structure:**
```
Name          | Roll Number | Q1                  | Q2    | ...
John Doe      | 001         | Carbon is element 6 | 14.5  | ...
Jane Smith    | 002         | C is atomic #6      | 14.5  | ...
```

### Output Files

The batch grading generates:
1. **Summary File** (`batch_results_<timestamp>.xlsx`): All students with scores
2. **Individual Files** (optional): Separate files per student

**Output columns:**
- Student name and roll number
- Per-question raw predictions
- Per-question mapped scores
- Average score
- Metadata (timestamp, models used)

## Training Models

To retrain models with new data:

```bash
python -m asag.train
```

To run hyperparameter tuning:

```bash
python -m asag.tuning
```

To improve SBERT model:

```bash
python scripts/retrain_sbert.py
```

## Sample Data

Sample student submissions are provided in:
- `student1_alice.csv`
- `student2_bob.csv`
- `student3_charlie.csv`
- `test_batch_all_students.csv`

## Dataset References

- **ASAP-SAS Dataset**: Included in `asap-sas/` directory
  - Training data: `train.tsv`, `train_rel_2.tsv`
  - Test data: `test.csv`
  - Leaderboard data and benchmarks

## Configuration

### File Upload Limits
- Maximum file size: 16MB (configured in `app.py`)
- Allowed formats: CSV, XLSX, XLS

### Model Directories
- Models loaded from: `models/`
- Results saved to: `results/`
- Uploads stored in: `uploads/` (temporary)

## Performance Metrics

The system tracks:
- Per-question prediction accuracy
- Score distribution statistics
- Cosine similarity metrics
- Model agreement rates

## Error Handling

The system includes comprehensive error handling for:
- Missing or corrupted model files
- Invalid file formats
- Malformed input data
- API request errors

Error responses include detailed traceback information for debugging.

## Future Enhancements

- Support for long-answer grading
- Fine-tuning on domain-specific datasets
- Integration with Learning Management Systems (LMS)
- Real-time model retraining with user feedback
- Multi-language support
- Advanced ensemble methods (stacking, voting)

---

**Last Updated**: November 15, 2025
