import os
from flask import Flask, request, jsonify, send_from_directory, send_file
import traceback
from werkzeug.utils import secure_filename
import json

from asag.predict import load_artifacts, predict_tfidf_ridge, predict_sbert
from asag.batch import process_batch_grading

app = Flask(__name__, static_url_path='', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

ARTS = load_artifacts('models')

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'models_loaded': list(ARTS.keys())})


BASE_DIR = os.path.abspath(os.path.dirname(__file__))


@app.route('/')
def index():
    # Serve the single-file web UI using absolute path to avoid cwd issues
    static_dir = os.path.join(BASE_DIR, 'static')
    return send_from_directory(static_dir, 'index.html')


@app.route('/teacher')
def teacher_dashboard():
    # Serve the teacher dashboard
    static_dir = os.path.join(BASE_DIR, 'static')
    return send_from_directory(static_dir, 'teacher.html')


@app.route('/batch-grade', methods=['POST'])
def batch_grade():
    """Handle batch grading file upload"""
    try:
        # Check if files were uploaded (support multiple files)
        if 'files[]' not in request.files:
            # Fallback to single file upload for backward compatibility
            if 'file' not in request.files:
                return jsonify({'ok': False, 'error': 'No file uploaded'}), 400
            files = [request.files['file']]
        else:
            files = request.files.getlist('files[]')
        
        if not files or len(files) == 0:
            return jsonify({'ok': False, 'error': 'No files selected'}), 400
        
        # Save all uploaded files
        saved_files = []
        for file in files:
            if file.filename == '':
                continue
            
            if not allowed_file(file.filename):
                return jsonify({'ok': False, 'error': f'Invalid file type: {file.filename}. Please upload CSV or Excel files'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            saved_files.append(filepath)
        
        if len(saved_files) == 0:
            return jsonify({'ok': False, 'error': 'No valid files uploaded'}), 400
        
        # Get model answers from form data
        model_answers_json = request.form.get('model_answers')
        if not model_answers_json:
            return jsonify({'ok': False, 'error': 'No model answers provided'}), 400
        
        model_answers = json.loads(model_answers_json)
        
        # Process batch grading (handles both single and multiple files)
        result = process_batch_grading(
            saved_files if len(saved_files) > 1 else saved_files[0],
            model_answers,
            output_dir=app.config['RESULTS_FOLDER']
        )
        
        # Clean up uploaded files
        for filepath in saved_files:
            if os.path.exists(filepath):
                os.remove(filepath)
        
        return jsonify({
            'ok': True,
            'message': f'Successfully graded {result["total_students"]} students from {len(saved_files)} file(s)',
            'summary_file': os.path.basename(result['summary_file']),
            'individual_files': [os.path.basename(f) for f in result['individual_files']],
            'total_students': result['total_students'],
            'results': result['results'],
            'metrics': result['metrics']  # Add metrics
        })
        
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Download generated result files"""
    try:
        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/results')
def list_results():
    """List all generated result files"""
    try:
        files = []
        if os.path.exists(app.config['RESULTS_FOLDER']):
            for filename in os.listdir(app.config['RESULTS_FOLDER']):
                if filename.endswith(('.xlsx', '.csv')):
                    filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
                    files.append({
                        'filename': filename,
                        'size': os.path.getsize(filepath),
                        'created': os.path.getctime(filepath)
                    })
        return jsonify({'ok': True, 'files': files})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.get_json(force=True)
        student = payload.get('student_answer', '')
        model_answer = payload.get('model_answer', '')
        mode = payload.get('mode', 'sbert')
        result = {}
        
        tfidf_pred = None
        sbert_pred = None
        
        if mode in ('tfidf', 'both'):
            try:
                raw_p, mapped_p = predict_tfidf_ridge(ARTS, student, model_answer)
                result['tfidf_pred'] = raw_p
                result['tfidf_pred_mapped'] = mapped_p
                result['tfidf_pred_rounded'] = int(round(raw_p))
                tfidf_pred = mapped_p
            except Exception as e:
                result['tfidf_error'] = str(e)
                
        if mode in ('sbert', 'both'):
            try:
                raw_p, mapped_p, cos = predict_sbert(ARTS, student, model_answer)
                result['sbert_pred'] = raw_p
                result['sbert_pred_mapped'] = mapped_p
                result['sbert_pred_rounded'] = int(round(raw_p))
                result['sbert_cosine'] = cos
                sbert_pred = mapped_p
            except Exception as e:
                result['sbert_error'] = str(e)
        
        # Add ensemble prediction if both models ran
        if tfidf_pred is not None and sbert_pred is not None:
            # Weighted average: TF-IDF gets 60%, SBERT gets 40% (SBERT model is undertrained)
            ensemble_score = int(round(0.6 * tfidf_pred + 0.4 * sbert_pred))
            result['ensemble_score'] = ensemble_score
        
        # Add cosine-based score as alternative (more reliable than undertrained SBERT)
        if 'sbert_cosine' in result:
            cos = result['sbert_cosine']
            # Map cosine similarity directly to scores (0-3 scale, not 0-4!)
            if cos >= 0.85:
                cos_score = 3  # Excellent match
            elif cos >= 0.70:
                cos_score = 2  # Good match
            elif cos >= 0.50:
                cos_score = 1  # Fair match
            else:
                cos_score = 0  # Poor match
            result['cosine_based_score'] = cos_score
            
            # Better ensemble using cosine score instead of broken SBERT
            if tfidf_pred is not None:
                better_ensemble = int(round(0.5 * tfidf_pred + 0.5 * cos_score))
                result['better_ensemble'] = better_ensemble
            
        return jsonify({'ok': True, 'result': result})
    except Exception:
        return jsonify({'ok': False, 'error': traceback.format_exc()}), 500


if __name__ == '__main__':
    # Run without the debug reloader to avoid automatic restarts when started
    # from a background terminal in this environment.
    app.run(host='0.0.0.0', port=5000, debug=False)
