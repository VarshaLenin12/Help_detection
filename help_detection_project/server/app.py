from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, session
import os
import uuid
from datetime import datetime
import json

def save_metadata(metadata):
    with open('metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)

def load_metadata():
    if os.path.exists('metadata.json'):
        with open('metadata.json', 'r') as f:
            return json.load(f)
    return {}


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a strong secret key

# Directory to save received files
UPLOAD_FOLDER = os.path.join('..', 'client', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dummy credentials (use environment variables or a secure method for real applications)
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = '123'

def generate_unique_filename(file_extension):
    """Generate a unique filename with a timestamp and UUID."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex
    return f"{timestamp}_{unique_id}{file_extension}"


def check_authentication():
    """Check if the user is authenticated."""
    return 'logged_in' in session and session['logged_in']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')  # Ensure this matches the form field name
        password = request.form.get('password')  # Ensure this matches the form field name
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('admin'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/admin')
def admin():
    if not check_authentication():
        return redirect(url_for('login'))
    
    # List files excluding .txt files
    files = [f for f in os.listdir(UPLOAD_FOLDER) if not f.endswith('.txt')]
    
    # Get file metadata (example: modification time)
    files_with_metadata = []
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file)
        file_stat = os.stat(file_path)
        files_with_metadata.append({
            'filename': file,
            'mtime': file_stat.st_mtime  # Modification time
        })

    # Sort files by modification time (latest first)
    sorted_files = sorted(files_with_metadata, key=lambda x: x['mtime'], reverse=True)

    # Prepare metadata dictionary (example)
    metadata = load_metadata()
    metadata_dict = {file['filename']: metadata.get(file['filename'], {'file_type': 'unknown', 'location': 'unknown', 'timestamp': 'unknown'}) for file in sorted_files}

    # Debugging: print metadata to console
    print("Metadata:", metadata_dict)
    
    return render_template('admin.html', files=[file['filename'] for file in sorted_files], metadata=metadata_dict)


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Get the file and metadata from the request
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file part'}), 400

        location = request.form.get('location', 'No location provided')
        file_type = request.form.get('type', 'unknown')

        # Generate a unique filename
        unique_filename = generate_unique_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

        # Save the file
        file.save(file_path)

        # Save metadata
        metadata = load_metadata()
        metadata[unique_filename] = {
            'location': location,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'file_type': file_type  # Save file_type here
        }
        save_metadata(metadata)

        return jsonify({"message": "File and metadata received"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
