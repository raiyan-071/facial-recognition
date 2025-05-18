from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from pymongo import MongoClient
from facial_recognition import load_known_faces, recognize_face
import face_recognition

app = Flask(__name__)

# MongoDB configuration
client = MongoClient('mongodb+srv://roshanrajmahato:EiQyv4qxQCSkX3M6@cluster0.90ynhbs.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['face_recognition_db']
collection = db['persons']

# Load known faces from MongoDB
known_face_encodings, known_face_names = load_known_faces()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    face_locations, face_names = recognize_face(frame, known_face_encodings, known_face_names)

    result = {'faces': []}
    for location, name in zip(face_locations, face_names):
        result['faces'].append({
            'name': name,
            'location': location
        })

    return jsonify(result)

@app.route('/register', methods=['POST'])
def register():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    name = request.form.get('name')
    if file.filename == '' or not name:
        return jsonify({'error': 'No selected file or name missing'})

    npimg = np.frombuffer(file.read(), np.uint8)
    image = face_recognition.load_image_file(npimg)
    encoding = face_recognition.face_encodings(image)

    if len(encoding) == 0:
        return jsonify({'error': 'No faces found in the image'})

    # Store the first encoding and name in MongoDB
    collection.insert_one({
        'name': name,
        'encoding': encoding[0].tolist()  # Convert numpy array to list
    })

    return jsonify({'success': f'{name} registered successfully'})

if __name__ == '__main__':
    app.run(debug=True)
