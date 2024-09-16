import os
import cv2
import numpy as np
import sqlite3
import base64
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from PIL import Image
import face_recognition
import mediapipe as mp
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.1)

DATABASE = 'database.db'


def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS FACES (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            BASE64_IMAGE TEXT NOT NULL,
            EMBEDDING BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


init_db()


def get_face_embedding(image_data):
    image = face_recognition.load_image_file(image_data)
    face_encodings = face_recognition.face_encodings(image)
    if face_encodings:
        return face_encodings[0]
    return None


def compare_face(image_embedding):
    global percentage
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT BASE64_IMAGE, EMBEDDING FROM FACES')
    rows = cursor.fetchall()
    conn.close()

    for row in rows:
        db_base64_image = row[0]
        db_embedding = np.frombuffer(row[1], dtype=np.float64)
        distance = np.linalg.norm(image_embedding - db_embedding)
        percentage = (1 - distance) * 100
        if distance < 0.5:
            return db_base64_image
    return None


@app.route('/upload', methods=['POST'])
def upload_file():
    data = request.json
    base64_str = data.get('base64')
    if not base64_str:
        return jsonify({'error': 'No Base64 data provided'}), 400

    try:
        image_data = base64.b64decode(base64_str.split(',')[1])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_np = np.array(image)
        results = face_detection.process(image_np)

        detections = []
        face_filenames = []

        if results.detections:
            for i, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image_np.shape
                bbox = [
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih)
                ]

                face_img = image_np[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
                face_pil_img = Image.fromarray(face_img)

                buffered = io.BytesIO()
                face_pil_img.save(buffered, format="PNG")
                face_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                face_embedding = get_face_embedding(io.BytesIO(buffered.getvalue()))
                if face_embedding is not None:
                    conn = sqlite3.connect(DATABASE)
                    cursor = conn.cursor()
                    cursor.execute('INSERT INTO FACES (BASE64_IMAGE, EMBEDDING) VALUES (?, ?)',
                                   (face_base64, face_embedding.tobytes()))
                    conn.commit()
                    face_id = cursor.lastrowid
                    conn.close()

                    detections.append({
                        'bbox': bbox,
                        'confidence': detection.score[0] if detection.score else None
                    })

                    face_filenames.append(face_base64)

            return jsonify(
                {'message': 'Face(s) detected and saved', 'faces': face_filenames, 'detections': detections}), 200

        return jsonify({'message': 'No faces detected'}), 200

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 400


@app.route('/compare', methods=['POST'])
def compare_file():
    session_id = request.json.get('session_id')
    base64_str = request.json.get('base64')

    if not session_id:
        return jsonify({'error': 'Session ID is required'}), 400

    if not base64_str:
        return jsonify({'error': 'No Base64 data provided'}), 400

    try:
        image_data = base64.b64decode(base64_str.split(',')[1])
        image_embedding = get_face_embedding(io.BytesIO(image_data))
        if image_embedding is None:
            return jsonify({'error': 'No face detected in the image'}), 400

        match = compare_face(image_embedding)
        if match:
            return jsonify({'message': 'Match found', 'matched_face': match, 'percentage': f'{percentage:.2f}%'}), 200
        else:
            return jsonify({'message': 'No match found'}), 200

    except Exception as e:
        return jsonify({'error': f'Invalid Base64 image data: {str(e)}'}), 400


@socketio.on('frame')
def handle_frame(data):
    if 'session_id' not in data or 'image' not in data:
        emit('error', {'error': 'session_id or image data is missing'})
        return

    image_data = data.get('image')
    session_id = data.get('session_id')

    try:
        image_data = base64.b64decode(image_data.split(',')[1])
        image_embedding = get_face_embedding(io.BytesIO(image_data))
        if image_embedding is None:
            emit('error', {'error': 'No face detected in the image'})
        else:
            match = compare_face(image_embedding)
            if match:
                emit('result', {'message': 'Match found', 'matched_face': match, 'percentage': f'{percentage:.2f}%'})
            else:
                emit('result', {'message': 'No match found'})

    except Exception as e:
        emit('error', {'error': f'Invalid Base64 image data: {str(e)}'})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/up')
def up():
    return render_template('up.html')


@app.route('/aprovado')
def aprovado():
    return render_template('approved.html')


if __name__ == '__main__':
    app.run(debug=True)
