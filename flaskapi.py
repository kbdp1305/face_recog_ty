from flask import Flask, request, jsonify
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np

import cv2
import tkinter as tk
from tkinter import messagebox, simpledialog
from facenet_pytorch import MTCNN, InceptionResnetV1
from utils.databases import FaceEmbeddingDB
from utils.distance_counting import DistanceCounting
from src.face_recognition import FaceRecog

# Initialize Flask app
app = Flask(__name__)

# Initialize database
db_uri = "mongodb+srv://dharmaworkdev:dharma123@facerecognitiontrial.jxq2i.mongodb.net/?retryWrites=true&w=majority&appName=FaceRecognitionTrial"
database = FaceEmbeddingDB(db_uri, 'trial_acces_plany', 'mydatabase')

# Initialize Face Recognition Model
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, 
              thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True)
facenet = InceptionResnetV1(pretrained="vggface2").eval()
face_recognizer = FaceRecog(mtcnn, facenet)

@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the Face Recognition API!'})

@app.route('/add_face', methods=['POST'])
def add_face():
    data = request.json
    name = data.get('name')
    pos = data.get('pos')
    img_path = data.get('img_path')  # Assuming image is stored and path is sent
    
    if not name or pos not in [1, 2, 3] or not img_path:
        return jsonify({'error': 'Invalid input'}), 400

    img = face_recognizer.preprocess_image(img_path)
    embedding,_ = face_recognizer.detect(img)
    
    if embedding is None:
        return jsonify({'error': 'No face detected'}), 400
    
    database.add_data(name, pos, embedding)
    return jsonify({'message': 'Face added successfully!'})

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    data = request.json
    pos = data.get('pos')
    img_path = data.get('img_path')

    if not pos or not img_path:
        return jsonify({'error': 'Invalid input'}), 400

    stored_embeddings = database.load_embeddings(pos)
    if not stored_embeddings:
        return jsonify({'error': 'No stored data found'}), 404

    img = face_recognizer.preprocess_image(img_path)
    new_embedding,_ = face_recognizer.detect(img)
    
    if new_embedding is None:
        return jsonify({'error': 'No face detected'}), 400

    best_match = max(stored_embeddings, key=lambda data: DistanceCounting.cosine_similarity(data['tensor'], new_embedding), default=None)
    similarity = DistanceCounting.cosine_similarity(best_match['tensor'], new_embedding) if best_match else 0
    print(similarity)

    if similarity > 0.65:
        return jsonify({'message': f"Access Granted: {best_match['name']}"})
    else:
        return jsonify({'message': "Access Denied: Face does not match"})

@app.route('/delete_face', methods=['POST'])
def delete_face():
    data = request.json
    name = data.get('name')

    if not name:
        return jsonify({'error': 'Invalid input'}), 400

    database.delete_by_name(name)
    return jsonify({'message': 'Face data deleted successfully!'})

if __name__ == '__main__':
    app.run(debug=True)
