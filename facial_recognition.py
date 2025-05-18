import face_recognition
import cv2
import os
from pymongo import MongoClient
import numpy as np

# MongoDB configuration
client = MongoClient('mongodb+srv://roshanrajmahato:EiQyv4qxQCSkX3M6@cluster0.90ynhbs.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')  # Change this if using a remote server
db = client['face_recognition_db']
collection = db['known_faces']

# Load known faces from MongoDB
def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    for document in collection.find():
        encoding = np.array(document['encoding'])
        known_face_encodings.append(encoding)
        known_face_names.append(document['name'])

    return known_face_encodings, known_face_names

def recognize_face(frame, known_face_encodings, known_face_names):
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)

    return face_locations, face_names
