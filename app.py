from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import face_recognition
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes 

@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    try:
        # Retrieve the known and unknown face images from the request
        known_image_file = request.files['known_image']
        unknown_image_file = request.files['unknown_image']

        # Load the images into numpy arrays
        known_image = face_recognition.load_image_file(known_image_file)
        unknown_image = face_recognition.load_image_file(unknown_image_file)

        # Find face encodings in both images (we assume one face per image)
        known_face_encodings = face_recognition.face_encodings(known_image)
        unknown_face_encodings = face_recognition.face_encodings(unknown_image)

        # Ensure we have encodings from both images
        if len(known_face_encodings) == 0 or len(unknown_face_encodings) == 0:
            return jsonify({"error": "Could not locate a face in one or both images"}), 400

        # Compare faces (taking only the first encoding if multiple faces are found)
        match = face_recognition.compare_faces([known_face_encodings[0]], unknown_face_encodings[0])[0]

        # Return the result as a JSON-serializable dictionary
        return jsonify({"match": bool(match)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/compare-base', methods=['POST'])
def compare_base():
    try:
        # Retrieve the known and unknown images in base64 format from the request
        known_image_base64 = request.json['known_image']
        unknown_image_base64 = request.json['unknown_image']

        # Decode the base64 images
        known_image_data = base64.b64decode(known_image_base64)
        unknown_image_data = base64.b64decode(unknown_image_base64)

        # Convert the byte data to numpy arrays
        known_image = Image.open(BytesIO(known_image_data))
        unknown_image = Image.open(BytesIO(unknown_image_data))

        # Convert to a format suitable for face_recognition
        known_image = face_recognition.load_image_file(BytesIO(known_image_data))
        unknown_image = face_recognition.load_image_file(BytesIO(unknown_image_data))

        # Find face encodings in both images (we assume one face per image)
        known_face_encodings = face_recognition.face_encodings(known_image)
        unknown_face_encodings = face_recognition.face_encodings(unknown_image)

        # Ensure we have encodings from both images
        if len(known_face_encodings) == 0 or len(unknown_face_encodings) == 0:
            return jsonify({"error": "Could not locate a face in one or both images"}), 400

        # Compare faces (taking only the first encoding if multiple faces are found)
        match = face_recognition.compare_faces([known_face_encodings[0]], unknown_face_encodings[0])[0]

        # Return the result as a JSON-serializable dictionary
        return jsonify({"match": bool(match)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
