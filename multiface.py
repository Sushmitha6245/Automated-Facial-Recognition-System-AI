import os
import cv2
import numpy as np
from deepface import DeepFace
from matplotlib import pyplot as plt

def load_known_faces(training_dir):
    known_faces = {}
    for person_name in os.listdir(training_dir):
        person_dir = os.path.join(training_dir, person_name)
        if os.path.isdir(person_dir):
            known_faces[person_name] = []
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                known_faces[person_name].append(img_path)
    return known_faces

def recognize_faces(image_path, known_faces, model_name="VGG-Face", distance_metric="cosine"):
    # Detect faces in the image
    faces = DeepFace.extract_faces(img_path=image_path, enforce_detection=False)

    # Load the image for drawing
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for face in faces:
        facial_area = face['facial_area']
        if isinstance(facial_area, dict):
            x = facial_area.get('x', 0)
            y = facial_area.get('y', 0)
            w = facial_area.get('w', 0)
            h = facial_area.get('h', 0)
        elif isinstance(facial_area, (list, tuple)) and len(facial_area) == 4:
            x, y, w, h = facial_area
        else:
            print(f"Unexpected facial_area format: {facial_area}")
            continue

        face_img = img[y:y+h, x:x+w]

        # Try to recognize the face
        try:
            result = DeepFace.find(img_path=face_img,
                                   db_path=training_dir,
                                   model_name=model_name,
                                   distance_metric=distance_metric,
                                   enforce_detection=False)

            if len(result) > 0 and len(result[0]) > 0:
                recognized_path = result[0]['identity'][0]
                recognized_name = os.path.basename(os.path.dirname(recognized_path))

                # Draw rectangle and name
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(img, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        except Exception as e:
            print(f"Error processing a face: {str(e)}")
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return img

# Set up paths
training_dir = r"D:\Desktop\infosys\Training data set"
test_image_path = r"D:\Desktop\infosys\Testing data set\Group images\f.jpg"

# Load known faces
known_faces = load_known_faces(training_dir)

# Recognize faces in the test image
result_img = recognize_faces(test_image_path, known_faces)

# Display the result
plt.figure(figsize=(12, 8))
plt.imshow(result_img)
plt.axis('off')
plt.title("Recognized Faces")
plt.show()
