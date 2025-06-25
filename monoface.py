import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from deepface import DeepFace

def preprocess_image(image):
    if image is not None and image.size != 0:
        image = cv2.resize(image, (224, 224))  # VGG-Face expects 224x224 images
        image = image / 255.0  # Normalize
        return image
    else:
        return None

def load_images_from_directory(base_dir):
    images = []
    labels = []
    for person_name in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person_name)
        if os.path.isdir(person_dir):  # Check if it's a directory
            for filename in os.listdir(person_dir):
                img_path = os.path.join(person_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(person_name)
    return images, labels

mono_face_test_directory = r"D:\Desktop\Testing data set\mono face images"

def get_face_embedding(image):
    if image is not None:
        embedding = DeepFace.represent(img_path=image, model_name="VGG-Face", enforce_detection=False)
        if embedding:
            return embedding[0]["embedding"]
    return None

def recognize_face(test_embedding, known_face_encodings, known_face_names, threshold=0.4):
    if not known_face_encodings:
        raise ValueError("No known face encodings available.")
    distances = [np.linalg.norm(test_embedding - known_face_encoding) for known_face_encoding in known_face_encodings]
    if not distances:
        raise ValueError("No distances computed. Check embeddings and images.")
    best_match_index = np.argmin(distances)
    if distances[best_match_index] < threshold:
        return known_face_names[best_match_index], distances[best_match_index]
    else:
        return "Unknown", None

def test_on_mono_faces(test_directory, train_images, train_labels, known_face_encodings, known_face_names):
    for filename in os.listdir(test_directory):
        test_image_path = os.path.join(test_directory, filename)
        test_image = cv2.imread(test_image_path)
        test_image_preprocessed = preprocess_image(test_image)
        if test_image_preprocessed is not None:
            test_embedding = get_face_embedding(test_image_preprocessed)
            if test_embedding is not None:
                min_distance = float('inf')
                best_match_image = None
                best_match_name = None

                for i, train_embedding in enumerate(known_face_encodings):
                    distance = np.linalg.norm(test_embedding - train_embedding)
                    if distance < min_distance:
                        min_distance = distance
                        best_match_image = train_images[i]
                        best_match_name = known_face_names[i]

                if best_match_image is not None:
                    plt.figure(figsize=(10, 5))
                    
                    # Plot testing image
                    plt.subplot(1, 2, 1)
                    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
                    plt.title(f"Testing Image: {filename}")
                    plt.axis('off')
                    
                    # Plot best match training image
                    plt.subplot(1, 2, 2)
                    plt.imshow(cv2.cvtColor(best_match_image, cv2.COLOR_BGR2RGB))
                    plt.title(f"Best Match: {best_match_name} (Distance: {min_distance:.2f})")
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                else:
                    print(f"No best match found for {filename}")
            else:
                print(f"Could not get embedding for test image: {test_image_path}")
        else:
            print(f"Could not preprocess test image: {test_image_path}")

# Load training images
training_dir = r"D:\Desktop\Training data set"
train_images, train_labels = load_images_from_directory(training_dir)

# Preprocess and encode training images
known_face_encodings = []
known_face_names = []
for img, label in zip(train_images, train_labels):
    img_preprocessed = preprocess_image(img)
    if img_preprocessed is not None:
        embedding = get_face_embedding(img_preprocessed)
        if embedding is not None:
            known_face_encodings.append(np.array(embedding))
            known_face_names.append(label)
        else:
            print(f"Could not get embedding for training image: {label}")
    else:
        print(f"Could not preprocess training image: {label}")

# Test on Mono faces
test_on_mono_faces(mono_face_test_directory, train_images, train_labels, known_face_encodings, known_face_names)
