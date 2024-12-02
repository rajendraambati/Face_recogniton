import streamlit as st
import cv2
import numpy as np
import torch
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import tempfile

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize MTCNN and ResNet models
mtcnn = MTCNN(keep_all=True, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load the face embeddings database
with open("face_database.pkl", "rb") as f:
    face_database = pickle.load(f)

def extract_face_embeddings(image, boxes):
    """Extract face embeddings from an image and bounding boxes."""
    faces = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            continue
        
        face = cv2.resize(face, (160, 160))
        face = np.transpose(face, (2, 0, 1)) / 255.0
        face = torch.tensor(face, dtype=torch.float32).unsqueeze(0).to(device)
        faces.append(face)
    
    if len(faces) > 0:
        faces = torch.cat(faces)
        embeddings = model(faces).detach().cpu().numpy()
        return embeddings
    return None

def detect_faces(image, database):
    """Detect faces in an image and label them using the database."""
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(img_rgb)

    if boxes is None:
        return img_rgb, "No faces detected."

    embeddings = extract_face_embeddings(img_rgb, boxes)
    if embeddings is None:
        return img_rgb, "No valid face embeddings extracted."

    for box, face_embedding in zip(boxes, embeddings):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

        min_distance = float('inf')
        second_min_distance = float('inf')
        best_match = "Unknown"

        for name, known_embeddings in database.items():
            distances = [np.linalg.norm(face_embedding - known_emb) for known_emb in known_embeddings]
            current_min_distance = min(distances)

            if current_min_distance < min_distance:
                second_min_distance = min_distance
                min_distance = current_min_distance
                best_match = name
            elif current_min_distance < second_min_distance:
                second_min_distance = current_min_distance

        confidence_gap = second_min_distance - min_distance
        threshold = 0.6
        gap_threshold = 0.1

        if min_distance < threshold and confidence_gap > gap_threshold:
            label = best_match
        else:
            label = "Unknown"

        cv2.putText(img_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    return img_rgb, None

# Streamlit App
st.title("Face Recognition App")
st.write("Upload an image, and the system will detect and label faces.")

uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    st.write("Your image is being processed...")
    # Load the uploaded image
    img = Image.open(uploaded_image).convert("RGB")
    
    # Save image to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        img.save(temp_file.name)
        img_path = temp_file.name

    # Perform face detection and recognition
    processed_image, message = detect_faces(img, face_database)

    if message:
        st.warning(message)

    # Display the processed image
    st.image(processed_image, channels="RGB", caption="Processed Image")