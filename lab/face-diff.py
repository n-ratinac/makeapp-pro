import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import procrustes

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Function to detect facial landmarks
def get_face_landmarks(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0]
        return [(int(pt.x * image.shape[1]), int(pt.y * image.shape[0])) for pt in landmarks.landmark]
    return None

# Function to calculate the bounding box of the face based on landmarks
def get_face_bounding_box(landmarks):
    xs = [x for x, y in landmarks]
    ys = [y for x, y in landmarks]
    
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    
    return (x_min, y_min, x_max, y_max)

# Function to crop the face using the bounding box with padding
def crop_face(image, bounding_box, padding=20):
    x_min, y_min, x_max, y_max = bounding_box

    # Add padding and ensure the coordinates are within image bounds
    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max = min(x_max + padding, image.shape[1])
    y_max = min(y_max + padding, image.shape[0])
    
    return image[y_min:y_max, x_min:x_max]

# Function to align landmarks using Procrustes Analysis
def procrustes_align(landmarks1, landmarks2):
    landmarks1 = np.array(landmarks1, dtype=np.float32)
    landmarks2 = np.array(landmarks2, dtype=np.float32)

    if landmarks1.shape != landmarks2.shape:
        raise ValueError(f"Landmark shapes do not match: {landmarks1.shape} vs {landmarks2.shape}")

    # Apply Procrustes Analysis to align landmarks
    _, landmarks1_aligned, landmarks2_aligned = procrustes(landmarks1, landmarks2)
    return landmarks1_aligned, landmarks2_aligned

# Function to compute transformation matrix
def compute_affine_transform(landmarks1, landmarks2):
    if len(landmarks1) < 3 or len(landmarks2) < 3:
        raise ValueError("At least 3 landmarks are required for affine transformation.")

    # Convert landmarks to float32 for compatibility with OpenCV functions
    points1 = np.float32(landmarks1)
    points2 = np.float32(landmarks2)

    # Estimate the affine transformation matrix
    M, inliers = cv2.estimateAffinePartial2D(points2, points1)

    if M is None:
        raise ValueError("Affine transformation matrix could not be computed.")

    return M

# Function to compute the difference between two images
def compute_difference(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference
    diff = cv2.absdiff(gray1, gray2)
    return diff

# Load two images
img1 = cv2.imread('before.png')  # Replace with the path to the first image
img2 = cv2.imread('after.png')  # Replace with the path to the second image

# Step 1: Detect facial landmarks in both images
landmarks1 = get_face_landmarks(img1)
landmarks2 = get_face_landmarks(img2)

if landmarks1 is None or landmarks2 is None:
    print("Face landmarks not detected in one or both images.")
    exit()

# Step 2: Get the bounding boxes for both faces
bounding_box1 = get_face_bounding_box(landmarks1)
bounding_box2 = get_face_bounding_box(landmarks2)

# Step 3: Crop the faces using the bounding boxes
cropped_face1 = crop_face(img1, bounding_box1)
cropped_face2 = crop_face(img2, bounding_box2)

# Resize both cropped faces to the same size (based on the first image)
cropped_face2 = cv2.resize(cropped_face2, (cropped_face1.shape[1], cropped_face1.shape[0]))

# Step 4: Align landmarks of the cropped faces using Procrustes Analysis
try:
    aligned_landmarks1, aligned_landmarks2 = procrustes_align(landmarks1, landmarks2)
except ValueError as e:
    print(f"Error during Procrustes alignment: {e}")
    exit()

# Step 5: Apply the transformation to align face2 to face1
try:
    M = compute_affine_transform(aligned_landmarks1, aligned_landmarks2)
    aligned_face2 = cv2.warpAffine(cropped_face2, M, (cropped_face1.shape[1], cropped_face1.shape[0]))
except ValueError as e:
    print(f"Error during affine transformation: {e}")
    exit()

# Step 6: Compute the difference between the aligned faces
difference = compute_difference(cropped_face1, aligned_face2)

# Step 7: Display the cropped and aligned faces, and the difference
cv2.imshow('Cropped Face 1', cropped_face1)
cv2.imshow('Aligned Cropped Face 2', aligned_face2)
cv2.imshow('Difference Image', difference)

cv2.waitKey(0)
cv2.destroyAllWindows()
