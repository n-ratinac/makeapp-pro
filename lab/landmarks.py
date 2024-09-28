import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize MediaPipe Face Mesh (for landmark detection)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

# Define padding and screen size
padding = 20  # Padding of 20px around the bounding box
screen_width = 2560  # Target screen width
screen_height = 1440  # Target screen height

# Function to display landmark numbers on the image
def draw_landmark_numbers(image, landmarks):
    for idx, landmark in enumerate(landmarks.landmark):
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])
        cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)  # Blue numbers

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB as MediaPipe works with RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Step 1: Detect face bounding box using face detection
    face_results = face_detection.process(rgb_frame)

    if face_results.detections:
        for detection in face_results.detections:
            # Get bounding box from the detection
            bboxC = detection.location_data.relative_bounding_box
            bbox_x = int(bboxC.xmin * frame.shape[1]) - padding
            bbox_y = int(bboxC.ymin * frame.shape[0]) - padding
            bbox_w = int(bboxC.width * frame.shape[1]) + padding * 2
            bbox_h = int(bboxC.height * frame.shape[0]) + padding * 2

            # Ensure bounding box stays within the image dimensions
            bbox_x = max(0, bbox_x)
            bbox_y = max(0, bbox_y)
            bbox_w = min(bbox_w, frame.shape[1] - bbox_x)
            bbox_h = min(bbox_h, frame.shape[0] - bbox_y)

            # Step 2: Crop the face region from the original frame
            cropped_face = frame[bbox_y:bbox_y + bbox_h, bbox_x:bbox_x + bbox_w]

            # Step 3: Find landmarks on the cropped face
            rgb_cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            face_mesh_results = face_mesh.process(rgb_cropped_face)

            # Step 4: Resize the cropped face to fit the screen while maintaining aspect ratio
            aspect_ratio = cropped_face.shape[1] / cropped_face.shape[0]
            if aspect_ratio > 1:
                new_width = screen_width
                new_height = int(screen_width / aspect_ratio)
            else:
                new_height = screen_height
                new_width = int(screen_height * aspect_ratio)

            resized_face = cv2.resize(cropped_face, (new_width, new_height))

            # Step 5: Draw landmark numbers on the resized face if landmarks are found
            if face_mesh_results.multi_face_landmarks:
                for landmarks in face_mesh_results.multi_face_landmarks:
                    draw_landmark_numbers(resized_face, landmarks)

            # Display the resized cropped face with landmark numbers
            cv2.imshow('Cropped, Resized Face with Landmark Numbers', resized_face)

    # Display the original frame
    cv2.imshow('Original Frame', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
