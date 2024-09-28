import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB as MediaPipe works with RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find face landmarks
    results = face_mesh.process(rgb_frame)

    # If landmarks are found
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get the bounding box of the face
            x_coords = [landmark.x * frame.shape[1] for landmark in face_landmarks.landmark]
            y_coords = [landmark.y * frame.shape[0] for landmark in face_landmarks.landmark]
            min_x = int(min(x_coords)) - padding
            max_x = int(max(x_coords)) + padding
            min_y = int(min(y_coords)) - padding
            max_y = int(max(y_coords)) + padding

            # Ensure bounding box stays within the image dimensions
            min_x = max(0, min_x)
            max_x = min(frame.shape[1], max_x)
            min_y = max(0, min_y)
            max_y = min(frame.shape[0], max_y)

            # Crop the face region with padding
            cropped_face = frame[min_y:max_y, min_x:max_x]

            # Get screen size to scale the cropped face
            screen_width = 2560  # You can adjust this to fit the actual screen width
            screen_height = 1440  # You can adjust this to fit the actual screen height

            # Resize the cropped face to fit the screen while maintaining aspect ratio
            aspect_ratio = cropped_face.shape[1] / cropped_face.shape[0]
            if aspect_ratio > 1:
                new_width = screen_width
                new_height = int(screen_width / aspect_ratio)
            else:
                new_height = screen_height
                new_width = int(screen_height * aspect_ratio)

            resized_face = cv2.resize(cropped_face, (new_width, new_height))

            results = face_mesh.process(resized_face)

            for idx, landmark in enumerate(face_landmarks.landmark):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                # Draw the index number at the landmark location
                cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.4, (255, 0, 0), 1, cv2.LINE_AA)  # Blue text for numbers

            

            # Display the frame with landmarks as numbers
                cv2.imshow('Facial Landmarks with Numbers', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break


# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
