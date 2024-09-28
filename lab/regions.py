
import cv2
import numpy as np
import mediapipe as mp
from shapely.geometry import Polygon, Point
from scipy.spatial import Delaunay

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6)

def landmark_coords(image, landmark_indices=[]):
    """
    Detects facial landmarks in an image using MediaPipe Face Mesh.
    
    Parameters:
    - image: The input image.
    
    Returns:
    - landmarks: The list of facial landmarks (MediaPipe normalized landmarks).
    """
    normalized_gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_gray_image = cv2.equalizeHist(normalized_gray_image)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
    else:
        landmarks = None
    
    if len(landmark_indices) == 0:
        return landmarks

    landmarks =  [landmarks.landmark[i] for i in landmark_indices]
    return landmarks

def draw(image, landmark_indices, color=(0, 255, 0), thickness=2):
    """
    Draws a polygon on an image using a list of MediaPipe landmarks.
    
    Parameters:
    - image: The input image.
    - landmarks: The list of landmarks (MediaPipe normalized landmarks).
    - landmark_indices: The specific indices of the landmarks to use for the polygon.
    - color: The color of the polygon in BGR format (default is green).
    - thickness: The thickness of the polygon lines (default is 2).
    
    Returns:
    - image: The image with the polygon drawn on it.
    """
    h, w, _ = image.shape
    landmarks = landmark_coords(image)
    landmarks = [(point.x, point.y) for point in landmarks.landmark]

    # Convert normalized landmark points to pixel coordinates
    points = [(int(landmarks[landmark][0] * w), int(landmarks[landmark][1] * h)) for landmark in landmark_indices]
    
    # Draw the polygon
    cv2.polylines(image, [np.array(points)], True, color, thickness)
    
    return image

def bbox(image, landmarks):
    """
    Computes the bounding box of the specified landmarks.
    """
    
    if landmarks is None:
        return None

    x_coords = [landmark.x for landmark in landmarks]
    y_coords = [landmark.y for landmark in landmarks]

    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    return x_min, y_min, x_max, y_max

def crop(image, targets):
    """
    Crops an image to the bounding box of the specified landmarks.
    """
    # Compute the bounding box
    box = bbox(image, targets)

    if box is None:
        return None, None

    x_min, y_min, x_max, y_max = box

    # Crop the image
    cropped_image = image[int(y_min * image.shape[0]):int(y_max * image.shape[0]), int(x_min * image.shape[1]):int(x_max * image.shape[1])]

    return cropped_image, box

def crop_masked(image):
    """
    Crops an image so that there are the fewest black pixels around the edges.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def align_regions(image1, image2, src_points, tgt_points):
    """
    Use afine transformation to align the regions.
    """

    src_points = [(point.x, point.y) for point in src_points]
    tgt_points = [(point.x, point.y) for point in tgt_points]

    # Convert the points to pixel values
    h, w, _ = image1.shape
    src_points = [(int(point[0] * w), int(point[1] * h)) for point in src_points]
    tgt_points = [(int(point[0] * w), int(point[1] * h)) for point in tgt_points]

    # Convert the points to numpy arrays
    src_points = np.array(src_points)
    tgt_points = np.array(tgt_points)

    # Find the best affine transformation matrix
    transformation_matrix = cv2.estimateAffinePartial2D(src_points, tgt_points)[0]

    # Apply the transformation to the source image
    aligned_image = cv2.warpAffine(image1, transformation_matrix, (w, h))

    return aligned_image

def align_perspective(src_image, tgt_image, region):
    """
    Aligns the src_image to the tgt_image based on the corresponding points.
    """

    src_points = inner_landmarks(src_image, region)
    tgt_points = inner_landmarks(tgt_image, region)

    # src_points = [(point.x, point.y) for point in src_points]
    # tgt_points = [(point.x, point.y) for point in tgt_points]

    # Convert the points to pixel values
    h, w, _ = tgt_image.shape

    src_points = [(int(point[0] * w), int(point[1] * h)) for point in src_points]
    tgt_points = [(int(point[0] * w), int(point[1] * h)) for point in tgt_points]

    # Convert the points to numpy arrays
    src_points = np.array(src_points)
    tgt_points = np.array(tgt_points)

    # Find the best homography matrix
    transformation_matrix, _ = cv2.findHomography(src_points, tgt_points, cv2.RANSAC)

    # Apply the transformation to the source image
    aligned_image = cv2.warpPerspective(src_image, transformation_matrix, (w, h))
    

    return aligned_image

def triangulate_region(image, polygon_indices):
    """
    Performs Delaunay triangulation of points and returns the indices of triangles
    that lie within the specified bounding polygon.

    Parameters:
    - points: A numpy array of shape (N, 2) representing the 2D coordinates of points.
    - polygon_indices: A list or array of indices representing the vertices of the bounding polygon.

    Returns:
    - List of tuples containing the indices of the triangles within the polygon.
    """

    landmarks = landmark_coords(image)
    landmarks = np.array([(point.x, point.y) for point in landmarks.landmark])
    # Create the polygon from the provided indices
    polygon_coords = [landmarks[i] for i in polygon_indices]
    
    polygon = Polygon(polygon_coords)
    
    polygon = polygon.buffer(0)


    # Combine the boundary points with interior points
    points_with_boundary = landmarks

    # Perform Delaunay triangulation on the points
    tri = Delaunay(points_with_boundary)

    # Function to check if a triangle is inside the polygon
    def is_triangle_inside_polygon(triangle, polygon):
        triangle_poly = Polygon(triangle)
        return polygon.contains(triangle_poly)

    # Filter triangles that are inside the polygon and store their indices
    triangles_indices = []
    for simplex in tri.simplices:
        triangle = points_with_boundary[simplex]
        if is_triangle_inside_polygon(triangle, polygon):
            triangles_indices.append(tuple(simplex))  # Append the triangle indices as a tuple

    return triangles_indices

def inner_landmarks(image, polygon_indices):
    """
    Returns the landmarks that are inside the specified polygon.
    """
    landmarks = landmark_coords(image)
    landmarks = np.array([(point.x, point.y) for point in landmarks.landmark])
    # Create the polygon from the provided indices
    polygon_coords = [landmarks[i] for i in polygon_indices]
    
    polygon = Polygon(polygon_coords)
    
    polygon = polygon.buffer(0)

    inner_landmarks = []
    for landmark in landmarks:
        point = Point(landmark)
        if polygon.contains(point):
            inner_landmarks.append(landmark)

    return inner_landmarks
    

    

    
    

    

    