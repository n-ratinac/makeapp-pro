import cv2
import numpy as np
import keypoints as kp
import regions


def mask(image, target_landmark_indices):
    """
    Create a mask for a facial region based on the provided landmarks and target indices.

    Parameters:
    - image: The input image.
    - landmarks: The list of landmarks (MediaPipe normalized landmarks).
    - targets: The specific indices of the landmarks to use for the polygon.

    Returns:
    - mask: The mask for the facial region.
    """

    # Extract landmarks based on the provided indices

    

    # all the points outside the polygon will be black
    mask = np.zeros_like(image)
    landmarks = regions.landmark_coords(image, target_landmark_indices)
    
    # Convert normalized landmark points to pixel coordinates
    points = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in landmarks]

    # Draw the polygon
    cv2.fillPoly(mask, [np.array(points)], (255, 255, 255))

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

if __name__ == "__main__":
    # Load the images
    image1 = cv2.imread("data/before/1.png")
    image2 = cv2.imread("data/after/1.png")

    cv2.imshow("Image 1", image1)
    cv2.imshow("Image 2", image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Compute the masked image
    masked_image = mask(image1, kp.EYE_L3)

    cv2.imshow("Masked Image", masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()