import cv2
import regions 
from masking import mask
import keypoints as kp
from skimage.metrics import structural_similarity as ssim

def regional_similarity(image1, image2, target_landmarks):
    """
    Computes the similarity between two images based on the target landmarks which represent the polygon that encloses the region.
    """
    # align the images for perspective correction

    

    image1 = regions.align_perspective(image1, image2, target_landmarks)

    # apply gaussian blur to the images
    #image1 = cv2.GaussianBlur(image1, (7, 7), 0)
    #image2 = cv2.GaussianBlur(image2, (7, 7), 0)

    # show the aligned images
    # cv2.imshow("Aligned 1", image1)
    # cv2.imshow("Aligned 2", image2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # create masks for the regions
    masked1 = mask(image1, target_landmarks)
    masked2 = mask(image2, target_landmarks)



    cv2.imshow("Mask 1", masked1)
    cv2.imshow("Mask 2", masked2)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # crop the aligned images
    cropped1 = regions.crop_masked(masked1)
    cropped2 = regions.crop_masked(masked2)

    # resize the images to the same size
    cropped1 = cv2.resize(cropped1, (cropped2.shape[1], cropped2.shape[0]))    
    
    cv2.imshow("Cropped 1", cropped1)
    cv2.imshow("Cropped 2", cropped2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # show the difference between the images
    difference = cv2.absdiff(cropped1, cropped2)
    cv2.imshow("Difference", difference)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    lab1 = cv2.cvtColor(cropped1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(cropped2, cv2.COLOR_BGR2LAB)
    
    score0 = ssim(lab1[:,:,0], lab2[:,:,0])
    score1 = ssim(lab1[:,:,1], lab2[:,:,1])
    score2 = ssim(lab1[:,:,2], lab2[:,:,2])

    return score0, score1, score2

if __name__ == "__main__":
    # Load the images
    image1 = cv2.imread("data/before/1.png")
    image2 = cv2.imread("data/after/1.png")
    

    # Show the images
    # cv2.imshow("Image 1", image1)
    # cv2.imshow("Image 2", image2)
    # hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)

    # # show all the channels of the image
    # cv2.imshow("Hue", hsv1[:,:,0])
    # cv2.imshow("Saturation", hsv1[:,:,1])
    # cv2.imshow("Value", hsv1[:,:,2])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Compute the regional similarity without the value channel
    similarity = regional_similarity(image1, image2, kp.EYE_L3)
    
    print("Regional Similarity:", similarity)