import cv2
import numpy as np

def preprocessor(image: np.ndarray) -> np.ndarray:
    # Preprocessing
# Normalize the image to enhance contrast
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Apply Gaussian blur to smooth the image and reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply binary thresholding to isolate stars
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    # Invert the thresholded image to make stars white and background black
    thresh = cv2.bitwise_not(thresh)

    # Optionally, apply morphological dilation to emphasize star blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed = cv2.dilate(thresh, kernel, iterations=1)
    return processed

def get_stars(image: np.ndarray) -> list[cv2.KeyPoint]:
    # Set up SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 1  # Minimum blob area to detect small stars
    params.maxArea = 50  # Maximum blob area for larger stars

    params.filterByCircularity = True
    params.minCircularity = 0.6  # Stars should be roughly circular

    params.filterByInertia = True
    params.minInertiaRatio = 0.5  # Circular blobs have higher inertia ratio

    # Create Blob Detector
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs in the processed image
    keypoints = detector.detect(image)
    for kp in keypoints:
        kp.pt = (int(kp.pt[0]),int(kp.pt[1]))
    return keypoints

def draw_keypoints(image: np.ndarray, keypoints: list[cv2.KeyPoint], show=False, write=False, filename="star_highlight.jpg") -> None:
    new_image = np.copy(image)
    for keypoint in keypoints:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])  # Keypoint coordinates
        radius = int(keypoint.size * 2)  # Adjust the multiplier to control circle size
        cv2.circle(new_image, (x, y), radius, (0, 255, 0), 2)  # Green circle, thickness 2
    if show:
        cv2.imshow('Stars Detected', new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if write:
        cv2.imwrite(filename,new_image)


def main() -> None:
    f_name = "burr_oak_colors"
    filename = f"test_images/{f_name}.jpg"
    orig_image = cv2.imread(filename,cv2.IMREAD_COLOR)
    processed_image = preprocessor(orig_image)
    keypoints = get_stars(processed_image)
    draw_keypoints(orig_image,keypoints,write=True)


if __name__ == "__main__":
    main()