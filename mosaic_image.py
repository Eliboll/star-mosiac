import cv2
import numpy as np

class mosaic_image:
    def __init__(self, image: np.ndarray):
        self.original:np.ndarray  = image
        self.__processed:np.ndarray = None
        self.__keypoints:list[cv2.KeyPoint] = None
        self.__relationships: dict[tuple[int, int], list[tuple[np.ndarray, float]]] = None
##############################################################################
# Private methods    
##############################################################################        
    def __preprocessor(self) -> np.ndarray:
        # Preprocessing
        # Normalize the image to enhance contrast
        image = cv2.normalize(self.original, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # Apply Gaussian blur to smooth the image and reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Apply binary thresholding to isolate stars
        _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

        # Invert the thresholded image to make stars white and background black
        thresh = cv2.bitwise_not(thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.__processed = cv2.dilate(thresh, kernel, iterations=1)
    
    def __get_stars(self) -> list[cv2.KeyPoint]:
        if self.__processed == None:
            self.__preprocessor()
        
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
        keypoints = detector.detect(self.__processed)
        for kp in keypoints:
            kp.pt = (int(kp.pt[0]),int(kp.pt[1]))
        self.__keypoints = keypoints

    def __get_relationships(self, size:float=.25):
        if self.__keypoints == None:
            self.__get_stars()
        
        max_distance = (self.orignal.shape[0] if self.orignal.shape[0] > self.orignal.shape[1] else self.orignal.shape[1]) * size
        final_map = {}
        for keypoint in self.__keypoints:
            in_range = []
            points = np.array([int(keypoint.pt[0]),int(keypoint.pt[1])])
            for neighbor in self.__keypoints:
                neighbor_points = (int(neighbor.pt[0]),int(neighbor.pt[1]))
                if abs(neighbor_points[0] - points[0]) > max_distance or abs(neighbor_points[1] - points[1]) > max_distance:
                    continue
                in_range.append(neighbor_points)
            # This section was used for the distance picture creation
            # copy_image = image.copy()
            # for point in in_range:
                #cv2.line(copy_image,points,(int(point.pt[0]),int(point.pt[1])),color=(0,255,0),thickness=1)
            # cv2.imwrite(f"neighborhoods/{points[0]}-{points[1]}.jpg",copy_image)
            in_range = np.array(in_range)
            vectors = in_range - points
            angles_radians = np.arctan2(vectors[:, 1], vectors[:, 0])
            final_list = []
            for point,angle in zip(in_range,angles_radians):
                final_list.append((point,angle))
            final_map[(points[0],points[1])] = final_list
        self.__relationships = final_map
##############################################################################
# Public methods    
##############################################################################        
    def draw_keypoints(self, show=False, write=False, filename="star_highlight.jpg") -> None:
        '''
        Highlights the stars in the image
        :param bool show: Display image as opencv window
        :param bool write: Save the file with filename
        :param str filename: Filename to save write as
        
        '''
        if not self.__keypoints:
            self.__get_stars()
        
        new_image = np.copy(self.original)
        for keypoint in self.__keypoints:
            x, y = int(keypoint.pt[0]), int(keypoint.pt[1])  # Keypoint coordinates
            radius = int(keypoint.size * 2)  # Adjust the multiplier to control circle size
            cv2.circle(new_image, (x, y), radius, (0, 255, 0), 2)  # Green circle, thickness 2
        if show:
            cv2.imshow('Stars Detected', new_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if write:
            cv2.imwrite(filename,new_image)
            
##############################################################################
# accessors methods    
##############################################################################
    def get_processed(self) -> np.ndarray:
        if not self.__processed:
            self.__preprocessor()
        return self.__processed
    
    def get_keypoints(self) -> list[cv2.KeyPoint]:
        if not self.__keypoints:
            self.__get_stars()
        return self.__keypoints
    
    def get_relationships(self) -> dict[tuple[int, int], list[tuple[np.ndarray, float]]]:
        if not self.__relationships:
            self.__get_relationships()
        return self.__relationships