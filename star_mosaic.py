import cv2
import numpy as np
from mosaic_image import mosaic_image
from match_keypoints import find_matching_points_parallel
import time

class star_mosaic:
    def __init__(self):
        self.images: list[mosaic_image] = []
        
    def add_raw_image(self, image:np.ndarray) -> None:
        self.images.append(mosaic_image(image))
    
    def add_mosaic_image(self, image:mosaic_image) -> None:
        self.images.append(image)
        
    def add_raw_list(self, images: list[np.ndarray]) -> None:
        for image in images:
            self.add_raw_image(image)
    
    def add_mosaic_list(self, images: list[mosaic_image]) -> None:
        for image in images:
            self.add_mosaic_image(image)
            
    def match_keypoints(self):
        img1_map = self.images[0].get_relationships()
        img2_map = self.images[1].get_relationships()
            
        start = time.time()
        matches = find_matching_points_parallel(img1_map, img2_map, cost_threshold=0.5, match_ratio_threshold=0.5, n_jobs=-1)
        end = time.time()
        print(f"{end - start:.2f} second elapsed")
        
        return matches[:100]
        
        