import cv2
import numpy as np
from mosaic_image import mosaic_image

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