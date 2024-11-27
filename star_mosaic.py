import cv2
import numpy as np
from mosaic_image import mosaic_image
from match_keypoints import find_matching_points_parallel
import time

class star_mosaic:
    def __init__(self, CACHE=False):
        self.images: list[mosaic_image] = []
        self.__matching_points = []
        self.__cache = CACHE
        
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
        if self.__cache:
            import os.path
            if os.path.exists(".matches.cache"):
                self.__matching_points = open_cache(".matches.cache")
        
        if self.__matching_points:
            return self.__matching_points
        img1_map = self.images[0].get_relationships()
        img2_map = self.images[1].get_relationships()
            
        start = time.time()
        matches = find_matching_points_parallel(img1_map, img2_map, cost_threshold=0.5, match_ratio_threshold=0.5, n_jobs=-1)
        end = time.time()
        print(f"{end - start:.2f} second elapsed")
        
        return matches[:100]
        if self.__cache:
            write_cache(".matches.cache", self.__matching_points)
        return self.__matching_points
        
        def open_cache(filename):
    cache = []
    with open(filename, "r") as fp:
        for line in fp.readlines():
            parts = line.split(',')
            c1 = (int(parts[0]), int(parts[1]))
            c2 = (int(parts[2]), int(parts[3]))
            cost = float(parts[4])
            ratio = float(parts[5])
            cache.append((c1,c2,cost,ratio))
    return cache
            
            
def write_cache(filename, matches):
    with open(filename,"w") as fp:
        for match in matches:
            fp.write(f"{match[0][0]},{match[0][1]},{match[1][0]},{match[1][1]},{match[2]},{match[3]}\n")
            
if __name__ == "__main__":
    from main import main
    main()
