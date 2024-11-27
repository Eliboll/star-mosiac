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
        raw_matches = find_matching_points_parallel(img1_map, img2_map, cost_threshold=0.5, match_ratio_threshold=0.5, n_jobs=-1)
        end = time.time()
        print(f"{end - start:.2f} second elapsed")
        matches = []
        for mat in raw_matches:
            if mat[3] > .93:
                matches.append(mat)
        
        self.__matching_points = matches[:100]
        self.show_keypoints(color=(0,255,0))
        #self.__matching_points = matches[:5]
        if self.__cache:
            write_cache(".matches.cache", self.__matching_points)
        return self.__matching_points
    
    def show_keypoints(self,color=None):
        matching_points = self.match_keypoints()
        img1 = self.images[0].original.copy()
        img2 = self.images[1].original.copy()
        for pt1,pt2,_,_ in matching_points:
            use_color = None
            if color==None:
                use_color = deterministic_color_picker(pt1[0]+pt1[1])
            else:
                use_color =  color
            cv2.circle(img1, pt1, 5, use_color, 2)
            cv2.circle(img2, pt2, 5, use_color, 2)
        cv2.imwrite("image-1-overlapping-keypoints.jpg",img1)
        cv2.imwrite("image-2-overlapping-keypoints.jpg",img2)
            
        
    def combine_images(self) -> np.ndarray:
        matching_points = self.match_keypoints()

        im1_pt = []
        im2_pt = []
        for pt in matching_points:
            im1_pt.append(pt[0])
            im2_pt.append(pt[1])
        im1_pt = np.array(im1_pt)
        im2_pt = np.array(im2_pt)
        
        img2 = self.images[1].original
        # calculate centroids
        img1_pivot = np.int64(np.mean(im1_pt,axis=0))
        img2_pivot = np.int64(np.mean(im2_pt,axis=0))
        # difference in where centroids are
        pvt_pnt_diff = img1_pivot - img2_pivot
        #vector from each star to centroid
        vectors1 = im1_pt - img1_pivot
        vectors2 = im2_pt - img2_pivot
        #calculate vector angle
        angles1 = np.arctan2(vectors1[:,1],vectors1[:,0])
        angles2 = np.arctan2(vectors2[:,1],vectors2[:,0])
        # normalize angles
        angle_diffs = (angles2 - angles1 + np.pi) % (2 * np.pi) - np.pi
        total_diffs = np.degrees(angle_diffs)
        # get median angle to account for mismatches
        angle = np.median(np.abs(np.round(total_diffs,-1)))
        #pivot image around centroid using above angle
        center = (int(img2_pivot[0]),int(img2_pivot[1]))
        rotation_matrix = cv2.getRotationMatrix2D(center,angle,1.0,)
        rotated_image = cv2.warpAffine(img2, rotation_matrix, (img2.shape[1]+1000,img2.shape[0]+2000))
        #overlap centroids
        matrix = np.float32([
            [1,0,pvt_pnt_diff[0]],
            [0,1,pvt_pnt_diff[1]]
        ])
        shifted = cv2.warpAffine(rotated_image, matrix, (rotated_image.shape[1],rotated_image.shape[0]))
        #write image
        image1 = self.images[0].original
        shifted[:image1.shape[0],:image1.shape[1]] = image1
        cv2.imwrite("final.jpg", shifted)
        
        ####### UNECESSARY CODE FOR DEBUGGING:
        #self.show_keypoints(color=(0,255,0))
        self.images[0].draw_keypoints(write=True,filename="split-1-stars.jpg")
        self.images[1].draw_keypoints(write=True,filename="split-2-stars.jpg")
        copy1, copy2 = self.images[0].original.copy(), self.images[1].original.copy()
        for pt1,pt2 in zip(im1_pt,im2_pt):
            color = deterministic_color_picker(pt1[0]+pt1[1]+pt2[1]+pt2[0])
            cv2.line(copy1,img1_pivot,pt1,color=color,thickness=2)
            cv2.line(copy2,img2_pivot,pt2,color=color,thickness=2)
        cv2.imwrite("pivot-1.jpg",copy1)
        cv2.imwrite("pivot-2.jpg",copy2)
        ####### END UNECESSARY
        
        return shifted
    

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
            
def deterministic_color_picker(marker):
    # Generate a unique color based on marker index
    np.random.seed(marker)  # Use marker as seed for consistent colors
    return tuple(np.random.randint(0, 255, size=3).tolist())
          
            
if __name__ == "__main__":
    from main import main
    main()