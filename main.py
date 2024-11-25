import cv2
import numpy as np
from mosaic_image import mosaic_image
from star_mosaic import star_mosaic
from partition_image import partition_image
        
        

def main() -> None:
    f_name = "burr_oak_colors"
    filename = f"test_images/{f_name}.jpg"
    orig_image = cv2.imread(filename,cv2.IMREAD_COLOR)
    
    images = partition_image(orig_image)
    
    mosaic = star_mosaic()
    mosaic.add_raw_list(images)
    
    mosaic.match_keypoints()
    pass



if __name__ == "__main__":
    main()