import cv2
import numpy as np

def partition_image(original_image: np.ndarray,x:bool=False,y:bool=False,imwrite:bool=False,filename:str="split") -> tuple[np.ndarray,np.ndarray]:
    if not x and not y:
        if original_image.shape[0] > original_image.shape[1]:
            y = True
        else:
            x = True
    if y:
        image1 = original_image[0:2*original_image.shape[0]//3,:]
        image2 = original_image[original_image.shape[0] - 2*original_image.shape[0]//3:,:]
    else:
        image1 = original_image[:0:2*original_image.shape[1]//3]
        image2 = original_image[:,original_image.shape[1] - 2*original_image.shape[1]//3:]
    if imwrite:
        cv2.imwrite(f"{filename}-1.jpg", image1)
        cv2.imwrite(f"{filename}-2.jpg", image2)
    return image1,image2

if __name__ == "__main__":
    f_name = "burr_oak_colors"
    filename = f"test_images/{f_name}.jpg"
    orig_image = cv2.imread(filename,cv2.IMREAD_COLOR)
    partition_image(orig_image,imwrite=True,filename=f"{f_name}-split")