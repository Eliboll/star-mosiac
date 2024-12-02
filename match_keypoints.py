# from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
import numpy as np
import cupy as cp

# perfect allignment
THRESHOLD = 100000

# 180 flipped
# THRESHOLD = 40000

# # 10 flipped
# THRESHOLD = 21000

MAX_RES = 0

BLACKLIST = []


def new_find_matching_points_parallel(plane1,plane2):
    import time
    start = time.time()
    
    plane1 = get_star_lookup_matrix(plane1)
    plane2 = get_star_lookup_matrix(plane2)
    
    end = time.time()
    print(f"calculation took {end - start:.2f} seconds")

    return find_matches(plane1,plane2)


def get_star_lookup_matrix(plane):
    new_dict = {}
    for key in plane.keys():
        new_dict[key] = {}
        new_dict[key]["matrix"]    = find_angles(key,plane[key])
        new_dict[key]["points"]    = cp.array([x[0] for x in plane[key]])
        #new_dict[key]["distances"] = get_vector_lengths(key,new_dict[key]["points"])
    return new_dict

def find_angles(main_pt, neighbors,rounding=5):
    main_pt = cp.array(main_pt)
    neighbors = cp.array([x[0] for x in neighbors])
    vectors = neighbors - main_pt
    angles = cp.arctan2(vectors[:,1], vectors[:,0])
    
    #pairwise
    angle_matrix = angles[:, None] - angles[None, :]
    angle_matrix = cp.mod(angle_matrix + cp.pi, 2 * cp.pi) - cp.pi
    return cp.round(angle_matrix,decimals=rounding)

def get_vector_lengths(point, neighbors):
    vectors = neighbors - cp.array(point)
    return cp.linalg.norm(vectors,axis=1)

def find_matches(plane1_dict, plane2_dict):
    results = []
    prepped_params = []
    for image_1_star in plane1_dict.keys():
        prepped_params.append((plane1_dict[image_1_star]['matrix'],plane2_dict,image_1_star))
    import time
    start = time.time()
    results = Parallel(n_jobs=-1)(
        delayed(match_matrix_job)(
            matrix,dictionary,point
        )
        for matrix,dictionary,point in prepped_params
    )

    # Filter out None results (unmatched points)
    #matches = [[np.ndarray(int(match[0][0]),int(match[0][1])),np.ndarray(int(match[0][0]),int(match[0][1]))] for match in results if match is not None]
    matches = [x for x in results if x is not None]
    end = time.time()
    print(f"took {end - start:.2f} seconds")
    global MAX_RES
    print(f"Max Result: {MAX_RES}")
    return matches
       
def match_matrix_job(matrix, plane2_dict, origianl_star):
    for image_2_star in plane2_dict.keys():
        # if image_2_star in BLACKLIST:
        #     continue
        res = compare_matrixes(matrix,plane2_dict[image_2_star]['matrix'])
        if res > THRESHOLD:
            print(f"match located: ({origianl_star[0]:4},{origianl_star[1]:4}) -> ({image_2_star[0]:4},{image_2_star[1]:4}) | res= {res}")
            global MAX_RES
            MAX_RES = max(res,MAX_RES)

            return (origianl_star,image_2_star)
    return None
            
def compare_matrixes(matrix1,matrix2,tolerance=0.000001):

    flat_matrix1 = matrix1.flatten()#cp.abs(matrix1.flatten())
    flat_matrix2 = matrix2.flatten()#cp.abs(matrix2.flatten())
    #matching_indices = cp.where([cp.any(cp.isclose(val, flat_matrix2, atol=tolerance)) for val in flat_matrix1])[0]
    matching_indices = cp.where(cp.isin(flat_matrix1, flat_matrix2))[0]
    rows, cols = cp.unravel_index(matching_indices, matrix1.shape)
    filtered_cols = cols[rows != cols]
    
    return len(filtered_cols) #/ len(flat_matrix1)
    
if __name__ == "__main__":
    from main import main
    main()