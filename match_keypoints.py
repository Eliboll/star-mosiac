from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
import numpy as np

def normalize_angles(angle_list):
    """
    Normalize angles to the range [0, 2π] and sort them for comparison.
    """
    return sorted([angle % (2 * np.pi) for angle in angle_list])

def compute_angular_signature(point_data):
    """
    Compute the angular signature for a given point's data.
    """
    return normalize_angles([angle for _, angle in point_data])


def compute_signature_distance_partial(sig1, sig2, max_cost=np.pi / 4):
    """
    Compute the distance between two angular signatures allowing for partial matches.

    :param sig1: First angular signature (sorted list of angles).
    :param sig2: Second angular signature (sorted list of angles).
    :param max_cost: Maximum angle mismatch allowed for matching.
    :return: The distance between the two signatures and the match percentage.
    """
    sig1 = np.array(sig1)
    sig2 = np.array(sig2)

    # Create a cost matrix for angle differences
    cost_matrix = np.abs(sig1[:, None] - sig2[None, :])
    cost_matrix = np.minimum(cost_matrix, 2 * np.pi - cost_matrix)  # Handle circular differences

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Calculate the total cost and count valid matches
    total_cost = 0
    valid_matches = 0
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < max_cost:
            total_cost += cost_matrix[r, c]
            valid_matches += 1

    # Normalize the cost by the number of matches
    if valid_matches > 0:
        average_cost = total_cost / valid_matches
    else:
        average_cost = float('inf')  # No valid matches

    match_ratio = valid_matches / max(len(sig1), len(sig2))
    return average_cost, match_ratio


def match_single_point(point1, signature1, plane2_signatures, cost_threshold, match_ratio_threshold):
    """
    Match a single point in Plane 1 against all points in Plane 2.
    """
    best_match = None
    best_cost = float('inf')
    best_ratio = 0

    for point2, signature2 in plane2_signatures.items():
        cost, match_ratio = compute_signature_distance_partial(signature1, signature2)

        if cost < best_cost and match_ratio > match_ratio_threshold and cost < cost_threshold:
            best_match = point2
            best_cost = cost
            best_ratio = match_ratio

    if best_match:
        return (point1, best_match, best_cost, best_ratio)
    else:
        return None


def find_matching_points_parallel(plane1, plane2, cost_threshold=0.1, match_ratio_threshold=0.5, n_jobs=-1):
    """
    Match points between two planes using parallel processing.

    :param plane1: Dictionary of points and their angular data for Plane 1.
    :param plane2: Dictionary of points and their angular data for Plane 2.
    :param cost_threshold: Maximum average cost for a match to be valid.
    :param match_ratio_threshold: Minimum match ratio for a match to be valid.
    :param n_jobs: Number of parallel jobs (default: -1, use all available cores).
    :return: List of matched points as tuples (point_in_plane1, point_in_plane2).
    """
    # Precompute angular signatures for all points in both planes
    plane1_signatures = {point: compute_angular_signature(data) for point, data in plane1.items()}
    plane2_signatures = {point: compute_angular_signature(data) for point, data in plane2.items()}

    # Parallelize the matching process
    results = Parallel(n_jobs=n_jobs)(
        delayed(match_single_point)(
            point1, signature1, plane2_signatures, cost_threshold, match_ratio_threshold
        )
        for point1, signature1 in plane1_signatures.items()
    )

    # Filter out None results (unmatched points)
    matches = [match for match in results if match is not None]
    matches = sorted(matches,key= lambda x: x[3], reverse=True)
    return matches