import numpy as np

def generate_gaussian_squares(center_coords: np.array, edge_size: int, std_dev: int, num_points: int) -> np.array:
    dist_from_centroid = edge_size / 2
    x_coords = np.random.normal(center_coords[0], std_dev, num_points)
    y_coords = np.random.normal(center_coords[1], std_dev, num_points)
    square_centroids = np.column_stack((x_coords, y_coords))
    squares = np.zeros((num_points, 4, 2))

    for i in range(num_points):
        squares[i][0] = square_centroids[i][0] - dist_from_centroid, square_centroids[i][1] + dist_from_centroid
        squares[i][1] = square_centroids[i][0] + dist_from_centroid, square_centroids[i][1] + dist_from_centroid
        squares[i][2] = square_centroids[i][0] + dist_from_centroid, square_centroids[i][1] - dist_from_centroid
        squares[i][3] = square_centroids[i][0] - dist_from_centroid, square_centroids[i][1] - dist_from_centroid

    min_coords = np.min(squares[:, :, :], axis=(0, 1))
    max_coords = np.max(squares[:, :, :], axis=(0, 1))
    
    bounding_box = np.array([
        [min_coords[0], max_coords[1]],
        [max_coords[0], max_coords[1]],
        [max_coords[0], min_coords[1]],
        [min_coords[0], min_coords[1]]
    ])

    dtype = [('point_of_interest', 'float', (2)), ('area_centroids', 'float', (num_points, 2)), ('area_vertices', 'float', (num_points, 4, 2)), ('bounding_box', 'float', (4, 2))]
    data = np.zeros(num_points, dtype=dtype)

    data['point_of_interest'] = np.array(center_coords)
    data['area_centroids'] = square_centroids
    data['area_vertices'] = squares
    data['bounding_box'] = bounding_box

    return data

def save_meta_data(meta_data: np.array, out_path: str) -> None:
    np.save(out_path, meta_data)

