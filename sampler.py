import numpy as np
import argparse
import pickle


def generate_gaussian_squares(center_coords: np.array, edge_size: int, std_dev: int, num_points: int) -> tuple[np.array, np.dtype]:
    dist_from_centroid = edge_size / 2
    x_coords = np.random.normal(center_coords[0], std_dev, num_points)
    y_coords = np.random.normal(center_coords[1], std_dev, num_points)
    square_centroids = np.column_stack((x_coords, y_coords))
    squares = np.zeros((num_points, 4, 2))

    for i in range(num_points):
        squares[i][0] = square_centroids[i][0] - \
            dist_from_centroid, square_centroids[i][1] + dist_from_centroid
        squares[i][1] = square_centroids[i][0] + \
            dist_from_centroid, square_centroids[i][1] + dist_from_centroid
        squares[i][2] = square_centroids[i][0] + \
            dist_from_centroid, square_centroids[i][1] - dist_from_centroid
        squares[i][3] = square_centroids[i][0] - \
            dist_from_centroid, square_centroids[i][1] - dist_from_centroid

    min_coords = np.min(squares[:, :, :], axis=(0, 1))
    max_coords = np.max(squares[:, :, :], axis=(0, 1))

    bounding_box = np.array([
        [min_coords[0], max_coords[1]],
        [max_coords[0], max_coords[1]],
        [max_coords[0], min_coords[1]],
        [min_coords[0], min_coords[1]]
    ])
    dtype = np.dtype[('point_of_interest', 'float', (2)), ('area_centroids', 'float', (num_points, 2)),
                     ('area_vertices', 'float', (num_points, 4, 2)), ('bounding_box', 'float', (4, 2))]

    data = np.zeros(num_points, dtype=dtype)

    data['point_of_interest'] = center_coords
    data['area_centroids'] = square_centroids
    data['area_vertices'] = squares
    data['bounding_box'] = bounding_box

    return data, dtype


def save_meta_data(meta_data: np.array, out_path: str) -> None:
    np.save(out_path, meta_data)


def save_dtype_to_pickle(dtype: np.dtype, out_path: str) -> None:
    with open(out_path, 'wb') as f:
        pickle.dump(dtype, f)


def main(**kwargs):
    center_coords_path = kwargs.get("center_coords_path")
    edge_size = kwargs.get("edge_size")
    std_dev = kwargs.get("std_dev")
    num_points = kwargs.get("num_points")
    meta_data_path = kwargs.get("meta_data_path")
    dtype_path = kwargs.get("dtype_path")

    center_coords = np.load(center_coords_path)
    data, dtype = generate_gaussian_squares(
        center_coords, edge_size, std_dev, num_points)
    save_meta_data(data, meta_data_path)
    save_dtype_to_pickle(dtype, dtype_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Gaussian Squares')
    parser.add_argument('--center_coords_path', type=str,
                        help='Path to npy containing center coordinates of the squares')
    parser.add_argument('--edge_size', type=int,
                        help='Edge size of the squares')
    parser.add_argument('--std_dev', type=int,
                        help='Standard deviation for generating coordinates')
    parser.add_argument('--num_points', type=int,
                        help='Number of points to generate')
    parser.add_argument('--meta_data_path', type=str, default="meta_data.npy",
                        help='Path to save the generated data')
    parser.add_argument('--dtype_path', type=str, default="dtype.pkl",
                        help='Path to save the generated dtype')

    return parser.parse_args()


if __name__ == '__main__':
    main(**parse_args())
