import numpy as np
import argparse
import geojson
from shapely.geometry import Polygon


def calculate_centroids(vertices: np.array) -> np.array:
    # Remove the last vertex as it is a duplicate of the first
    vertices = vertices[:, :-1, :]
    return np.mean(vertices, axis=1)


def load_geojson(geojson_path: str) -> np.array:
    with open(geojson_path) as f:
        data = geojson.load(f)
    coordinates = np.array(
        data['features'][0]['geometry']['coordinates'][0][0])
    object_ids = data['features'][0]['properties']['OBJECTID']
    return coordinates, object_ids


def generate_overlapping_squares(
        polygon: np.array,
        object_id: int,
        edge_size: float) -> np.array:

    x_min, y_min = np.min(polygon, axis=0)
    x_max, y_max = np.max(polygon, axis=0)

    x_buff = (((x_max - x_min) % edge_size) // 2)
    y_buff = (((y_max - y_min) % edge_size) // 2)

    x_start = x_min - x_buff
    y_start = y_min - y_buff
    x_end = x_max + x_buff
    y_end = y_max + y_buff

    squares = np.array([])
    x = x_start
    while x < x_end:
        y = y_start
        while y < y_end:
            square = np.array([[x, y],
                               [x + edge_size, y],
                               [x + edge_size, y + edge_size],
                               [x, y + edge_size],
                               [x, y]])
            if Polygon(square).intersects(Polygon(polygon)):
                squares.append(square)
            y += edge_size
        x += edge_size

    dtype = np.dtype[
        ('object_id', 'int', (1)),
        ('centroid', 'float', (2)),
        ('bounding_box', 'float', (5, 2))
        ('area_vertices', 'float', (squares.size, 5, 2)),
    ]

    data = np.zeros((1, 1, squares.size, 1), dtype=dtype)
    data['object_id'] = object_id
    data['centroid'] = calculate_centroids(polygon)[0].view()
    data['bounding_polygon'] = polygon
    data['polygons'] = squares

    return data


def generate_gaussian_squares(
        polygon: np.array,
        object_id: int,
        edge_size: float,
        std_dev: float,
        num_points: int,
        dtype: np.dtype) -> np.array:

    centroid = calculate_centroids(polygon)[0]
    dist_from_centroid = edge_size / 2
    x_coords = np.random.normal(centroid[0], std_dev, num_points)
    y_coords = np.random.normal(centroid[1], std_dev, num_points)
    square_centroids = np.column_stack((x_coords, y_coords))
    squares = np.zeros((num_points, 5, 2))

    for i in range(num_points):
        squares[i][0] = square_centroids[i][0] - \
            dist_from_centroid, square_centroids[i][1] + dist_from_centroid
        squares[i][1] = square_centroids[i][0] + \
            dist_from_centroid, square_centroids[i][1] + dist_from_centroid
        squares[i][2] = square_centroids[i][0] + \
            dist_from_centroid, square_centroids[i][1] - dist_from_centroid
        squares[i][3] = square_centroids[i][0] - \
            dist_from_centroid, square_centroids[i][1] - dist_from_centroid
        squares[i][4] = squares[i][0]

    data = np.zeros((1, 1, 1, num_points), dtype=dtype)
    data['object_id'] = object_id
    data['centroid'] = centroid
    data['bounding_polygon'] = polygon
    data['polygons'] = squares
    return data


def save_meta_data(meta_data: np.array, out_path: str) -> None:
    np.save(out_path, meta_data)


def main(**kwargs):
    coords_path = kwargs.get("coords_path")
    edge_size = kwargs.get("edge_size")
    meta_data_path = kwargs.get("meta_data_path")
    polygons, object_ids = load_geojson(coords_path)

    match kwargs.get("method"):
        case "overlap":
            data = np.array([generate_overlapping_squares(p, i, edge_size)
                             for p, i in zip(polygons, object_ids)])
            save_meta_data(data, meta_data_path)
        case "gaussian":
            std_dev = kwargs.get("std_dev")
            num_points = kwargs.get("num_points")
            dtype = np.dtype[('object_id', 'int', (1)),
                             ('centroid', 'float', (2)),
                             ('bounding_box', 'float', (5, 2))
                             ('polygons', 'float', (num_points, 5, 2)),]
            data = np.array([generate_gaussian_squares(
                p, i, edge_size, std_dev, num_points, dtype) for p, i in zip(polygons, object_ids)])
            save_meta_data(data, meta_data_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Gaussian Squares')
    parser.add_argument('--geojson_path', type=str,
                        help='Path to geojson file containing the polygons')
    parser.add_argument('--edge_size', type=int,
                        help='Edge size of the squares')
    parser.add_argument('--std_dev', type=int,
                        help='Standard deviation for generating coordinates')
    parser.add_argument('--num_points', type=int,
                        help='Number of points to generate')
    parser.add_argument('--meta_data_path', type=str, default="meta_data.npy",
                        help='Path to save the generated data')
    parser.add_argument('--year_range', type=int, default="1985-2022",
                        help='Years to sample from in the format "YYYY-YYYY"')
    parser.add_argument('--method', type=str, default="overlap",
                        help='Method to sample from. Either "overlap" or "gaussian"')

    return parser.parse_args()


if __name__ == '__main__':
    main(**parse_args())
