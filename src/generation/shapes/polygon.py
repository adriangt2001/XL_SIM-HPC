import numpy as np


class Polygon:
    def __init__(self, center: np.ndarray, radius: float, n_sides: int,
                 angle: float = 0.0, brightness: float = 1.0):
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        self.n_sides = int(n_sides)
        self.angle = float(angle)
        self.brightness = float(brightness)

    def _get_vertices(self, w: int, h: int) -> np.ndarray:
        cx_px = self.center[0] * w
        cy_px = self.center[1] * h
        r_px = self.radius * w

        angles = np.linspace(0, 2 * np.pi, self.n_sides, endpoint=False) + self.angle
        vertices_x = cx_px + r_px * np.cos(angles)
        vertices_y = cy_px + r_px * np.sin(angles)

        return np.stack([vertices_x, vertices_y], axis=1)

    def _point_to_edge_distance(self, px: float, py: float, v1: np.ndarray, v2: np.ndarray) -> float:
        x1, y1 = v1
        x2, y2 = v2

        edge_vec = np.array([x2 - x1, y2 - y1])
        edge_len_sq = np.sum(edge_vec ** 2)

        if edge_len_sq < 1e-10:
            return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)

        t = np.clip(((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / edge_len_sq, 0.0, 1.0)

        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)

        return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    def _is_inside_polygon(self, px: np.ndarray, py: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        n = len(vertices)
        original_shape = np.broadcast_shapes(px.shape, py.shape)
        px_b, py_b = np.broadcast_arrays(px, py)
        inside = np.zeros(original_shape, dtype=bool)

        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]

            cond1 = (v1[1] > py_b) != (v2[1] > py_b)
            slope = (px_b - v1[0]) * (v2[1] - v1[1]) - (v2[0] - v1[0]) * (py_b - v1[1])
            cond2 = (v2[1] > v1[1]) == (slope > 0)

            inside ^= (cond1 & cond2)

        return inside

    def render(self, image: np.ndarray, aa_width: float = 1.0) -> np.ndarray:
        h, w = image.shape

        vertices = self._get_vertices(w, h)

        cx_px = self.center[0] * w
        cy_px = self.center[1] * h
        r_px = self.radius * w

        padding = int(np.ceil(r_px + aa_width)) + 1
        x_min = max(0, int(cx_px - padding))
        x_max = min(w, int(cx_px + padding) + 1)
        y_min = max(0, int(cy_px - padding))
        y_max = min(h, int(cy_px + padding) + 1)

        y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]

        inside = self._is_inside_polygon(x_coords, y_coords, vertices)

        min_edge_dist = np.full(x_coords.shape, np.inf, dtype=np.float64)
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]

            edge_dist = self._point_to_edge_distance(x_coords, y_coords, v1, v2)
            min_edge_dist = np.minimum(min_edge_dist, edge_dist)

        alpha = np.where(inside, 1.0, 0.0)

        boundary_mask = min_edge_dist < aa_width
        alpha = np.where(boundary_mask, np.clip(1.0 - min_edge_dist / aa_width, alpha, 1.0), alpha)

        image[y_min:y_max, x_min:x_max] = np.maximum(
            image[y_min:y_max, x_min:x_max],
            alpha * self.brightness
        )

        return image

    @staticmethod
    def generate(rng: np.random.Generator, n: int,
                 n_sides: int | tuple[int, int],
                 center: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
                 radius: tuple[float, float] = (0.001, 0.1),
                 angle: tuple[float, float] = (0, 2 * np.pi),
                 brightness: tuple[float, float] = (0.5, 1.0)) -> list['Polygon']:

        polygons = []

        for _ in range(n):
            center_point = np.array([
                rng.uniform(center[0][0], center[0][1]),
                rng.uniform(center[1][0], center[1][1])
            ])
            radius_value = rng.uniform(radius[0], radius[1])
            angle_value = rng.uniform(angle[0], angle[1])
            brightness_value = rng.uniform(brightness[0], brightness[1])

            if isinstance(n_sides, tuple) or isinstance(n_sides, list):
                n_sides_value = rng.integers(n_sides[0], n_sides[1] + 1)
            else:
                n_sides_value = n_sides

            polygons.append(Polygon(center_point, radius_value, n_sides_value, angle_value, brightness_value))

        return polygons



    def transform(self, translation: np.ndarray, rotation: float, scale: float, center: np.ndarray) -> 'Polygon':
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)

        relative_pos = self.center - center
        rotated_pos = np.array([
            cos_r * relative_pos[0] - sin_r * relative_pos[1],
            sin_r * relative_pos[0] + cos_r * relative_pos[1]
        ])
        scaled_pos = rotated_pos * scale
        new_center = center + scaled_pos + translation - center

        new_radius = self.radius * scale
        new_angle = self.angle + rotation

        return Polygon(new_center, new_radius, self.n_sides, new_angle, self.brightness)

    @staticmethod
    def render_many(image: np.ndarray, polygons: list['Polygon'], aa_width: float = 1.0) -> np.ndarray:
        for polygon in polygons:
            image = polygon.render(image, aa_width=aa_width)
        return image


class PolygonContour:
    def __init__(self, center: np.ndarray, radius: float, n_sides: int,
                 angle: float = 0.0, brightness: float = 1.0, thickness: float = 0.01):
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        self.n_sides = int(n_sides)
        self.angle = float(angle)
        self.brightness = float(brightness)
        self.thickness = float(thickness)

    def _get_vertices(self, w: int, h: int) -> np.ndarray:
        cx_px = self.center[0] * w
        cy_px = self.center[1] * h
        r_px = self.radius * w

        angles = np.linspace(0, 2 * np.pi, self.n_sides, endpoint=False) + self.angle
        vertices_x = cx_px + r_px * np.cos(angles)
        vertices_y = cy_px + r_px * np.sin(angles)

        return np.stack([vertices_x, vertices_y], axis=1)

    def _point_to_edge_distance(self, px: float, py: float, v1: np.ndarray, v2: np.ndarray) -> float:
        x1, y1 = v1
        x2, y2 = v2

        edge_vec = np.array([x2 - x1, y2 - y1])
        edge_len_sq = np.sum(edge_vec ** 2)

        if edge_len_sq < 1e-10:
            return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)

        t = np.clip(((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / edge_len_sq, 0.0, 1.0)

        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)

        return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    def render(self, image: np.ndarray, aa_width: float = 1.0) -> np.ndarray:
        h, w = image.shape

        vertices = self._get_vertices(w, h)

        cx_px = self.center[0] * w
        cy_px = self.center[1] * h
        r_px = self.radius * w
        t_px = self.thickness * w

        padding = int(np.ceil(r_px + t_px / 2 + aa_width)) + 1
        x_min = max(0, int(cx_px - padding))
        x_max = min(w, int(cx_px + padding) + 1)
        y_min = max(0, int(cy_px - padding))
        y_max = min(h, int(cy_px + padding) + 1)

        y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]

        min_edge_dist = np.full(x_coords.shape, np.inf, dtype=np.float64)
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]

            edge_dist = self._point_to_edge_distance(x_coords, y_coords, v1, v2)
            min_edge_dist = np.minimum(min_edge_dist, edge_dist)

        half_thickness = t_px / 2
        alpha = np.clip(1.0 - (min_edge_dist - half_thickness) / aa_width, 0.0, 1.0)
        alpha = np.where(min_edge_dist <= half_thickness, 1.0, alpha)

        image[y_min:y_max, x_min:x_max] = np.maximum(
            image[y_min:y_max, x_min:x_max],
            alpha * self.brightness
        )

        return image

    @staticmethod
    def generate(rng: np.random.Generator, n: int,
                 n_sides: int | tuple[int, int],
                 center: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
                 radius: tuple[float, float] = (0.001, 0.1),
                 thickness: tuple[float, float] = (0.001, 0.01),
                 angle: tuple[float, float] = (0, 2 * np.pi),
                 brightness: tuple[float, float] = (0.5, 1.0)) -> list['PolygonContour']:

        contours = []

        for _ in range(n):
            center_point = np.array([
                rng.uniform(center[0][0], center[0][1]),
                rng.uniform(center[1][0], center[1][1])
            ])
            radius_value = rng.uniform(radius[0], radius[1])
            angle_value = rng.uniform(angle[0], angle[1])
            thickness_value = rng.uniform(thickness[0], thickness[1])
            brightness_value = rng.uniform(brightness[0], brightness[1])

            if isinstance(n_sides, tuple)  or isinstance(n_sides, list):
                n_sides_value = rng.integers(n_sides[0], n_sides[1] + 1)
            else:
                n_sides_value = n_sides

            contours.append(PolygonContour(center_point, radius_value, n_sides_value, angle_value, brightness_value, thickness_value))

        return contours



    def transform(self, translation: np.ndarray, rotation: float, scale: float, center: np.ndarray) -> 'PolygonContour':
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)

        relative_pos = self.center - center
        rotated_pos = np.array([
            cos_r * relative_pos[0] - sin_r * relative_pos[1],
            sin_r * relative_pos[0] + cos_r * relative_pos[1]
        ])
        scaled_pos = rotated_pos * scale
        new_center = center + scaled_pos + translation - center

        new_radius = self.radius * scale
        new_angle = self.angle + rotation
        new_thickness = self.thickness * scale

        return PolygonContour(new_center, new_radius, self.n_sides, new_angle, self.brightness, new_thickness)

    @staticmethod
    def render_many(image: np.ndarray, contours: list['PolygonContour'], aa_width: float = 1.0) -> np.ndarray:
        for contour in contours:
            image = contour.render(image, aa_width=aa_width)
        return image


if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, n_sides in enumerate([3, 4, 5, 6, 8, 12]):
        row = idx // 3
        col = idx % 3

        polygons = Polygon.generate(33 + idx, 5, n_sides=n_sides, r_min=0.05, r_max=0.15)
        image = np.zeros((1000, 1000))
        drawn = Polygon.render_many(image, polygons, aa_width=1.5)

        axes[row, col].imshow(drawn, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'{n_sides}-sided Polygons')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, n_sides in enumerate([3, 4, 5, 6, 8, 12]):
        row = idx // 3
        col = idx % 3

        contours = PolygonContour.generate(42 + idx, 5, n_sides=n_sides,
                                          r_min=0.05, r_max=0.15,
                                          t_min=0.003, t_max=0.008)
        image = np.zeros((1000, 1000))
        drawn = PolygonContour.render_many(image, contours, aa_width=1.5)

        axes[row, col].imshow(drawn, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'{n_sides}-sided Contours')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()
