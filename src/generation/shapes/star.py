import numpy as np


class Star:
    def __init__(self, center: np.ndarray, outer_radius: float, inner_radius: float,
                 n_arms: int, angle: float = 0.0, brightness: float = 1.0):
        self.center = np.array(center, dtype=np.float64)
        self.outer_radius = float(outer_radius)
        self.inner_radius = float(inner_radius)
        self.n_arms = int(n_arms)
        self.angle = float(angle)
        self.brightness = float(brightness)

    def _get_vertices(self, w: int, h: int) -> np.ndarray:
        cx_px = self.center[0] * w
        cy_px = self.center[1] * h
        r_outer_px = self.outer_radius * w
        r_inner_px = self.inner_radius * w

        vertices = []
        for i in range(self.n_arms * 2):
            angle = 2 * np.pi * i / (self.n_arms * 2) + self.angle
            if i % 2 == 0:
                r = r_outer_px
            else:
                r = r_inner_px

            x = cx_px + r * np.cos(angle)
            y = cy_px + r * np.sin(angle)
            vertices.append([x, y])

        return np.array(vertices)

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
        r_px = self.outer_radius * w

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
                 n_arms: int | tuple[int, int],
                 center: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
                 outer_radius: tuple[float, float] = (0.03, 0.15),
                 inner_radius: tuple[float, float] | None = None,
                 inner_ratio: tuple[float, float] = (0.3, 0.7),
                 angle: tuple[float, float] = (0, 2 * np.pi),
                 brightness: tuple[float, float] = (0.5, 1.0)) -> list['Star']:

        stars = []

        for _ in range(n):
            center_point = np.array([
                rng.uniform(center[0][0], center[0][1]),
                rng.uniform(center[1][0], center[1][1])
            ])
            outer_radius_value = rng.uniform(outer_radius[0], outer_radius[1])

            if inner_radius is not None:
                inner_radius_value = rng.uniform(inner_radius[0], inner_radius[1])
            else:
                inner_ratio_value = rng.uniform(inner_ratio[0], inner_ratio[1])
                inner_radius_value = outer_radius_value * inner_ratio_value

            angle_value = rng.uniform(angle[0], angle[1])
            brightness_value = rng.uniform(brightness[0], brightness[1])

            if isinstance(n_arms, tuple) or isinstance(n_arms, list):
                n_arms_value = rng.integers(n_arms[0], n_arms[1] + 1)
            else:
                n_arms_value = n_arms

            stars.append(Star(center_point, outer_radius_value, inner_radius_value, n_arms_value, angle_value, brightness_value))

        return stars



    def transform(self, translation: np.ndarray, rotation: float, scale: float, center: np.ndarray) -> 'Star':
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)

        relative_pos = self.center - center
        rotated_pos = np.array([
            cos_r * relative_pos[0] - sin_r * relative_pos[1],
            sin_r * relative_pos[0] + cos_r * relative_pos[1]
        ])
        scaled_pos = rotated_pos * scale
        new_center = center + scaled_pos + translation - center

        new_outer_radius = self.outer_radius * scale
        new_inner_radius = self.inner_radius * scale
        new_angle = self.angle + rotation

        return Star(new_center, new_outer_radius, new_inner_radius, self.n_arms, new_angle, self.brightness)

    @staticmethod
    def render_many(image: np.ndarray, stars: list['Star'], aa_width: float = 1.0) -> np.ndarray:
        for star in stars:
            image = star.render(image, aa_width=aa_width)
        return image


class StarContour:
    def __init__(self, center: np.ndarray, outer_radius: float, inner_radius: float,
                 n_arms: int, angle: float = 0.0, brightness: float = 1.0, thickness: float = 0.01):
        self.center = np.array(center, dtype=np.float64)
        self.outer_radius = float(outer_radius)
        self.inner_radius = float(inner_radius)
        self.n_arms = int(n_arms)
        self.angle = float(angle)
        self.brightness = float(brightness)
        self.thickness = float(thickness)

    def _get_vertices(self, w: int, h: int) -> np.ndarray:
        cx_px = self.center[0] * w
        cy_px = self.center[1] * h
        r_outer_px = self.outer_radius * w
        r_inner_px = self.inner_radius * w

        vertices = []
        for i in range(self.n_arms * 2):
            angle = 2 * np.pi * i / (self.n_arms * 2) + self.angle
            if i % 2 == 0:
                r = r_outer_px
            else:
                r = r_inner_px

            x = cx_px + r * np.cos(angle)
            y = cy_px + r * np.sin(angle)
            vertices.append([x, y])

        return np.array(vertices)

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
        r_px = self.outer_radius * w
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
                 n_arms: int | tuple[int, int],
                 center: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
                 outer_radius: tuple[float, float] = (0.03, 0.15),
                 inner_radius: tuple[float, float] | None = None,
                 inner_ratio: tuple[float, float] = (0.3, 0.7),
                 thickness: tuple[float, float] = (0.001, 0.01),
                 angle: tuple[float, float] = (0, 2 * np.pi),
                 brightness: tuple[float, float] = (0.5, 1.0)) -> list['StarContour']:

        contours = []

        for _ in range(n):
            center_point = np.array([
                rng.uniform(center[0][0], center[0][1]),
                rng.uniform(center[1][0], center[1][1])
            ])
            outer_radius_value = rng.uniform(outer_radius[0], outer_radius[1])

            if inner_radius is not None:
                inner_radius_value = rng.uniform(inner_radius[0], inner_radius[1])
            else:
                inner_ratio_value = rng.uniform(inner_ratio[0], inner_ratio[1])
                inner_radius_value = outer_radius_value * inner_ratio_value

            angle_value = rng.uniform(angle[0], angle[1])
            thickness_value = rng.uniform(thickness[0], thickness[1])
            brightness_value = rng.uniform(brightness[0], brightness[1])

            if isinstance(n_arms, tuple):
                n_arms_value = rng.integers(n_arms[0], n_arms[1] + 1)
            else:
                n_arms_value = n_arms

            contours.append(StarContour(center_point, outer_radius_value, inner_radius_value, n_arms_value, angle_value, brightness_value, thickness_value))

        return contours



    def transform(self, translation: np.ndarray, rotation: float, scale: float, center: np.ndarray) -> 'StarContour':
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)

        relative_pos = self.center - center
        rotated_pos = np.array([
            cos_r * relative_pos[0] - sin_r * relative_pos[1],
            sin_r * relative_pos[0] + cos_r * relative_pos[1]
        ])
        scaled_pos = rotated_pos * scale
        new_center = center + scaled_pos + translation - center

        new_outer_radius = self.outer_radius * scale
        new_inner_radius = self.inner_radius * scale
        new_angle = self.angle + rotation
        new_thickness = self.thickness * scale

        return StarContour(new_center, new_outer_radius, new_inner_radius, self.n_arms, new_angle, self.brightness, new_thickness)

    @staticmethod
    def render_many(image: np.ndarray, contours: list['StarContour'], aa_width: float = 1.0) -> np.ndarray:
        for contour in contours:
            image = contour.render(image, aa_width=aa_width)
        return image


if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, n_arms in enumerate([3, 4, 5, 6, 7, 8]):
        row = idx // 3
        col = idx % 3

        stars = Star.generate(33 + idx, 5, n_arms=n_arms)
        image = np.zeros((1000, 1000))
        drawn = Star.render_many(image, stars, aa_width=1.5)

        axes[row, col].imshow(drawn, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'{n_arms}-armed Stars')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, n_arms in enumerate([3, 4, 5, 6, 7, 8]):
        row = idx // 3
        col = idx % 3

        contours = StarContour.generate(42 + idx, 5, n_arms=n_arms, t_min=0.003, t_max=0.008)
        image = np.zeros((1000, 1000))
        drawn = StarContour.render_many(image, contours, aa_width=1.5)

        axes[row, col].imshow(drawn, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'{n_arms}-armed Star Contours')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()
