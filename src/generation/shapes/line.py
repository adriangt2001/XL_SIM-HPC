import numpy as np


class Line:
    """Infinite line passing through two points."""

    def __init__(self, point1: np.ndarray, point2: np.ndarray, brightness: float = 1.0, thickness: float = 0.01):
        self.point1 = np.array(point1, dtype=np.float64)
        self.point2 = np.array(point2, dtype=np.float64)
        self.brightness = float(brightness)
        self.thickness = float(thickness)

    def _point_to_line_distance(self, px: np.ndarray, py: np.ndarray, p1_px: np.ndarray, p2_px: np.ndarray) -> np.ndarray:
        x1, y1 = p1_px
        x2, y2 = p2_px

        dx = x2 - x1
        dy = y2 - y1

        line_len_sq = dx**2 + dy**2
        if line_len_sq < 1e-10:
            return np.sqrt((px - x1)**2 + (py - y1)**2)

        dist = np.abs(dy * px - dx * py + x2 * y1 - y2 * x1) / np.sqrt(line_len_sq)
        return dist

    def render(self, image: np.ndarray, aa_width: float = 1.0) -> np.ndarray:
        h, w = image.shape

        p1_px = np.array([self.point1[0] * w, self.point1[1] * h])
        p2_px = np.array([self.point2[0] * w, self.point2[1] * h])
        t_px = self.thickness * w

        y_coords, x_coords = np.ogrid[0:h, 0:w]

        line_dist = self._point_to_line_distance(x_coords, y_coords, p1_px, p2_px)

        half_thickness = t_px / 2
        alpha = np.clip(1.0 - (line_dist - half_thickness) / aa_width, 0.0, 1.0)
        alpha = np.where(line_dist <= half_thickness, 1.0, alpha)

        image = np.maximum(image, alpha * self.brightness)

        return image

    def transform(self, translation: np.ndarray, rotation: float, scale: float, center: np.ndarray) -> 'Line':
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)

        def transform_point(point):
            relative_pos = point - center
            rotated_pos = np.array([
                cos_r * relative_pos[0] - sin_r * relative_pos[1],
                sin_r * relative_pos[0] + cos_r * relative_pos[1]
            ])
            scaled_pos = rotated_pos * scale
            return center + scaled_pos + translation - center

        new_point1 = transform_point(self.point1)
        new_point2 = transform_point(self.point2)
        new_thickness = self.thickness * scale

        return Line(new_point1, new_point2, self.brightness, new_thickness)

    @staticmethod
    def generate(rng: np.random.Generator, n: int,
                 point1: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
                 point2: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
                 thickness: tuple[float, float] = (0.001, 0.01),
                 brightness: tuple[float, float] = (0.5, 1.0)) -> list['Line']:

        lines = []

        for _ in range(n):
            point1_value = np.array([
                rng.uniform(point1[0][0], point1[0][1]),
                rng.uniform(point1[1][0], point1[1][1])
            ])
            point2_value = np.array([
                rng.uniform(point2[0][0], point2[0][1]),
                rng.uniform(point2[1][0], point2[1][1])
            ])
            thickness_value = rng.uniform(thickness[0], thickness[1])
            brightness_value = rng.uniform(brightness[0], brightness[1])

            lines.append(Line(point1_value, point2_value, brightness_value, thickness_value))

        return lines

    @staticmethod
    def render_many(image: np.ndarray, lines: list['Line'], aa_width: float = 1.0) -> np.ndarray:
        for line in lines:
            image = line.render(image, aa_width=aa_width)
        return image


class Segment:
    """Line segment between two points."""

    def __init__(self, point1: np.ndarray, point2: np.ndarray, brightness: float = 1.0, thickness: float = 0.01):
        self.point1 = np.array(point1, dtype=np.float64)
        self.point2 = np.array(point2, dtype=np.float64)
        self.brightness = float(brightness)
        self.thickness = float(thickness)

    def _point_to_segment_distance_optimized(self, px: np.ndarray, py: np.ndarray, p1_px: np.ndarray, p2_px: np.ndarray) -> np.ndarray:
        x1, y1 = p1_px
        x2, y2 = p2_px

        dx = x2 - x1
        dy = y2 - y1

        segment_len_sq = dx**2 + dy**2
        if segment_len_sq < 1e-10:
            return np.sqrt((px - x1)**2 + (py - y1)**2)

        t = np.clip(((px - x1) * dx + (py - y1) * dy) / segment_len_sq, 0.0, 1.0)

        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    def _point_to_segment_distance(self, px: np.ndarray, py: np.ndarray, p1_px: np.ndarray, p2_px: np.ndarray) -> np.ndarray:
        x1, y1 = p1_px
        x2, y2 = p2_px

        dx = x2 - x1
        dy = y2 - y1

        segment_len_sq = dx**2 + dy**2
        if segment_len_sq < 1e-10:
            return np.sqrt((px - x1)**2 + (py - y1)**2)

        t = np.clip(((px - x1) * dx + (py - y1) * dy) / segment_len_sq, 0.0, 1.0)

        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    def render(self, image: np.ndarray, aa_width: float = 1.0) -> np.ndarray:
        h, w = image.shape

        p1_px = np.array([self.point1[0] * w, self.point1[1] * h])
        p2_px = np.array([self.point2[0] * w, self.point2[1] * h])
        t_px = self.thickness * w

        x_min = max(0, int(min(p1_px[0], p2_px[0]) - t_px / 2 - aa_width) - 1)
        x_max = min(w, int(max(p1_px[0], p2_px[0]) + t_px / 2 + aa_width) + 2)
        y_min = max(0, int(min(p1_px[1], p2_px[1]) - t_px / 2 - aa_width) - 1)
        y_max = min(h, int(max(p1_px[1], p2_px[1]) + t_px / 2 + aa_width) + 2)

        y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]

        segment_dist = self._point_to_segment_distance(x_coords, y_coords, p1_px, p2_px)

        half_thickness = t_px / 2
        alpha = np.clip(1.0 - (segment_dist - half_thickness) / aa_width, 0.0, 1.0)
        alpha = np.where(segment_dist <= half_thickness, 1.0, alpha)

        image[y_min:y_max, x_min:x_max] = np.maximum(
            image[y_min:y_max, x_min:x_max],
            alpha * self.brightness
        )

        return image

    def transform(self, translation: np.ndarray, rotation: float, scale: float, center: np.ndarray) -> 'Segment':
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)

        def transform_point(point):
            relative_pos = point - center
            rotated_pos = np.array([
                cos_r * relative_pos[0] - sin_r * relative_pos[1],
                sin_r * relative_pos[0] + cos_r * relative_pos[1]
            ])
            scaled_pos = rotated_pos * scale
            return center + scaled_pos + translation - center

        new_point1 = transform_point(self.point1)
        new_point2 = transform_point(self.point2)
        new_thickness = self.thickness * scale

        return Segment(new_point1, new_point2, self.brightness, new_thickness)

    @staticmethod
    def generate(rng: np.random.Generator, n: int,
                 point1: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
                 point2: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
                 thickness: tuple[float, float] = (0.001, 0.01),
                 brightness: tuple[float, float] = (0.5, 1.0)) -> list['Segment']:

        segments = []

        for _ in range(n):
            point1_value = np.array([
                rng.uniform(point1[0][0], point1[0][1]),
                rng.uniform(point1[1][0], point1[1][1])
            ])
            point2_value = np.array([
                rng.uniform(point2[0][0], point2[0][1]),
                rng.uniform(point2[1][0], point2[1][1])
            ])
            thickness_value = rng.uniform(thickness[0], thickness[1])
            brightness_value = rng.uniform(brightness[0], brightness[1])

            segments.append(Segment(point1_value, point2_value, brightness_value, thickness_value))

        return segments

    @staticmethod
    def render_many(image: np.ndarray, segments: list['Segment'], aa_width: float = 1.0) -> np.ndarray:
        for segment in segments:
            image = segment.render(image, aa_width=aa_width)
        return image


class SemiPlane:
    """Half-plane defined by a line. All points on one side of the line are filled."""

    def __init__(self, point1: np.ndarray, point2: np.ndarray, brightness: float = 1.0, side: int = 1):
        self.point1 = np.array(point1, dtype=np.float64)
        self.point2 = np.array(point2, dtype=np.float64)
        self.brightness = float(brightness)
        self.side = int(side)

    def _signed_distance_to_line(self, px: np.ndarray, py: np.ndarray, p1_px: np.ndarray, p2_px: np.ndarray) -> np.ndarray:
        x1, y1 = p1_px
        x2, y2 = p2_px

        dx = x2 - x1
        dy = y2 - y1

        signed_dist = dy * (px - x1) - dx * (py - y1)

        return signed_dist

    def render(self, image: np.ndarray, aa_width: float = 1.0) -> np.ndarray:
        h, w = image.shape

        p1_px = np.array([self.point1[0] * w, self.point1[1] * h])
        p2_px = np.array([self.point2[0] * w, self.point2[1] * h])

        y_coords, x_coords = np.ogrid[0:h, 0:w]

        signed_dist = self._signed_distance_to_line(x_coords, y_coords, p1_px, p2_px)

        if self.side > 0:
            inside = signed_dist >= 0
            edge_dist = -signed_dist
        else:
            inside = signed_dist <= 0
            edge_dist = signed_dist

        alpha = np.where(inside, 1.0, 0.0)

        boundary_mask = np.abs(edge_dist) < aa_width
        alpha = np.where(boundary_mask, np.clip(1.0 - np.abs(edge_dist) / aa_width, alpha, 1.0), alpha)

        image = np.maximum(image, alpha * self.brightness)

        return image

    def transform(self, translation: np.ndarray, rotation: float, scale: float, center: np.ndarray) -> 'SemiPlane':
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)

        def transform_point(point):
            relative_pos = point - center
            rotated_pos = np.array([
                cos_r * relative_pos[0] - sin_r * relative_pos[1],
                sin_r * relative_pos[0] + cos_r * relative_pos[1]
            ])
            scaled_pos = rotated_pos * scale
            return center + scaled_pos + translation - center

        new_point1 = transform_point(self.point1)
        new_point2 = transform_point(self.point2)

        return SemiPlane(new_point1, new_point2, self.brightness, self.side)

    @staticmethod
    def generate(rng: np.random.Generator, n: int,
                 point1: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
                 point2: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
                 brightness: tuple[float, float] = (0.5, 1.0),
                 side: int | None = None) -> list['SemiPlane']:

        semi_planes = []

        for _ in range(n):
            point1_value = np.array([
                rng.uniform(point1[0][0], point1[0][1]),
                rng.uniform(point1[1][0], point1[1][1])
            ])
            point2_value = np.array([
                rng.uniform(point2[0][0], point2[0][1]),
                rng.uniform(point2[1][0], point2[1][1])
            ])
            brightness_value = rng.uniform(brightness[0], brightness[1])
            side_value = side if side is not None else rng.choice([-1, 1])

            semi_planes.append(SemiPlane(point1_value, point2_value, brightness_value, side_value))

        return semi_planes

    @staticmethod
    def render_many(image: np.ndarray, semi_planes: list['SemiPlane'], aa_width: float = 1.0) -> np.ndarray:
        for semi_plane in semi_planes:
            image = semi_plane.render(image, aa_width=aa_width)
        return image


if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    lines = Line.generate(33, 5, t_min=0.003, t_max=0.008)
    image = np.zeros((1000, 1000))
    drawn = Line.render_many(image, lines, aa_width=1.5)
    axes[0].imshow(drawn, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Lines')
    axes[0].axis('off')

    segments = Segment.generate(42, 8, t_min=0.005, t_max=0.012)
    image = np.zeros((1000, 1000))
    drawn = Segment.render_many(image, segments, aa_width=1.5)
    axes[1].imshow(drawn, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Segments')
    axes[1].axis('off')

    semi_planes = SemiPlane.generate(51, 3, l_min=0.3, l_max=0.6)
    image = np.zeros((1000, 1000))
    drawn = SemiPlane.render_many(image, semi_planes, aa_width=2.0)
    axes[2].imshow(drawn, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Semi-Planes')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
