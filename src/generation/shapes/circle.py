import numpy as np

class Circle:
    def __init__(self, center: np.ndarray, radius: float, brightness: float = 1.0):
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        self.brightness = float(brightness)
    
    def render(self, image: np.ndarray, aa_width: float = 1.0) -> np.ndarray:
        h, w = image.shape

        cx_px = self.center[0] * w
        cy_px = self.center[1] * h
        r_px = self.radius * w

        padding = int(np.ceil(aa_width)) + 1
        x_min = max(0, int(cx_px - r_px - padding))
        x_max = min(w, int(cx_px + r_px + padding) + 1)
        y_min = max(0, int(cy_px - r_px - padding))
        y_max = min(h, int(cy_px + r_px + padding) + 1)

        y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]
        dist = np.sqrt((x_coords - cx_px) ** 2 + (y_coords - cy_px) ** 2)
        edge_dist = dist - r_px
        alpha = np.clip(1.0 - edge_dist / aa_width, 0.0, 1.0)

        image[y_min:y_max, x_min:x_max] = np.maximum(
            image[y_min:y_max, x_min:x_max],
            alpha * self.brightness
        )

        return image

    @staticmethod
    def generate(rng: np.random.Generator, n: int,
                 center: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
                 radius: tuple[float, float] = (0.001, 0.1),
                 brightness: tuple[float, float] = (0.5, 1.0)) -> list['Circle']:
        circles = []

        for _ in range(n):
            center_point = np.array([
                rng.uniform(center[0][0], center[0][1]),
                rng.uniform(center[1][0], center[1][1])
            ])
            radius_value = rng.uniform(radius[0], radius[1])
            brightness_value = rng.uniform(brightness[0], brightness[1])

            circles.append(Circle(center_point, radius_value, brightness_value))

        return circles

    def transform(self, translation: np.ndarray, rotation: float, scale: float, center: np.ndarray) -> 'Circle':
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

        return Circle(new_center, new_radius, self.brightness)

    @staticmethod
    def render_many(image: np.ndarray, circles: list['Circle'], aa_width: float = 1.0) -> np.ndarray:
        for circle in circles:
            image = circle.render(image, aa_width=aa_width)
        return image

class Circumference:
    def __init__(self, center: np.ndarray, radius: float, brightness: float = 1.0, thickness: float = 0.01):
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        self.brightness = float(brightness)
        self.thickness = float(thickness)

    def render(self, image: np.ndarray, aa_width: float = 1.0) -> np.ndarray:
        h, w = image.shape

        cx_px = self.center[0] * w
        cy_px = self.center[1] * h
        r_px = self.radius * w
        t_px = self.thickness * w

        padding = int(np.ceil(t_px / 2 + aa_width)) + 1
        x_min = max(0, int(cx_px - r_px - padding))
        x_max = min(w, int(cx_px + r_px + padding) + 1)
        y_min = max(0, int(cy_px - r_px - padding))
        y_max = min(h, int(cy_px + r_px + padding) + 1)

        y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]
        dist = np.sqrt((x_coords - cx_px) ** 2 + (y_coords - cy_px) ** 2)
        line_dist = np.abs(dist - r_px)

        half_thickness = t_px / 2
        alpha = np.clip(1.0 - (line_dist - half_thickness) / aa_width, 0.0, 1.0)
        alpha = np.where(line_dist <= half_thickness, 1.0, alpha)

        image[y_min:y_max, x_min:x_max] = np.maximum(
            image[y_min:y_max, x_min:x_max],
            alpha * self.brightness
        )

        return image

    def transform(self, translation: np.ndarray, rotation: float, scale: float, center: np.ndarray) -> 'Circumference':
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
        new_thickness = self.thickness * scale

        return Circumference(new_center, new_radius, self.brightness, new_thickness)

    @staticmethod
    def generate(rng: np.random.Generator, n: int,
                 center: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
                 radius: tuple[float, float] = (0.001, 0.1),
                 thickness: tuple[float, float] = (0.001, 0.01),
                 brightness: tuple[float, float] = (0.5, 1.0)) -> list['Circumference']:
        circumferences = []

        for _ in range(n):
            center_point = np.array([
                rng.uniform(center[0][0], center[0][1]),
                rng.uniform(center[1][0], center[1][1])
            ])
            radius_value = rng.uniform(radius[0], radius[1])
            thickness_value = rng.uniform(thickness[0], thickness[1])
            brightness_value = rng.uniform(brightness[0], brightness[1])

            circumferences.append(Circumference(center_point, radius_value, brightness_value, thickness_value))

        return circumferences

    @staticmethod
    def render_many(image: np.ndarray, circumferences: list['Circumference'], aa_width: float = 1.0) -> np.ndarray:
        for circ in circumferences:
            image = circ.render(image, aa_width=aa_width)
        return image