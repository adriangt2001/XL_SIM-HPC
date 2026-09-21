import numpy as np


class Ellipse:
    def __init__(self, center: np.ndarray, semi_major: float, semi_minor: float,
                 angle: float = 0.0, brightness: float = 1.0):
        self.center = np.array(center, dtype=np.float64)
        self.semi_major = float(semi_major)
        self.semi_minor = float(semi_minor)
        self.angle = float(angle)
        self.brightness = float(brightness)

    def render(self, image: np.ndarray, aa_width: float = 1.0) -> np.ndarray:
        h, w = image.shape

        cx_px = self.center[0] * w
        cy_px = self.center[1] * h
        a_px = self.semi_major * w
        b_px = self.semi_minor * w

        max_radius = max(a_px, b_px)
        padding = int(np.ceil(max_radius + aa_width)) + 1
        x_min = max(0, int(cx_px - padding))
        x_max = min(w, int(cx_px + padding) + 1)
        y_min = max(0, int(cy_px - padding))
        y_max = min(h, int(cy_px + padding) + 1)

        y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]

        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)

        dx = x_coords - cx_px
        dy = y_coords - cy_px

        x_rot = cos_a * dx + sin_a * dy
        y_rot = -sin_a * dx + cos_a * dy

        ellipse_dist = np.sqrt((x_rot / a_px) ** 2 + (y_rot / b_px) ** 2)
        edge_dist = ellipse_dist - 1.0

        edge_dist_px = edge_dist * min(a_px, b_px)
        alpha = np.clip(1.0 - edge_dist_px / aa_width, 0.0, 1.0)

        image[y_min:y_max, x_min:x_max] = np.maximum(
            image[y_min:y_max, x_min:x_max],
            alpha * self.brightness
        )

        return image

    @staticmethod
    def generate(rng: np.random.Generator, n: int,
                 center: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
                 semi_major: tuple[float, float] = (0.001, 0.1),
                 semi_minor: tuple[float, float] = (0.001, 0.1),
                 angle: tuple[float, float] = (0, 2 * np.pi),
                 brightness: tuple[float, float] = (0.5, 1.0)) -> list['Ellipse']:
        ellipses = []

        for _ in range(n):
            center_point = np.array([
                rng.uniform(center[0][0], center[0][1]),
                rng.uniform(center[1][0], center[1][1])
            ])
            semi_major_value = rng.uniform(semi_major[0], semi_major[1])
            semi_minor_value = rng.uniform(semi_minor[0], semi_minor[1])
            angle_value = rng.uniform(angle[0], angle[1])
            brightness_value = rng.uniform(brightness[0], brightness[1])

            ellipses.append(Ellipse(center_point, semi_major_value, semi_minor_value, angle_value, brightness_value))

        return ellipses

    def transform(self, translation: np.ndarray, rotation: float, scale: float, center: np.ndarray) -> 'Ellipse':
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)

        relative_pos = self.center - center
        rotated_pos = np.array([
            cos_r * relative_pos[0] - sin_r * relative_pos[1],
            sin_r * relative_pos[0] + cos_r * relative_pos[1]
        ])
        scaled_pos = rotated_pos * scale
        new_center = center + scaled_pos + translation - center

        new_semi_major = self.semi_major * scale
        new_semi_minor = self.semi_minor * scale
        new_angle = self.angle + rotation

        return Ellipse(new_center, new_semi_major, new_semi_minor, new_angle, self.brightness)

    @staticmethod
    def render_many(image: np.ndarray, ellipses: list['Ellipse'], aa_width: float = 1.0) -> np.ndarray:
        for ellipse in ellipses:
            image = ellipse.render(image, aa_width=aa_width)
        return image


class EllipseContour:
    def __init__(self, center: np.ndarray, semi_major: float, semi_minor: float,
                 angle: float = 0.0, brightness: float = 1.0, thickness: float = 0.01):
        self.center = np.array(center, dtype=np.float64)
        self.semi_major = float(semi_major)
        self.semi_minor = float(semi_minor)
        self.angle = float(angle)
        self.brightness = float(brightness)
        self.thickness = float(thickness)

    def render(self, image: np.ndarray, aa_width: float = 1.0) -> np.ndarray:
        h, w = image.shape

        cx_px = self.center[0] * w
        cy_px = self.center[1] * h
        a_px = self.semi_major * w
        b_px = self.semi_minor * w
        t_px = self.thickness * w

        max_radius = max(a_px, b_px)
        padding = int(np.ceil(max_radius + t_px / 2 + aa_width)) + 1
        x_min = max(0, int(cx_px - padding))
        x_max = min(w, int(cx_px + padding) + 1)
        y_min = max(0, int(cy_px - padding))
        y_max = min(h, int(cy_px + padding) + 1)

        y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]

        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)

        dx = x_coords - cx_px
        dy = y_coords - cy_px

        x_rot = cos_a * dx + sin_a * dy
        y_rot = -sin_a * dx + cos_a * dy

        ellipse_dist = np.sqrt((x_rot / a_px) ** 2 + (y_rot / b_px) ** 2)
        line_dist_normalized = np.abs(ellipse_dist - 1.0)

        line_dist_px = line_dist_normalized * min(a_px, b_px)

        half_thickness = t_px / 2
        alpha = np.clip(1.0 - (line_dist_px - half_thickness) / aa_width, 0.0, 1.0)
        alpha = np.where(line_dist_px <= half_thickness, 1.0, alpha)

        image[y_min:y_max, x_min:x_max] = np.maximum(
            image[y_min:y_max, x_min:x_max],
            alpha * self.brightness
        )

        return image

    @staticmethod
    def generate(rng: np.random.Generator, n: int,
                 center: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
                 semi_major: tuple[float, float] = (0.001, 0.1),
                 semi_minor: tuple[float, float] = (0.001, 0.1),
                 thickness: tuple[float, float] = (0.001, 0.01),
                 angle: tuple[float, float] = (0, 2 * np.pi),
                 brightness: tuple[float, float] = (0.5, 1.0)) -> list['EllipseContour']:
        contours = []

        for _ in range(n):
            center_point = np.array([
                rng.uniform(center[0][0], center[0][1]),
                rng.uniform(center[1][0], center[1][1])
            ])
            semi_major_value = rng.uniform(semi_major[0], semi_major[1])
            semi_minor_value = rng.uniform(semi_minor[0], semi_minor[1])
            angle_value = rng.uniform(angle[0], angle[1])
            thickness_value = rng.uniform(thickness[0], thickness[1])
            brightness_value = rng.uniform(brightness[0], brightness[1])

            contours.append(EllipseContour(center_point, semi_major_value, semi_minor_value, angle_value, brightness_value, thickness_value))

        return contours


    def transform(self, translation: np.ndarray, rotation: float, scale: float, center: np.ndarray) -> 'EllipseContour':
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)

        relative_pos = self.center - center
        rotated_pos = np.array([
            cos_r * relative_pos[0] - sin_r * relative_pos[1],
            sin_r * relative_pos[0] + cos_r * relative_pos[1]
        ])
        scaled_pos = rotated_pos * scale
        new_center = center + scaled_pos + translation - center

        new_semi_major = self.semi_major * scale
        new_semi_minor = self.semi_minor * scale
        new_angle = self.angle + rotation
        new_thickness = self.thickness * scale

        return EllipseContour(new_center, new_semi_major, new_semi_minor, new_angle, self.brightness, new_thickness)

    @staticmethod
    def render_many(image: np.ndarray, contours: list['EllipseContour'], aa_width: float = 1.0) -> np.ndarray:
        for contour in contours:
            image = contour.render(image, aa_width=aa_width)
        return image


if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt

    ellipses = Ellipse.generate(33, 7)

    image = np.zeros((2000, 2000))
    drawn = Ellipse.render_many(image.copy(), ellipses, aa_width=2.0)

    plt.figure(figsize=(10, 10))
    plt.imshow(drawn, cmap='gray', vmin=0, vmax=1)
    plt.title('Filled Ellipses')
    plt.axis('off')
    plt.show()

    contours = EllipseContour.generate(42, 7, t_min=0.005, t_max=0.015)

    image = np.zeros((2000, 2000))
    drawn = EllipseContour.render_many(image.copy(), contours, aa_width=1.5)

    plt.figure(figsize=(10, 10))
    plt.imshow(drawn, cmap='gray', vmin=0, vmax=1)
    plt.title('Ellipse Contours')
    plt.axis('off')
    plt.show()
