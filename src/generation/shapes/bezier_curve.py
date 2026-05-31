import numpy as np


def de_casteljau(control_points, t):
    points = control_points[:]

    while len(points) > 1:
        new_points = []
        for i in range(len(points) - 1):
            x = (1 - t) * points[i][0] + t * points[i + 1][0]
            y = (1 - t) * points[i][1] + t * points[i + 1][1]
            new_points.append((x, y))
        points = new_points

    return points[0]


class BezierCurve:
    def __init__(self, start: np.ndarray, controls: list[np.ndarray], end: np.ndarray,
                 brightness: float = 1.0, thickness: float = 0.01):
        self.start = np.array(start, dtype=np.float64)
        self.controls = [np.array(c, dtype=np.float64) for c in controls]
        self.end = np.array(end, dtype=np.float64)
        self.brightness = float(brightness)
        self.thickness = float(thickness)

    def render(self, image: np.ndarray, samples: int = 1000, aa_width: float = 1.0) -> np.ndarray:
        h, w = image.shape

        control_points = [self.start] + self.controls + [self.end]

        curve_points = []
        for i in range(samples):
            t = i / (samples - 1)
            pt = de_casteljau(control_points, t)
            curve_points.append(pt)

        curve_points = np.array(curve_points)

        cx_px = curve_points[:, 0] * w
        cy_px = curve_points[:, 1] * h
        t_px = self.thickness * w

        x_min = max(0, int(np.min(cx_px) - t_px / 2 - aa_width) - 1)
        x_max = min(w, int(np.max(cx_px) + t_px / 2 + aa_width) + 2)
        y_min = max(0, int(np.min(cy_px) - t_px / 2 - aa_width) - 1)
        y_max = min(h, int(np.max(cy_px) + t_px / 2 + aa_width) + 2)

        y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]

        min_dist = np.full((y_max - y_min, x_max - x_min), np.inf, dtype=np.float64)

        for i in range(len(cx_px)):
            dx = x_coords - cx_px[i]
            dy = y_coords - cy_px[i]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            min_dist = np.minimum(min_dist, dist)

        half_thickness = t_px / 2
        alpha = np.clip(1.0 - (min_dist - half_thickness) / aa_width, 0.0, 1.0)
        alpha = np.where(min_dist <= half_thickness, 1.0, alpha)

        image[y_min:y_max, x_min:x_max] = np.maximum(
            image[y_min:y_max, x_min:x_max],
            alpha * self.brightness
        )

        return image

    @staticmethod
    def generate(rng: np.random.Generator, n: int,
                 start: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
                 end: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
                 control_points_range: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
                 n_controls: int | tuple[int, int] = (0, 5),
                 thickness: tuple[float, float] = (0.001, 0.01),
                 brightness: tuple[float, float] = (0.5, 1.0)) -> list['BezierCurve']:
        curves = []

        for _ in range(n):
            if isinstance(n_controls, tuple)  or isinstance(n_controls, list):
                n_controls_value = rng.integers(n_controls[0], n_controls[1] + 1)
            else:
                n_controls_value = n_controls

            start_point = np.array([
                rng.uniform(start[0][0], start[0][1]),
                rng.uniform(start[1][0], start[1][1])
            ])
            end_point = np.array([
                rng.uniform(end[0][0], end[0][1]),
                rng.uniform(end[1][0], end[1][1])
            ])
            controls = [
                np.array([
                    rng.uniform(control_points_range[0][0], control_points_range[0][1]),
                    rng.uniform(control_points_range[1][0], control_points_range[1][1])
                ]) for _ in range(n_controls_value)
            ]
            thickness_value = rng.uniform(thickness[0], thickness[1])
            brightness_value = rng.uniform(brightness[0], brightness[1])

            curves.append(BezierCurve(start_point, controls, end_point, brightness_value, thickness_value))

        return curves



    def transform(self, translation: np.ndarray, rotation: float, scale: float, center: np.ndarray) -> 'BezierCurve':
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

        new_start = transform_point(self.start)
        new_end = transform_point(self.end)
        new_controls = [transform_point(cp) for cp in self.controls]
        new_thickness = self.thickness * scale

        return BezierCurve(new_start, new_controls, new_end, self.brightness, new_thickness)

    @staticmethod
    def render_many(image: np.ndarray, curves: list['BezierCurve'],
                   samples: int = 1000, aa_width: float = 1.0) -> np.ndarray:
        for curve in curves:
            image = curve.render(image, samples=samples, aa_width=aa_width)
        return image


class ClosedBezierCurve:
    def __init__(self, control_points: list[np.ndarray], brightness: float = 1.0, thickness: float = 0.01):
        self.control_points = [np.array(cp, dtype=np.float64) for cp in control_points]
        self.brightness = float(brightness)
        self.thickness = float(thickness)

    def render(self, image: np.ndarray, samples: int = 1000, aa_width: float = 1.0) -> np.ndarray:
        h, w = image.shape

        n = len(self.control_points)
        if n < 3:
            return image

        curve_points = []

        for seg in range(n):
            p0 = self.control_points[seg]
            p1 = self.control_points[(seg + 1) % n]
            p2 = self.control_points[(seg + 2) % n]

            for i in range(samples // n):
                t = i / (samples // n)

                a = p0 + (p1 - p0) * t
                b = p1 + (p2 - p1) * t
                point = a + (b - a) * t

                curve_points.append(point)

        curve_points = np.array(curve_points)

        cx_px = curve_points[:, 0] * w
        cy_px = curve_points[:, 1] * h
        t_px = self.thickness * w

        x_min = max(0, int(np.min(cx_px) - t_px / 2 - aa_width) - 1)
        x_max = min(w, int(np.max(cx_px) + t_px / 2 + aa_width) + 2)
        y_min = max(0, int(np.min(cy_px) - t_px / 2 - aa_width) - 1)
        y_max = min(h, int(np.max(cy_px) + t_px / 2 + aa_width) + 2)

        y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]

        min_dist = np.full((y_max - y_min, x_max - x_min), np.inf, dtype=np.float64)

        for i in range(len(cx_px)):
            dx = x_coords - cx_px[i]
            dy = y_coords - cy_px[i]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            min_dist = np.minimum(min_dist, dist)

        half_thickness = t_px / 2
        alpha = np.clip(1.0 - (min_dist - half_thickness) / aa_width, 0.0, 1.0)
        alpha = np.where(min_dist <= half_thickness, 1.0, alpha)

        image[y_min:y_max, x_min:x_max] = np.maximum(
            image[y_min:y_max, x_min:x_max],
            alpha * self.brightness
        )

        return image

    @staticmethod
    def generate(rng: np.random.Generator, n: int,
                 center: tuple[tuple[float, float], tuple[float, float]] = ((0.1, 0.9), (0.1, 0.9)),
                 radius: tuple[float, float] = (0.05, 0.15),
                 radius_variation: tuple[float, float] = (0.5, 1.5),
                 n_points: int | tuple[int, int] = (3, 8),
                 thickness: tuple[float, float] = (0.001, 0.01),
                 brightness: tuple[float, float] = (0.5, 1.0)) -> list['ClosedBezierCurve']:
        curves = []

        for _ in range(n):
            if isinstance(n_points, tuple)  or isinstance(n_points, list):
                n_points_value = rng.integers(n_points[0], n_points[1] + 1)
            else:
                n_points_value = n_points

            center_point = np.array([
                rng.uniform(center[0][0], center[0][1]),
                rng.uniform(center[1][0], center[1][1])
            ])
            radius_value = rng.uniform(radius[0], radius[1])

            angles = np.linspace(0, 2 * np.pi, n_points_value, endpoint=False)
            angles += rng.uniform(0, 2 * np.pi)

            control_points = []
            for angle in angles:
                r = radius_value * rng.uniform(radius_variation[0], radius_variation[1])
                point = center_point + r * np.array([np.cos(angle), np.sin(angle)])
                control_points.append(point)

            thickness_value = rng.uniform(thickness[0], thickness[1])
            brightness_value = rng.uniform(brightness[0], brightness[1])

            curves.append(ClosedBezierCurve(control_points, brightness_value, thickness_value))

        return curves



    def transform(self, translation: np.ndarray, rotation: float, scale: float, center: np.ndarray) -> 'ClosedBezierCurve':
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

        new_control_points = [transform_point(cp) for cp in self.control_points]
        new_thickness = self.thickness * scale

        return ClosedBezierCurve(new_control_points, self.brightness, new_thickness)

    @staticmethod
    def render_many(image: np.ndarray, curves: list['ClosedBezierCurve'],
                   samples: int = 1000, aa_width: float = 1.0) -> np.ndarray:
        for curve in curves:
            image = curve.render(image, samples=samples, aa_width=aa_width)
        return image


if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt

    curves = BezierCurve.generate(196, 10, t_min=0.005, t_max=0.015)

    image = np.zeros((2000, 2000))
    drawn = BezierCurve.render_many(image, curves, samples=1000, aa_width=1.5)

    plt.figure(figsize=(10, 10))
    plt.imshow(drawn, cmap='gray', vmin=0, vmax=1)
    plt.title('Bezier Curves')
    plt.axis('off')
    plt.show()

    closed_curves = ClosedBezierCurve.generate(42, 8, t_min=0.005, t_max=0.015)

    image = np.zeros((2000, 2000))
    drawn = ClosedBezierCurve.render_many(image, closed_curves, samples=2000, aa_width=1.5)

    plt.figure(figsize=(10, 10))
    plt.imshow(drawn, cmap='gray', vmin=0, vmax=1)
    plt.title('Closed Bezier Curves')
    plt.axis('off')
    plt.show()
