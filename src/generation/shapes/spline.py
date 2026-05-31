import numpy as np
from scipy.interpolate import CubicSpline, splprep, splev


class Spline:
    def __init__(self, control_points: list[np.ndarray], brightness: float = 1.0, thickness: float = 0.01):
        self.control_points = [np.array(cp, dtype=np.float64) for cp in control_points]
        self.brightness = float(brightness)
        self.thickness = float(thickness)

    def render(self, image: np.ndarray, samples: int = 1000, aa_width: float = 1.0) -> np.ndarray:
        h, w = image.shape

        if len(self.control_points) < 2:
            return image

        points = np.array(self.control_points)
        x_coords = points[:, 0]
        y_coords = points[:, 1]

        if len(self.control_points) == 2:
            t_param = np.linspace(0, 1, samples)
            curve_x = x_coords[0] + (x_coords[1] - x_coords[0]) * t_param
            curve_y = y_coords[0] + (y_coords[1] - y_coords[0]) * t_param
        else:
            t_param_control = np.linspace(0, 1, len(self.control_points))

            spline_x = CubicSpline(t_param_control, x_coords, bc_type='natural')
            spline_y = CubicSpline(t_param_control, y_coords, bc_type='natural')

            t_param = np.linspace(0, 1, samples)
            curve_x = spline_x(t_param)
            curve_y = spline_y(t_param)

        cx_px = curve_x * w
        cy_px = curve_y * h
        t_px = self.thickness * w

        x_min = max(0, int(np.min(cx_px) - t_px / 2 - aa_width) - 1)
        x_max = min(w, int(np.max(cx_px) + t_px / 2 + aa_width) + 2)
        y_min = max(0, int(np.min(cy_px) - t_px / 2 - aa_width) - 1)
        y_max = min(h, int(np.max(cy_px) + t_px / 2 + aa_width) + 2)

        y_coords_grid, x_coords_grid = np.ogrid[y_min:y_max, x_min:x_max]

        min_dist = np.full((y_max - y_min, x_max - x_min), np.inf, dtype=np.float64)

        for i in range(len(cx_px)):
            dx = x_coords_grid - cx_px[i]
            dy = y_coords_grid - cy_px[i]
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
                 points_range: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
                 n_points: int | tuple[int, int] = (3, 8),
                 thickness: tuple[float, float] = (0.001, 0.01),
                 brightness: tuple[float, float] = (0.5, 1.0)) -> list['Spline']:

        splines = []

        for _ in range(n):
            if isinstance(n_points, tuple) or isinstance(n_points, list):
                n_points_value = rng.integers(n_points[0], n_points[1] + 1)
            else:
                n_points_value = n_points

            control_points = []
            for _ in range(n_points_value):
                point = np.array([
                    rng.uniform(points_range[0][0], points_range[0][1]),
                    rng.uniform(points_range[1][0], points_range[1][1])
                ])
                control_points.append(point)

            thickness_value = rng.uniform(thickness[0], thickness[1])
            brightness_value = rng.uniform(brightness[0], brightness[1])

            splines.append(Spline(control_points, brightness_value, thickness_value))

        return splines



    def transform(self, translation: np.ndarray, rotation: float, scale: float, center: np.ndarray) -> 'Spline':
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

        return Spline(new_control_points, self.brightness, new_thickness)

    @staticmethod
    def render_many(image: np.ndarray, splines: list['Spline'],
                   samples: int = 1000, aa_width: float = 1.0) -> np.ndarray:
        for spline in splines:
            image = spline.render(image, samples=samples, aa_width=aa_width)
        return image


class ClosedSpline:
    def __init__(self, control_points: list[np.ndarray], brightness: float = 1.0, thickness: float = 0.01):
        self.control_points = [np.array(cp, dtype=np.float64) for cp in control_points]
        self.brightness = float(brightness)
        self.thickness = float(thickness)

    def render(self, image: np.ndarray, samples: int = 1000, aa_width: float = 1.0) -> np.ndarray:
        h, w = image.shape

        if len(self.control_points) < 3:
            return image

        points = np.array(self.control_points)

        points_closed = np.vstack([points, points[0:1]])
        x_coords = points_closed[:, 0]
        y_coords = points_closed[:, 1]

        t_param_control = np.linspace(0, 1, len(points_closed))

        spline_x = CubicSpline(t_param_control, x_coords, bc_type='periodic')
        spline_y = CubicSpline(t_param_control, y_coords, bc_type='periodic')

        t_param = np.linspace(0, 1, samples, endpoint=False)
        curve_x = spline_x(t_param)
        curve_y = spline_y(t_param)

        cx_px = curve_x * w
        cy_px = curve_y * h
        t_px = self.thickness * w

        x_min = max(0, int(np.min(cx_px) - t_px / 2 - aa_width) - 1)
        x_max = min(w, int(np.max(cx_px) + t_px / 2 + aa_width) + 2)
        y_min = max(0, int(np.min(cy_px) - t_px / 2 - aa_width) - 1)
        y_max = min(h, int(np.max(cy_px) + t_px / 2 + aa_width) + 2)

        y_coords_grid, x_coords_grid = np.ogrid[y_min:y_max, x_min:x_max]

        min_dist = np.full((y_max - y_min, x_max - x_min), np.inf, dtype=np.float64)

        for i in range(len(cx_px)):
            dx = x_coords_grid - cx_px[i]
            dy = y_coords_grid - cy_px[i]
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
                 brightness: tuple[float, float] = (0.5, 1.0)) -> list['ClosedSpline']:

        splines = []

        for _ in range(n):
            if isinstance(n_points, tuple) or isinstance(n_points, list):
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

            splines.append(ClosedSpline(control_points, brightness_value, thickness_value))

        return splines



    def transform(self, translation: np.ndarray, rotation: float, scale: float, center: np.ndarray) -> 'ClosedSpline':
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

        return ClosedSpline(new_control_points, self.brightness, new_thickness)

    @staticmethod
    def render_many(image: np.ndarray, splines: list['ClosedSpline'],
                   samples: int = 1000, aa_width: float = 1.0) -> np.ndarray:
        for spline in splines:
            image = spline.render(image, samples=samples, aa_width=aa_width)
        return image


if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt

    splines = Spline.generate(196, 10, n_points_min=4, n_points_max=8, t_min=0.005, t_max=0.015)

    image = np.zeros((2000, 2000))
    drawn = Spline.render_many(image, splines, samples=1000, aa_width=1.5)

    plt.figure(figsize=(10, 10))
    plt.imshow(drawn, cmap='gray', vmin=0, vmax=1)
    plt.title('Splines')
    plt.axis('off')
    plt.show()

    closed_splines = ClosedSpline.generate(42, 8, n_points_min=5, n_points_max=10, t_min=0.005, t_max=0.015)

    image = np.zeros((2000, 2000))
    drawn = ClosedSpline.render_many(image, closed_splines, samples=2000, aa_width=1.5)

    plt.figure(figsize=(10, 10))
    plt.imshow(drawn, cmap='gray', vmin=0, vmax=1)
    plt.title('Closed Splines')
    plt.axis('off')
    plt.show()
