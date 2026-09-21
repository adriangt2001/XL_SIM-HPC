from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# n = 5000

# pts = np.cumsum(np.random.randn(3, n), axis=1)

# k = 10
# x2 = np.interp(np.arange(n * k), np.arange(n) * k, pts[0])
# y2 = np.interp(np.arange(n * k), np.arange(n) * k, pts[1])
# z2 = np.interp(np.arange(n * k), np.arange(n) * k, pts[2])

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(projection='3d')

# ax.scatter(x2, y2, z2, c=range(n*k), linewidths=0,
#            marker='o', s=3, cmap=plt.cm.jet)
# # ax.scatter(x2, y2, z2, c=range(n), linewidths=0,
# #            marker='o', s=3, cmap=plt.cm.jet)
# ax.axis('equal')
# ax.set_axis_off()
# fig.show()
# input()


class Brownian:
    def __init__(self, rng: np.random.Generator, start: np.ndarray | tuple | list, num_steps: int, dtime: int, delta: int, brightness: float = 1.0, thickness: float = 0.01):
        self.rng = rng
        self.start = np.asarray(start)
        self.num_steps = int(num_steps)
        self.dtime = float(dtime)
        self.delta = float(delta)
        self.brightness = float(brightness)
        self.thickness = float(thickness)

    def _generate_path_normalized(self):
        r = self.rng.normal(size=self.start.shape + (self.num_steps, ), scale = self.delta*sqrt(self.dtime))
        pts = np.cumsum(r, axis=-1)
        pts = pts - pts.min(axis=1)[..., np.newaxis]
        # pts = (pts - pts.min(axis=1)[..., np.newaxis]) / (pts.max(axis=1)[..., np.newaxis] - pts.min(axis=1)[..., np.newaxis])
        return pts

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

    def render(self, image: np.ndarray, aa_width: float = 1.0):
        H, W = image.shape

        pts_normalized = self._generate_path_normalized()
        pts = pts_normalized * np.array([H, W])[..., np.newaxis]
        t_px = self.thickness * W
        half_thickness = t_px / 2

        for idx in range(pts.shape[1] - 1):
            p1_px = pts[..., idx]
            p2_px = pts[..., idx + 1]

            x_min = max(0, int(min(p1_px[0], p2_px[0]) - t_px / 2 - aa_width) - 1)
            x_max = min(W, int(max(p1_px[0], p2_px[0]) + t_px / 2 + aa_width) + 2)
            y_min = max(0, int(min(p1_px[1], p2_px[1]) - t_px / 2 - aa_width) - 1)
            y_max = min(H, int(max(p1_px[1], p2_px[1]) + t_px / 2 + aa_width) + 2)
            
            y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]

            segment_dist = self._point_to_segment_distance(x_coords, y_coords, p1_px, p2_px)

            alpha = np.clip(1.0 - (segment_dist - half_thickness) / aa_width, 0.0, 1.0)
            alpha = np.where(segment_dist <= half_thickness, 1.0, alpha)

            image[y_min:y_max, x_min:x_max] = np.maximum(
                image[y_min:y_max, x_min:x_max],
                alpha * self.brightness
            )

        return image

    @staticmethod
    def generate(rng: np.random.Generator, n: int,
                 num_steps: tuple[int, int],
                 total_time: tuple[float, float],
                 delta: tuple[float, float],
                 thickness: tuple[float, float] = (0.001, 0.01),
                 brightness: tuple[float, float] = (0.5, 1.0)):
        brownians = []
        
        for _ in range(n):
            start = np.array([
                rng.uniform(0.0, 1.0),
                rng.uniform(0.0, 1.0)
            ])
            num_steps_value = rng.uniform(*num_steps)
            dtime_value = rng.uniform(*total_time) / num_steps_value
            delta_value = rng.uniform(*delta)
            thickness_value = rng.uniform(*thickness)
            brightness_value = rng.uniform(*brightness)

            brownians.append(Brownian(rng, start, num_steps_value, dtime_value, delta_value, brightness_value, thickness_value))
        return brownians

    def transform(self,):
        pass

    @staticmethod
    def render_many(image: np.ndarray, brownians: list['Brownian'], aa_width: float = 1.0) -> np.ndarray:
        for brownian in brownians:
            image = brownian.render(image, aa_width=aa_width)
        return image

if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 1, figsize=(15, 5))

    brownian = Brownian.generate(np.random.default_rng(), 2, (100, 200), (2.0, 4.0), (0.5, 0.5))
    image = np.zeros((512, 512))
    drawn = Brownian.render_many(image, brownian, aa_width=1.5)
    axes.imshow(drawn, cmap='gray', vmin=0, vmax=1)
    axes.set_title('Brownian')
    axes.axis('off')

    plt.tight_layout()
    plt.show()