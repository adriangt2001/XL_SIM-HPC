"""
Example custom composite shape: SheriffStar
A circle with a star on top, like a sheriff's badge.
"""
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import shapes
sys.path.insert(0, str(Path(__file__).parent.parent))

from shapes.circle import Circle
from shapes.star import Star
from shapes.composite import Composite


class SheriffStar:
    """A composite shape consisting of a circle with a star overlay."""

    def __init__(self, center: np.ndarray, circle_radius: float,
                 star_outer_radius: float, star_inner_radius: float,
                 n_arms: int = 5, angle: float = 0.0, brightness: float = 1.0):
        """
        Create a sheriff star (circle + star).

        Args:
            center: Center position [x, y] in normalized coordinates
            circle_radius: Radius of the background circle
            star_outer_radius: Outer radius of the star
            star_inner_radius: Inner radius of the star
            n_arms: Number of star arms (default: 5)
            angle: Rotation angle in radians
            brightness: Brightness value [0, 1]
        """
        self.center = np.array(center, dtype=np.float64)
        self.circle_radius = float(circle_radius)
        self.star_outer_radius = float(star_outer_radius)
        self.star_inner_radius = float(star_inner_radius)
        self.n_arms = int(n_arms)
        self.angle = float(angle)
        self.brightness = float(brightness)

        # Create the composite with circle and star
        # Circle is at origin (0, 0) relative to composite center
        circle = Circle(
            center=np.array([0.0, 0.0]),
            radius=circle_radius,
            brightness=brightness * 0.7  # Slightly dimmer circle
        )

        # Star is also at origin, will be rendered on top
        star = Star(
            center=np.array([0.0, 0.0]),
            outer_radius=star_outer_radius,
            inner_radius=star_inner_radius,
            n_arms=n_arms,
            angle=angle,
            brightness=brightness
        )

        # Create composite
        self.composite = Composite(
            shapes=[circle, star],
            center=center,
            rotation=0.0,
            scale=1.0
        )

    def render(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Render the sheriff star onto the image."""
        return self.composite.render(image, **kwargs)

    def transform(self, translation: np.ndarray, rotation: float,
                  scale: float, center: np.ndarray) -> 'SheriffStar':
        """Apply transformation to the sheriff star."""
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)

        relative_pos = self.center - center
        rotated_pos = np.array([
            cos_r * relative_pos[0] - sin_r * relative_pos[1],
            sin_r * relative_pos[0] + cos_r * relative_pos[1]
        ])
        scaled_pos = rotated_pos * scale
        new_center = center + scaled_pos + translation - center

        return SheriffStar(
            center=new_center,
            circle_radius=self.circle_radius * scale,
            star_outer_radius=self.star_outer_radius * scale,
            star_inner_radius=self.star_inner_radius * scale,
            n_arms=self.n_arms,
            angle=self.angle + rotation,
            brightness=self.brightness
        )

    @staticmethod
    def generate(rng: np.random.Generator, n: int,
                 center: tuple[tuple[float, float], tuple[float, float]] = ((0, 1), (0, 1)),
                 circle_radius: tuple[float, float] = (0.08, 0.15),
                 star_outer_radius: tuple[float, float] = (0.05, 0.12),
                 star_inner_radius: tuple[float, float] = (0.02, 0.06),
                 n_arms: int | tuple[int, int] = 5,
                 angle: tuple[float, float] = (0, 2 * np.pi),
                 brightness: tuple[float, float] = (0.5, 1.0)) -> list['SheriffStar']:

        sheriff_stars = []

        for _ in range(n):
            center_point = np.array([
                rng.uniform(center[0][0], center[0][1]),
                rng.uniform(center[1][0], center[1][1])
            ])
            circle_radius_value = rng.uniform(circle_radius[0], circle_radius[1])
            star_outer_radius_value = rng.uniform(star_outer_radius[0], star_outer_radius[1])
            star_inner_radius_value = rng.uniform(star_inner_radius[0], star_inner_radius[1])
            angle_value = rng.uniform(angle[0], angle[1])
            brightness_value = rng.uniform(brightness[0], brightness[1])

            if isinstance(n_arms, tuple):
                n_arms_value = rng.integers(n_arms[0], n_arms[1] + 1)
            else:
                n_arms_value = n_arms

            sheriff_stars.append(SheriffStar(
                center=center_point,
                circle_radius=circle_radius_value,
                star_outer_radius=star_outer_radius_value,
                star_inner_radius=star_inner_radius_value,
                n_arms=n_arms_value,
                angle=angle_value,
                brightness=brightness_value
            ))

        return sheriff_stars

    @staticmethod
    def render_many(image: np.ndarray, sheriff_stars: list['SheriffStar'],
                    **kwargs) -> np.ndarray:
        for sheriff_star in sheriff_stars:
            image = sheriff_star.render(image, **kwargs)
        return image
