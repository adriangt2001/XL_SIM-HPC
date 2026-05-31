import numpy as np
from typing import Protocol, runtime_checkable


@runtime_checkable
class Renderable(Protocol):
    """Protocol for any shape that can be rendered."""

    def render(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Render the shape onto an image."""
        ...

    def transform(self, translation: np.ndarray, rotation: float, scale: float, center: np.ndarray) -> 'Renderable':
        """Apply a transformation to this shape and return a new transformed instance."""
        ...


class Composite:
    def __init__(self, shapes: list[Renderable], center: np.ndarray = None,
                 rotation: float = 0.0, scale: float = 1.0):
        self.shapes = shapes
        self.center = np.array(center if center is not None else [0.5, 0.5], dtype=np.float64)
        self.rotation = float(rotation)
        self.scale = float(scale)

    def render(self, image: np.ndarray, **kwargs) -> np.ndarray:
        for shape in self.shapes:
            transformed_shape = shape.transform(
                translation=self.center,
                rotation=self.rotation,
                scale=self.scale,
                center=np.array([0.0, 0.0])
            )
            image = transformed_shape.render(image, **kwargs)

        return image

    def transform(self, translation: np.ndarray, rotation: float, scale: float, center: np.ndarray) -> 'Composite':
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)

        relative_pos = self.center - center
        rotated_pos = np.array([
            cos_r * relative_pos[0] - sin_r * relative_pos[1],
            sin_r * relative_pos[0] + cos_r * relative_pos[1]
        ])
        scaled_pos = rotated_pos * scale
        new_center = center + scaled_pos + translation - center

        new_rotation = self.rotation + rotation
        new_scale = self.scale * scale

        return Composite(self.shapes, new_center, new_rotation, new_scale)

    @staticmethod
    def render_many(image: np.ndarray, composites: list['Composite'], **kwargs) -> np.ndarray:
        for composite in composites:
            image = composite.render(image, **kwargs)
        return image
