from pathlib import Path
from typing import Any, Dict, Tuple


class ConfigValidator:
    @staticmethod
    def validate_seed(seed: Any) -> int:
        if not isinstance(seed, int) or seed < 0:
            raise ValueError(f"Seed must be a non-negative integer, got: {seed}")
        return seed

    @staticmethod
    def validate_image_size(size: Any) -> Tuple[int, int]:
        if not isinstance(size, list) or len(size) != 2:
            raise ValueError(f"Image size must be [width, height], got: {size}")
        w, h = size
        if not isinstance(w, int) or not isinstance(h, int) or w <= 0 or h <= 0:
            raise ValueError(f"Image dimensions must be positive integers, got: {size}")
        return (w, h)

    @staticmethod
    def validate_background_color(color: Any) -> float:
        if not isinstance(color, (int, float)) or color < 0 or color > 1:
            raise ValueError(f"Background color must be between 0 and 1, got: {color}")
        return float(color)

    @staticmethod
    def validate_output_dir(path: Any) -> Path:
        if not isinstance(path, str):
            raise ValueError(f"Output directory must be a string, got: {path}")

        output_path = Path(path)

        if output_path.exists():
            if not output_path.is_dir():
                raise ValueError(f"Output path exists but is not a directory: {path}")
            if any(output_path.iterdir()):
                raise ValueError(f"Output directory is not empty: {path}")

        return output_path

    @staticmethod
    def validate_count(count: Any) -> int:
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"Image count must be a positive integer, got: {count}")
        return count

    @staticmethod
    def validate_range(param_range: Any, param_name: str) -> Tuple[float, float]:
        if not isinstance(param_range, list) or len(param_range) != 2:
            raise ValueError(f"{param_name} must be [min, max], got: {param_range}")

        min_val, max_val = param_range
        if not isinstance(min_val, (int, float)) or not isinstance(
            max_val, (int, float)
        ):
            raise ValueError(f"{param_name} values must be numeric, got: {param_range}")

        if min_val > max_val:
            raise ValueError(f"{param_name} min ({min_val}) > max ({max_val})")

        return (float(min_val), float(max_val))

    @staticmethod
    def validate_shape_config(shape_config: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(shape_config, dict):
            raise ValueError(f"Shape config must be a dictionary, got: {shape_config}")

        if "type" not in shape_config:
            raise ValueError("Shape config must have a 'type' field")

        if "count" not in shape_config:
            raise ValueError(
                f"Shape config for {shape_config['type']} must have a 'count' field"
            )

        ConfigValidator.validate_range(
            shape_config["count"], f"{shape_config['type']}.count"
        )

        return shape_config
