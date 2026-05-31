import importlib.util
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from .config import ConfigValidator


class ShapeFactory:
    def __init__(self):
        self.builtin_shapes = {
            "Circle": "src.generation.shapes.circle.Circle",
            "Circumference": "src.generation.shapes.circle.Circumference",
            "Star": "src.generation.shapes.star.Star",
            "StarContour": "src.generation.shapes.star.StarContour",
            "SheriffStar": "src.generation.shapes.sheriff_star.SheriffStar",
            "Ellipse": "src.generation.shapes.ellipse.Ellipse",
            "EllipseContour": "src.generation.shapes.ellipse.EllipseContour",
            "Polygon": "src.generation.shapes.polygon.Polygon",
            "PolygonContour": "src.generation.shapes.polygon.PolygonContour",
            "BezierCurve": "src.generation.shapes.bezier_curve.BezierCurve",
            "ClosedBezierCurve": "src.generation.shapes.bezier_curve.ClosedBezierCurve",
            "Line": "src.generation.shapes.line.Line",
            "Segment": "src.generation.shapes.line.Segment",
            "SemiPlane": "src.generation.shapes.line.SemiPlane",
            "Spline": "src.generation.shapes.spline.Spline",
            "ClosedSpline": "src.generation.shapes.spline.ClosedSpline",
            "Brownian": "src.generation.shapes.brownian.Brownian",
        }
        self.custom_shapes = {}

    def load_custom_shape(self, shape_name: str, file_path: str):
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise ValueError(f"Custom shape file not found: {file_path}")

        spec = importlib.util.spec_from_file_location(shape_name, file_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[shape_name] = module
        spec.loader.exec_module(module)

        if hasattr(module, shape_name):
            self.custom_shapes[shape_name] = getattr(module, shape_name)
        else:
            raise ValueError(f"Could not find class '{shape_name}' in {file_path}")

    def get_shape_class(self, shape_type: str):
        if shape_type in self.custom_shapes:
            return self.custom_shapes[shape_type]

        if shape_type not in self.builtin_shapes:
            raise ValueError(f"Unknown shape type: {shape_type}")

        module_path, class_name = self.builtin_shapes[shape_type].rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)


class DatasetGenerator:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.shape_factory = ShapeFactory()

        self.config = self._load_config()
        self._parse_config()

        self.rng = np.random.default_rng(self.config["seed"])

        self.output_dir = self.config["output_dir"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise ValueError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    def _parse_config(self):
        validator = ConfigValidator()

        self.config["seed"] = validator.validate_seed(self.config.get("seed", 0))
        self.config["image_size"] = validator.validate_image_size(
            self.config.get("image_size", [256, 256])
        )
        self.config["background_color"] = validator.validate_background_color(
            self.config.get("background_color", 0.0)
        )
        self.config["output_dir"] = validator.validate_output_dir(
            self.config.get("output_dir", "./output")
        )

        if "image_types" not in self.config or not isinstance(
            self.config["image_types"], list
        ):
            raise ValueError("Config must have 'image_types' list")

        for img_type in self.config["image_types"]:
            if "name" not in img_type:
                raise ValueError("Each image type must have a 'name'")

            validator.validate_count(img_type.get("count", 1))

            if "shapes" not in img_type or not isinstance(img_type["shapes"], list):
                raise ValueError(
                    f"Image type '{img_type['name']}' must have 'shapes' list"
                )

            for shape_config in img_type["shapes"]:
                validator.validate_shape_config(shape_config)

                if "custom_file" in shape_config:
                    self.shape_factory.load_custom_shape(
                        shape_config["type"], shape_config["custom_file"]
                    )

    def _create_blank_image(self) -> np.ndarray:
        h, w = self.config["image_size"]
        bg_color = self.config["background_color"]
        return np.full((h, w), bg_color, dtype=np.float64)

    def _generate_shapes(self, shape_config: Dict[str, Any], num_shapes: int) -> List:
        shape_class = self.shape_factory.get_shape_class(shape_config["type"])
        params = shape_config.get("params", {})

        try:
            shapes = shape_class.generate(self.rng, num_shapes, **params)
        except Exception as e:
            raise ValueError(f"Failed to generate {shape_config['type']}: {e}")

        return shapes

    def _generate_image(
        self, image_type_config: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[Dict]]:
        image = self._create_blank_image()
        metadata = []

        shapes = []
        for shape_config in image_type_config["shapes"]:
            count_min, count_max = shape_config["count"]
            num_shapes = self.rng.integers(int(count_min), int(count_max) + 1)

            shapes_of_type = self._generate_shapes(shape_config, num_shapes)
            shapes.extend(shapes_of_type)

        # shuffle so that the shapes are not rendered in the order they are generated
        self.rng.shuffle(shapes)

        aa_width = self.config.get("aa_width", 1.0)
        for shape in shapes:
            image = shape.render(image, aa_width=aa_width)
            metadata.append(
                {
                    "type": shape_config["type"],
                    "params": self._extract_shape_params(shape),
                }
            )

        return image, metadata

    def _extract_shape_params(self, shape_instance) -> Dict[str, Any]:
        params = {}

        for attr_name in dir(shape_instance):
            if attr_name.startswith("_"):
                continue

            attr_value = getattr(shape_instance, attr_name)

            if callable(attr_value):
                continue

            if isinstance(attr_value, np.ndarray):
                params[attr_name] = attr_value.tolist()
            elif isinstance(attr_value, list):
                if len(attr_value) > 0 and isinstance(attr_value[0], np.ndarray):
                    params[attr_name] = [v.tolist() for v in attr_value]
                else:
                    params[attr_name] = attr_value
            elif isinstance(attr_value, (int, float, str, bool, type(None))):
                params[attr_name] = attr_value

        return params

    def _save_image(
        self, image: np.ndarray, metadata: List[Dict], image_type_name: str
    ):
        filename = f"{image_type_name}_{uuid.uuid4().hex}"
        image_path = self.output_dir / f"{filename}.png"

        from PIL import Image

        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(image_uint8, mode="L").save(image_path)

        if self.config.get("save_metadata", False):
            metadata_path = self.output_dir / f"{filename}.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

    def generate(self):
        print(f"Generating dataset with seed: {self.config['seed']}")
        print(f"Output directory: {self.output_dir}")

        total_images = sum(img_type["count"] for img_type in self.config["image_types"])
        print(f"Total images to generate: {total_images}")

        generated_count = 0

        for img_type in self.config["image_types"]:
            print(
                f"\nGenerating {img_type['count']} images of type '{img_type['name']}'..."
            )

            for i in range(img_type["count"]):
                image, metadata = self._generate_image(img_type)
                self._save_image(image, metadata, img_type["name"])
                generated_count += 1

                if (i + 1) % 10 == 0 or (i + 1) == img_type["count"]:
                    print(f"  Generated {i + 1}/{img_type['count']}")

        print(f"\nDataset generation complete! Generated {generated_count} images.")


def main(args):
    generator = DatasetGenerator(args.config_shape)
    generator.generate()


if __name__ == "__main__":
    main()
