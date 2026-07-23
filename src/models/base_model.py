import json

from torch.nn import Module


class BaseModel(Module):
    @classmethod
    def from_json_file(cls, filename: str):
        with open(filename, mode="r") as f:
            config = json.load(f)
        return cls(**config)
