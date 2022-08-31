import json
from typing import Any


def read_json_file(path: str) -> Any:
    with open(path, "r") as jfile:
        return json.load(jfile)
