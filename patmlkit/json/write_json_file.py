import json
from typing import Any


def write_json_file(path: str, data: Any) -> None:
    with open(path, "w") as jfile:
        return json.dump(jfile, data)
