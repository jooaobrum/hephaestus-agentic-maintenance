from pathlib import Path

import yaml


def load_prompt(path: str | Path, name: str) -> str:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return str(config["prompts"][name])
