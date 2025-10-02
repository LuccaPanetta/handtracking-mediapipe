from __future__ import annotations
from pathlib import Path
import yaml
from typing import List

CONFIG_PATH = Path.cwd() / "config" / "gestures.yaml"


def load_gestures() -> List[str]:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        gestures = data.get("gestures", [])
        if isinstance(gestures, list) and gestures:
            return [str(g) for g in gestures]
    
    return ["Open", "Fist", "Peace", "ThumbsUp"] #Si el yaml no funciona te carga estos 4
