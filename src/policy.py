import yaml

def load_policy(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f.read())
    if data is None:
        raise ValueError(f"Policy file parsed as None (empty or invalid): {path}")
    return data