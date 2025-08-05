import yaml

def load_config(path="src/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
