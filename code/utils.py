import json
import numpy as np


def load_config():
    with open('config/config.json') as f:
        params = json.load(f)
        return params


def inp(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) % 2


if __name__ == "__main__":
    params = load_config()
    print(params["n_bins"])
