import argparse
import json
import numpy as np

def dict2namespace(config):
    """Reference: https://github.com/ermongroup/ncsnv2/blob/master/main.py, line 155."""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace



def namespace2dict(ns):
    """Convert a Namespace object to a regular dictionary."""
    if isinstance(ns, argparse.Namespace):
        d = vars(ns)
        return {k: namespace2dict(v) for k, v in d.items()}
    elif isinstance(ns, dict):
        return {k: namespace2dict(v) for k, v in ns.items()}
    elif isinstance(ns, list):
        return [namespace2dict(x) for x in ns]
    else:
        return ns


class NumpyEncoder(json.JSONEncoder):
    """Customized json encoder for numpy array data."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() # Convert to list for json serialization.
        return json.JSONEncoder.default(self, obj)
