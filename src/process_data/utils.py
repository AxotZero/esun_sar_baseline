import yaml
import json
import itertools
import pickle
import inspect
from functools import lru_cache
from .data_config import FeatureType

def save_pickle(dic, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dic, f, protocol=4)


def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    return message_dict


def load_yaml(fp):
    return yaml.load(open(fp, "r", encoding='utf-8'),
                     Loader=yaml.SafeLoader)


def save_yaml(data, fp):
    yaml.dump(
        data,
        open(fp, "w", encoding="utf-8"),
        allow_unicode=True,
        default_flow_style=False)


def save_json(data, fp):
    with open(fp, "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, indent=4)
        f.write("{}\n".format(json_str))
        f.flush()


def load_json(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        return json.loads(f.read())


@lru_cache(None)
def get_feats_name(config):
    input_feats = []
    for name, _type in inspect.getmembers(config):
        if inspect.ismethod(_type) or name.startswith('_'):
            continue
        if getattr(config, name) in [FeatureType.NUMERICAL, FeatureType.CATEGORICAL]:
            input_feats.append(name)
    return input_feats