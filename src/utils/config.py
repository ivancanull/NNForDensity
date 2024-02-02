
import hashlib
import json
import yaml
import os
from ast import literal_eval
from typing import Any, Dict, List, Tuple, Union

from multimethod import multimethod


class Config(dict):
    def __getattr__(self, key: str) -> Any:
        if key not in self:
            d = self
            ## try hierarchical access
            keys = key.split(".")
            for k in keys:
                if k not in d:
                    raise AttributeError(key)
                d = d[k]
            return d
        else:
            return self[key]

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]

    def load(self, fpath: str, *, recursive: bool = False) -> None:
        if not os.path.exists(fpath):
            raise FileNotFoundError(fpath)
        fpaths = [fpath]
        if recursive:
            while fpath:
                fpath = os.path.dirname(fpath)
                for fname in ["default.yaml", "default.yml"]:
                    fpaths.append(os.path.join(fpath, fname))
        for fpath in reversed(fpaths):
            if os.path.exists(fpath):
                with open(fpath, "r") as f:
                    cfg_dict = yaml.safe_load(f)
                self.update(cfg_dict)

    def reload(self, fpath: str, *, recursive: bool = False) -> None:
        self.clear()
        self.load(fpath, recursive=recursive)

    @multimethod
    def update(self, other: Dict) -> None:
        for key, value in other.items():
            if isinstance(value, dict):
                if key not in self or not isinstance(self[key], Config):
                    self[key] = Config()
                self[key].update(value)
            else:
                self[key] = value

    @multimethod
    def update(self, opts: Union[List, Tuple]) -> None:
        index = 0
        while index < len(opts):
            opt = opts[index]
            if opt.startswith("--"):
                opt = opt[2:]
            if "=" in opt:
                key, value = opt.split("=", 1)
                index += 1
            else:
                key, value = opt, opts[index + 1]
                index += 2
            current = self
            subkeys = key.split(".")
            try:
                value = literal_eval(value)
            except:
                pass
            for subkey in subkeys[:-1]:
                current = current.setdefault(subkey, Config())
            current[subkeys[-1]] = value

    def dict(self) -> Dict[str, Any]:
        configs = dict()
        for key, value in self.items():
            if isinstance(value, Config):
                value = value.dict()
            configs[key] = value
        return configs

    def flat_dict(self) -> Dict[str, Any]:
        def _flatten_dict(dd, separator: str = "_", prefix: str = ""):
            return (
                {
                    prefix + separator + k if prefix else k: v
                    for kk, vv in dd.items()
                    for k, v in _flatten_dict(vv, separator, kk).items()
                }
                if isinstance(dd, dict)
                else {prefix: dd}
            )

        return _flatten_dict(self.dict(), separator=".")

    def hash(self) -> str:
        buffer = json.dumps(self.dict(), sort_keys=True)
        return hashlib.sha256(buffer.encode()).hexdigest()

    def dump_to_yml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.dict(), f)

    def __str__(self) -> str:
        texts = []
        for key, value in self.items():
            if isinstance(value, Config):
                seperator = "\n"
            else:
                seperator = " "
            text = key + ":" + seperator + str(value)
            lines = text.split("\n")
            for k, line in enumerate(lines[1:]):
                lines[k + 1] = (" " * 2) + line
            texts.extend(lines)
        return "\n".join(texts)


def create_folders(
    current_dir,
    configs: Config,
):
    home_dir = os.path.abspath(os.path.join(current_dir, '..'))

    # create model path
    model_dir = os.path.join(home_dir, 'models', configs.case)
    if not os.path.exists(model_dir): # and (not configs.distributed or rank == 0):
        os.makedirs(model_dir) 
    configs.model_dir = model_dir

    # create log path
    log_dir = os.path.join(home_dir, 'runs', configs.trial_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir) 
    configs.log_dir = log_dir
    
    # define fig dir as [pdf_dir, png_dir]
    pdf_dir = os.path.join(home_dir, 'figures/pdf', configs.trial_name)
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir) 
    png_dir = os.path.join(home_dir, 'figures/png', configs.trial_name)
    if not os.path.exists(png_dir):
        os.makedirs(png_dir) 
    configs.fig_dir = [pdf_dir, png_dir]

    return configs
