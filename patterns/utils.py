import argparse
import hashlib
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from typing import Type

import yaml

import wandb


def generate_run_name(
    config: dict, aliases=None, droppable=None, append_hash=False, bool_aliases=None
):
    if aliases is None:
        aliases = {}

    if bool_aliases is None:
        bool_aliases = {}

    name_parts = []

    for key, value in config.items():
        if key in aliases:
            key_alias = aliases.get(key, key)

            if isinstance(value, (list, tuple)):
                value = "_".join(map(str, value))

            name_parts.append(f"{key_alias}{value}")

        elif key in bool_aliases and isinstance(value, bool):
            value_alias = bool_aliases[key].get(value)
            if value_alias is not None:
                name_parts.append(value_alias)

    run_name = "_".join(name_parts)

    if append_hash:
        config_str = str(sorted(config.items())).encode()
        config_hash = hashlib.md5(config_str).hexdigest()[
            :8
        ]  # Get the first 8 characters of the hash
        run_name += f"_{config_hash}"

    return run_name


@contextmanager
def wandb_run(config_cls: Type, config: dict, **kwargs):
    try:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

        if not config.get("no_wandb", False):
            kwargs.setdefault("settings", wandb.Settings(start_method="thread"))
            kwargs.setdefault("id", run_id)
            wandb.init(config=config, **kwargs)
            config_ = config_cls(**wandb.config)

            print("\nConfig:")
            print(yaml.dump(asdict(config_), default_flow_style=False))

        else:
            config_ = config_cls(**config)

        yield config_

    finally:
        if config["no_wandb"]:
            wandb.finish()

def parse_arguments(dataclass_type: Type, description="Model Configuration", **kwargs):
    parser = argparse.ArgumentParser(description=description)
    
    # Convert dataclass to dictionary to loop through fields
    default_config = asdict(dataclass_type())

    # For each field in the dataclass, add an argument to the parser
    for field, default_value in default_config.items():
        if field in kwargs:
            default_value = kwargs[field]

        if isinstance(default_value, bool):
            parser.add_argument(f"--{field}", type=bool, default=default_value, help=f"Set {field} (default: {default_value})")
        else:
            parser.add_argument(f"--{field}", type=type(default_value), default=default_value, help=f"Set {field} (default: {default_value})")
    
    args = parser.parse_args()
    
    # Convert args namespace to dictionary and then to the dataclass
    config = dataclass_type(**vars(args))

    return config
