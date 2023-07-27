import hashlib
from contextlib import contextmanager
from datetime import datetime

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
def wandb_run(**kwargs):
    try:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        kwargs.setdefault("settings", wandb.Settings(start_method="thread"))
        kwargs.setdefault("id", run_id)
        wandb.init(**kwargs)

        yield

    finally:
        wandb.finish()
