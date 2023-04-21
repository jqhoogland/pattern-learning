import hashlib

def generate_run_name(config: dict, aliases=None, droppable=None, append_hash=False, bool_aliases=None):
    if aliases is None:
        aliases = {}
        
    if bool_aliases is None:
        bool_aliases = {}

    name_parts = []

    for key, value in config.items():
        if key in aliases:
            key_alias = aliases.get(key, key)
            name_parts.append(f"{key_alias}{value}")

        elif key in bool_aliases and isinstance(value, bool):
            value_alias = bool_aliases[key].get(value)
            if value_alias is not None:
                name_parts.append(value_alias)
    
    run_name = "_".join(name_parts)

    if append_hash:
        config_str = str(sorted(config.items())).encode()
        config_hash = hashlib.md5(config_str).hexdigest()[:8]  # Get the first 8 characters of the hash
        run_name += f"_hash{config_hash}"

    return run_name
