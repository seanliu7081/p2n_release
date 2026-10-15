"""Load state-dict checkpoints saved before the root package was renamed."""


def migrate_config_targets(value):
    """Copy configuration containers, changing only legacy Hydra target prefixes.

    Paths, variant IDs, source hashes, and the nested tokenizer.oat package are
    preserved. This deliberately does not restore removed policy architectures.
    """
    from omegaconf import OmegaConf

    if OmegaConf.is_config(value):
        return OmegaConf.create(migrate_config_targets(
            OmegaConf.to_container(value, resolve=False)))
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == '_target_' and isinstance(item, str) and item.startswith('oat.'):
                result[key] = 'src.' + item[len('oat.'):]
            else:
                result[key] = migrate_config_targets(item)
        return result
    if isinstance(value, list):
        return [migrate_config_targets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(migrate_config_targets(item) for item in value)
    return value


def load_checkpoint(*args, **kwargs):
    """Read a checkpoint without modifying the artifact or tensor state."""
    import torch

    payload = torch.load(*args, **kwargs)
    if isinstance(payload, dict):
        for key in ('cfg', 'policy_config', 'metadata'):
            if key in payload:
                payload[key] = migrate_config_targets(payload[key])
    return payload
