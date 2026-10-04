"""OmegaConf YAML and CLI configuration loading, adapted from TraceRL."""

from __future__ import annotations

from omegaconf import OmegaConf


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf
