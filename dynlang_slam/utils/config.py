"""Configuration management for DynLang-SLAM."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str = "configs/default.yaml", overrides: list[str] | None = None) -> DictConfig:
    """Load configuration from YAML file with optional CLI overrides.

    Args:
        config_path: Path to YAML config file
        overrides: List of CLI overrides in format ["key=value", ...]

    Returns:
        Merged configuration as DictConfig
    """
    base_cfg = OmegaConf.load(config_path)

    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(base_cfg, cli_cfg)
    else:
        cfg = base_cfg

    return cfg


def print_config(cfg: DictConfig) -> None:
    """Pretty-print the configuration."""
    print("=" * 60)
    print("DynLang-SLAM Configuration")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
