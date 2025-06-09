"""Configuration management using Hydra."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# Register configuration schemas
@dataclass
class ModelConfig:
    name: str
    pretrained_model_name: str
    max_seq_length: int = 512
    do_lower_case: bool = False
    use_marker_tokens: bool = True
    marker_start: str = "[START]"
    marker_end: str = "[END]"

@dataclass
class LossConfig:
    name: str
    loss_fct: str
    matryoshka_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 384, 512, 768])
    matryoshka_weights: List[float] = field(default_factory=lambda: [1.0] * 6)

@dataclass
class TrainingConfig:
    num_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    lr_scheduler_type: str
    optim: str
    tf32: bool
    bf16: bool
    output_dir: str
    logging_steps: int
    save_strategy: str
    eval_strategy: str
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    loss: LossConfig
    evaluation: Dict[str, Any]
    callbacks: List[str]
    early_stopping: Dict[str, Any]

@dataclass
class DatasetConfig:
    name: str
    hf_dataset_name: str
    train_split: str
    test_split: str
    val_split: str
    text_fields: Dict[str, str]
    streaming: bool
    num_proc: int
    batch_size: int
    shuffle_buffer_size: int
    cache_dir: str
    keep_in_memory: bool
    preprocess: Dict[str, Any]

@dataclass
class LoggingConfig:
    level: str
    format: str
    file: str

@dataclass
class Config:
    project_name: str
    debug: bool
    output_dir: str
    log_dir: str
    logging: LoggingConfig
    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig

def setup_config(config_path: str = "../configs", config_name: str = "config") -> DictConfig:
    """Set up Hydra configuration.
    
    Args:
        config_path: Path to the config directory
        config_name: Name of the config file (without .yaml)
        
    Returns:
        DictConfig: The composed configuration
    """
    # Convert to absolute path
    config_path = str(Path(__file__).parent / config_path)
    
    # Initialize Hydra
    hydra.initialize(version_base=None, config_path=config_path)
    
    # Load the configuration
    cfg = hydra.compose(config_name=config_name)
    
    # Resolve any interpolations
    return OmegaConf.to_container(cfg, resolve=True)

def register_configs():
    """Register configuration schemas with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="training", name="base_training", node=TrainingConfig)
    cs.store(group="dataset", name="base_dataset", node=DatasetConfig)
