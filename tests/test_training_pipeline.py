"""Test the full training pipeline with a tiny dataset."""

import os
import shutil
import tempfile
from pathlib import Path
import yaml

import pytest
import numpy as np

from train_futo_model.run import Trainer


def get_test_config(config_dir: Path, tmpdir: Path):
    """Create a minimal test configuration.

    Args:
        config_dir: Path to the config directory

    Returns:
        dict: Test configuration
    """
    with open(config_dir / "training" / "base.yaml") as f:
        training_config = yaml.safe_load(f)

    with open(config_dir / "model" / "base.yaml") as f:
        model_config = yaml.safe_load(f)

    with open(config_dir / "dataset" / "base.yaml") as f:
        dataset_config = yaml.safe_load(f)

    # apply overrides
    ## training
    training_config["num_epochs"] = 1
    training_config["per_device_train_batch_size"] = 2
    training_config["per_device_eval_batch_size"] = 2
    training_config["output_dir"] = (
        "${oc.env:TEST_OUTPUT_DIR,'./test_outputs'}"
    )
    training_config["logging_steps"] = 1
    training_config["save_strategy"] = "no"
    training_config["eval_strategy"] = "no"
    training_config["save_total_limit"] = 1

    ## model
    model_config["name"] = "test-model"
    model_config["pretrained_model_name"] = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    model_config["max_seq_length"] = 64
    model_config["do_lower_case"] = False
    model_config["use_marker_tokens"] = False
    model_config["matryoshka_dims"] = [64, 384]
    return {
        "project_name": "test-project",
        "action": "train",
        "model": model_config,
        "training": training_config,
        "dataset": dataset_config,
        "debug": True,
        "output_dir": str(tmpdir),
    }


class TestTrainingPipeline:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up test environment and clean up afterwards."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)

        # Set environment variables
        self.original_env = os.environ.copy()
        os.environ["TEST_OUTPUT_DIR"] = os.path.join(self.temp_dir, "outputs")

        yield  # Run the test

        # Clean up
        os.chdir(self.original_dir)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_full_training_pipeline(self, config_dir, tmp_path):
        """Test the full training pipeline with a tiny dataset."""
        # Create and convert config
        cfg = get_test_config(config_dir, tmp_path)
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(cfg)

        # Initialize trainer - this will load the dataset in debug mode
        # (100 examples)
        trainer = Trainer(cfg)

        # Run training
        trainer.train()

        # Skip model saving/loading in test to avoid downloading large models
        # Just verify the model was created and can process input
        assert trainer.model is not None

        # Test encoding with the model
        test_input = ["test query"]
        embeddings = trainer.model.encode(test_input)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(test_input)
