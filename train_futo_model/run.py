"""Training script for fine-tuning sentence-transformers models with Hydra.

This module provides a configurable training pipeline for fine-tuning
sentence-transformers models using Hydra for configuration management.
"""

import os
import logging
from pathlib import Path
from typing import Any
from collections import Counter

import hydra
from datasets import load_dataset, Dataset
from hydra.utils import instantiate
from omegaconf import DictConfig
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer,
)
from sentence_transformers.losses import (
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
    SentenceEvaluator,
)
from sentence_transformers.util import cos_sim
from sentence_transformers.training_args import BatchSamplers
from torch.profiler import profile, ProfilerActivity
from transformers.integrations import WandbCallback
import numpy as np

from prepare_data import convert_to_image

# Set up logging
logger = logging.getLogger(__name__)


class Trainer:
    """Main training class for fine-tuning sentence-transformers models.

    Parameters
    ----------
    cfg : DictConfig
        Configuration dictionary
    """

    def __init__(self, cfg: DictConfig):
        """Initialize the trainer with configuration."""
        self.cfg = cfg
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

    def setup_model(self) -> None:
        """Initialize the model with the given configuration."""
        model_name = self.cfg.model.pretrained_model_name
        logger.info("Initializing model: %s", model_name)
        self.model = SentenceTransformer(model_name)

        # Add special tokens if needed
        # if self.cfg.model.use_marker_tokens:
        #     self._add_marker_tokens()

    def _add_marker_tokens(self) -> None:
        """Add special marker tokens to the tokenizer."""
        tokenizer = self.model.tokenizer

        # Add special tokens if they don't exist
        special_tokens = [
            self.cfg.model.marker_start,
            self.cfg.model.marker_end,
        ]

        # Only add tokens that aren't already in the tokenizer
        special_tokens = [
            token
            for token in special_tokens
            if token not in tokenizer.get_vocab()
        ]

        if special_tokens:
            tokenizer.add_tokens(special_tokens)
            self.model._first_module().auto_model.resize_token_embeddings(
                len(tokenizer)
            )
            logger.info("Added special tokens: %s", special_tokens)

    def load_data(self) -> None:
        """Load and prepare the training and evaluation data."""
        logger.info("Loading dataset: %s", self.cfg.dataset.name)

        # Load the dataset
        if self.cfg.debug:
            train_dataset = load_dataset(
                self.cfg.dataset.name,
                split="train",
                streaming=True,
            ).take(100)

            eval_dataset = load_dataset(
                self.cfg.dataset.name,
                split="validation",
                streaming=True,
            ).take(100)

            test_dataset = load_dataset(
                self.cfg.dataset.name,
                split="test",
                streaming=True,
            ).take(100)

            self.train_dataset = Dataset.from_generator(
                lambda: train_dataset
            ).map(convert_to_image)
            self.eval_dataset = Dataset.from_generator(
                lambda: eval_dataset
            ).map(convert_to_image)
            self.test_dataset = Dataset.from_generator(
                lambda: test_dataset
            ).map(convert_to_image)
        else:
            datset = load_dataset(self.cfg.dataset.name).map(
                convert_to_image, num_proc=-1
            )
            self.train_dataset = self._maybe_subsample(
                datset["train"],
                proportion=self.cfg.dataset.train_subsample_proportion,
            )
            self.eval_dataset = self._maybe_subsample(
                datset["validation"],
                proportion=self.cfg.dataset.val_subsample_proportion,
            )
            self.test_dataset = self._maybe_subsample(
                datset["test"],
                proportion=self.cfg.dataset.test_subsample_proportion,
            )

        for split_name in [
            "train",
            "eval",
            "test",
        ]:
            dataset = getattr(self, f"{split_name}_dataset")
            if dataset is not None:
                dataset = dataset.rename_columns(
                    {
                        self.cfg.dataset.positive_column: "positive",
                        self.cfg.dataset.anchor_column: "anchor",
                    }
                )
                setattr(self, f"{split_name}_dataset", dataset)

    def _maybe_subsample(self, dataset: Dataset, proportion: float) -> Dataset:
        """Subsample the dataset if proportion is less than 1.

        Parameters
        ----------
        dataset : Dataset
            Dataset to subsample
        proportion : float
            Proportion of dataset to subsample

        Returns
        -------
        Dataset
            Subsampled dataset
        """
        if not 0.0 <= proportion <= 1.0:
            raise ValueError(
                f"Proportion must be between 0 and 1, got: {proportion}"
            )
        if proportion < 1.0:
            indices = np.arange(len(dataset))
            sampled_indices = np.random.choice(
                indices,
                size=int(len(indices) * proportion),
                replace=False,
            )
            return dataset.select(sampled_indices)
        return dataset

    def get_training_arguments(self) -> SentenceTransformerTrainingArguments:
        """Create training arguments from config.

        Returns
        -------
        SentenceTransformerTrainingArguments
            Training arguments
        """
        batch_sampler = getattr(
            BatchSamplers, self.cfg.training.batch_sampler.upper()
        )

        return instantiate(
            self.cfg.training.training_arguments, batch_sampler=batch_sampler
        )

    def get_loss_function(self) -> MatryoshkaLoss:
        """Create loss function from config.

        Returns
        -------
        MatryoshkaLoss
            Loss function
        """
        inner_train_loss = MultipleNegativesRankingLoss(self.model)
        return MatryoshkaLoss(
            self.model,
            inner_train_loss,
            matryoshka_dims=self.cfg.model.matryoshka_dims,
        )

    def _get_experiment_tracking_project_name(self) -> str:
        """Get experiment tracking project name from config.

        Returns
        -------
        str
            Experiment tracking project name
        """
        if self.cfg.debug:
            return "debug-" + self.cfg.project_name

        return self.cfg.project_name

    def get_callbacks(self) -> list[Any]:
        """Instantiate and return callbacks from config.

        Returns
        -------
        list[Any]
            List of instantiated callbacks
        """
        callbacks = []

        if not hasattr(self.cfg.training, "callbacks"):
            return callbacks

        for callback_cfg in self.cfg.training.callbacks.values():
            try:
                callback = instantiate(callback_cfg)

                # Set project name for wandb callback because otherwise
                # sentence transformers will use the default project name
                if isinstance(callback, WandbCallback):
                    os.environ["WANDB_PROJECT"] = (
                        self._get_experiment_tracking_project_name()
                    )

                callbacks.append(callback)
                logger.debug(
                    "Initialized callback: %s", type(callback).__name__
                )
            except Exception as e:
                logger.warning(
                    "Failed to initialize callback %s: %s",
                    callback_cfg.get("_target_", "unknown"),
                    str(e),
                )

        return callbacks

    def get_evaluator(self) -> SentenceEvaluator:
        """Get evaluator from config.

        Returns
        -------
        SentenceEvaluator
            Evaluator
        """
        if self.cfg.action == "test":
            return self.get_test_evaluator()
        elif self.cfg.action == "validate" or self.cfg.action == "train":
            return self.get_validation_evaluator()
        else:
            raise ValueError(f"Unknown action: {self.cfg.action}")

    def get_validation_evaluator(self) -> SentenceEvaluator:
        """Get validation evaluator from config.

        Returns
        -------
        SentenceEvaluator
            Validation evaluator
        """
        return self.build_evaluator(self.eval_dataset)

    def get_test_evaluator(self) -> SentenceEvaluator:
        """Get test evaluator from config.

        Returns
        -------
        SentenceEvaluator
            Test evaluator
        """
        return self.build_evaluator(self.test_dataset)

    def build_evaluator(self, dataset: Dataset) -> SequentialEvaluator:
        matryoshka_evaluators = []
        # Iterate over the different dimensions

        # get topk distractors from train dataset
        distractor_counter = Counter(dataset["positive"])
        distractors = [
            distractor
            for distractor, _ in distractor_counter.most_common(
                self.cfg.training.num_distractors
            )
        ]
        used_words = list(set(dataset["positive"])) + distractors
        corpus = {idx: word for idx, word in enumerate(used_words)}
        queries = {idx: anchor for idx, anchor in enumerate(dataset["anchor"])}
        relevant_docs = {}
        for idx, positive in enumerate(dataset["positive"]):
            relevant_docs[idx] = [used_words.index(positive)]

        for dim in self.cfg.model.matryoshka_dims:
            if (
                not self.cfg.training.eval_matryoshka_dims
                and dim != self.cfg.model.matryoshka_dims[-1]
            ):
                continue
            ir_evaluator = InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name=f"dim_{dim}",
                truncate_dim=dim,
                score_functions={"cosine": cos_sim},
            )
            matryoshka_evaluators.append(ir_evaluator)

        # Create a sequential evaluator
        evaluator = SequentialEvaluator(matryoshka_evaluators)
        return evaluator

    def train(self):
        """
        Run the training loop.

        This method initializes the model and data, prepares the training
        arguments, creates the trainer, starts the training, and saves the
        final model.
        """
        # Initialize model and data
        self.setup_model()
        self.load_data()

        # Prepare training arguments
        training_args = self.get_training_arguments()

        # Prepare dataset columns

        train_dataset = self.train_dataset.select_columns(
            ["anchor", "positive"]
        )

        # Create trainer
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            loss=self.get_loss_function(),
            evaluator=self.get_evaluator(),
            callbacks=self.get_callbacks(),
        )

        # Start training
        logger.info("Starting training...")

        if self.cfg.training.enable_profiling:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                trainer.train()
                print("CPU")
                print(
                    prof.key_averages().table(
                        sort_by="cpu_time_total", row_limit=50
                    )
                )
                print("CUDA")
                print(
                    prof.key_averages().table(
                        sort_by="cuda_time_total", row_limit=50
                    )
                )
        else:
            trainer.train()

        # Save the final model
        output_dir = Path(self.cfg.output_dir) / "final_model"
        trainer.save_model(str(output_dir))
        if not self.cfg.debug:
            trainer.push_to_hub()
        logger.info("Training complete. Model saved to %s", output_dir)

    def test(self):
        """Run the test loop."""
        # Initialize model and data
        self.setup_model()
        self.load_data()

        # Prepare dataset columns

        test_dataset = self.test_dataset.select_columns(["anchor", "positive"])

        # Create trainer
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=self.get_training_arguments(),
            train_dataset=test_dataset,
            loss=self.get_loss_function(),
            evaluator=self.get_evaluator(),
            callbacks=self.get_callbacks(),
        )

        # Start training
        logger.info("Starting test...")

        trainer.evaluate(test_dataset)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Configuration dictionary from Hydra
    """
    # Initialize and run trainer
    trainer = Trainer(cfg)
    if cfg.action == "train":
        trainer.train()
    elif cfg.action == "test":
        trainer.test()
    else:
        raise ValueError(f"Unknown action: {cfg.action}")


if __name__ == "__main__":
    main()
