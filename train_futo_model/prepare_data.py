import os
from collections import defaultdict
from typing import Any

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
import numpy as np
from PIL import Image

from util.geometry import sample_trajectory
from util.keyboard import futo


MARKER_START_TOKEN = "<start>"
MARKER_END_TOKEN = "<end>"


def convert_to_image(
    instance: dict[str, Any], image_size: tuple[int, int] = (224, 224)
) -> dict[str, Any]:
    raw_trajectory = np.array([(i["x"], i["y"]) for i in instance["data"]])
    image = np.zeros(image_size, dtype=np.uint8)
    for x, y in (raw_trajectory * (image_size[0] - 1)).clip(
        0, image_size[0] - 1
    ):
        image[int(y), int(x)] = 255

    instance["image"] = Image.fromarray(image).convert("L")
    return instance


def _preprocess_futo(
    instances: dict[str, list[Any]],
    trajectory_representation_length: int = 30,
    num_samples_per_instance: int = 10,
    std_x_size_factor: float = 1.6,
    std_y_size_factor: float = 1.6,
) -> dict[str, list[Any]]:
    preprocessed_instances = defaultdict(list)

    for idx_instance, data in enumerate(instances["data"]):
        raw_trajectory = np.array([(i["x"], i["y"]) for i in data])

        for _ in range(num_samples_per_instance):
            sampled_trajectory = sample_trajectory(
                raw_trajectory,
                num_samples=trajectory_representation_length,
                noise_std=np.array(
                    [
                        std_x_size_factor * futo._KEY_WIDTH,
                        std_y_size_factor * futo._KEY_HEIGHT,
                    ]
                ),
            )
            trajectory_sampled_keys = [
                futo.get_key_by_position(pos) for pos in sampled_trajectory
            ]
            trajectory_sampled = sampled_trajectory.tolist()
            trajectory_word = "".join(trajectory_sampled_keys)

            tokens = instances["sentence"][idx_instance].split()
            masked_sentence = " ".join(
                tokens[: instances["word_idx"][idx_instance]]
                + [MARKER_START_TOKEN, trajectory_word, MARKER_END_TOKEN]
                + tokens[instances["word_idx"][idx_instance] + 1 :]
            )
            assert (
                tokens[instances["word_idx"][idx_instance]]
                == instances["word"][idx_instance]
            )

            preprocessed_instances["masked_sentence"].append(masked_sentence)
            preprocessed_instances["trajectory_sampled"].append(
                trajectory_sampled
            )
            preprocessed_instances["trajectory_sampled_keys"].append(
                trajectory_sampled_keys
            )
            preprocessed_instances["trajectory_word"].append(trajectory_word)

            for k, v in instances.items():
                preprocessed_instances[k].append(v[idx_instance])

    return preprocessed_instances


class DatasetBuilder:
    def __init__(
        self,
        use_futo_dataset: bool = True,
        std_x_size_factor: float = 0.6,
        std_y_size_factor: float = 0.6,
    ):
        self.use_futo_dataset = use_futo_dataset
        self.std_x_size_factor = std_x_size_factor
        self.std_y_size_factor = std_y_size_factor

    def _build_futo_dataset(self):
        dataset = load_dataset(
            "futo-org/swipe.futo.org", streaming=True
        )  # FIXME Remove take after testing
        dataset = DatasetDict(
            {
                k: Dataset.from_generator(lambda: v.take(1000))
                for k, v in dataset.items()
            }
        )
        preprocessed_data = dataset.map(
            _preprocess_futo,
            batched=True,
            fn_kwargs={
                "std_x_size_factor": self.std_x_size_factor,
                "std_y_size_factor": self.std_y_size_factor,
            },
            num_proc=os.cpu_count(),
        )
        return preprocessed_data

    def build(self):
        split_to_datasets = defaultdict(list)
        if self.use_futo_dataset:
            dataset = self._build_futo_dataset()
            for split in dataset.keys():
                split_to_datasets[split].append(dataset[split])

        return DatasetDict(
            {k: concatenate_datasets(v) for k, v in split_to_datasets.items()}
        )

    def upload_to_huggingface(self, dataset: DatasetDict):
        dataset.push_to_hub("leonweber/swipe")


if __name__ == "__main__":
    builder = DatasetBuilder()
    dataset = builder.build()

    builder.upload_to_huggingface(dataset)
