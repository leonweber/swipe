import datasets
from typing import Any
import numpy as np
from util.keyboard import get_key_by_position
from util.geometry import sample_trajectory


        
        

def preprocess(instance: dict[str, Any], num_samples: int = 30) -> dict[str, Any]:
    raw_trajectory = np.array([(i["x"], i["y"]) for i in instance["data"]])
    sampled_trajectory = sample_trajectory(raw_trajectory, num_samples=num_samples)
    trajectory_sampled_keys = [get_key_by_position(pos) for pos in sampled_trajectory]
    instance["trajectory_sampled"] = sampled_trajectory.tolist()
    instance["trajectory_sampled_keys"] = trajectory_sampled_keys
    trajectory_word = "".join(trajectory_sampled_keys)
    # instance["masked_sentence"] = instance["sentence"].replace(instance["word"], trajectory_word, 1) 
    tokens = instance["sentence"].split()
    instance["masked_sentence"] = " ".join(tokens[:instance["word_idx"]] + [MARKER_START_TOKEN, trajectory_word, MARKER_END_TOKEN] + tokens[instance["word_idx"]+1:])
    assert tokens[instance["word_idx"]] == instance["word"]


    return instance
    

