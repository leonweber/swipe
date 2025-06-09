## General Ideas
- Normalize everything to QWERTY-layout by calculating the trajectory as if it was typed on a QWERTY keyboard

## 29.05.2025

- Existing datasets for noisy typing are typically English
- I will rather do everything LLM-generated:
    - Prompt in English: `Please generate 10 random messages with the word "[WORD]" that a [AGE]-year-old [JOB??] could write on [PLATFORM] in [LANGUAGE].`
    - Then: `Now introduce typical typos and errors that a human could make when typing this message.`
    - Then diff the original and the noisy message and construct training examples from it.
- Use Gemma3 or Gemini 2.5 flash depending on what's cheaper.


## 05.06.2025

- After discussing with Alex, I have decided to focus on swiping for now
- We'll keep the swipe model separate from the typing model 
- The model should be easily extensible to new words, so we'll use an embedding model
- My current plan is to follow https://www.philschmid.de/fine-tune-embedding-model-for-rag for training
- The model should embed the trajectory as in "LLM Powered Text Entry Decoding and Flexible Typing on Smartphones", the context of the word, and a (user-extensible) dictionary


## 06.06.2025
- Implemented preprocessing
- Proper sampling (resampling with noise as in "LLM Powered Text Entry Decoding and Flexible Typing on Smartphones") is not implemented yet
    - But it is probably crucial to mitigate the effects of the discretization in constructing the trajectory string

## 07.06.2025

- Implemented rough draft of training
- Needs refactoring
- TODOs:
    - Integrate Hydra
    - Refactor spaghetti main in `train.py`
    - Try to run on runpod
    - Sampling with Gaussian noise

## 08.06.2025

- Implemented sampling with Gaussian noise

- TODOs:
    - Integrate Hydra
    - Refactor spaghetti main in `train.py`
    - Try to run on runpod
    - Define poetry entry points
    - Set up GitHub Actions
    - Maybe compute augmentations on the fly instead of storing them

## 10.06.2025

- Implemented Hydra integration
- Implemented refactoring of `train.py`

- TODOs:
    - Maybe compute augmentations on the fly instead of storing them
        - This might be tricky though because the DataLoader is not exposed in SentenceTransformers: https://github.com/UKPLab/sentence-transformers/issues/2707#issuecomment-2152759592
        - I can either use the untested workaround suggested in the issue above or the deprecated fit method outlined here: https://sbert.net/docs/sentence_transformer/training_overview.html#deprecated-training
        - Ah, there might be a fundamental issue with this approach: The evaluator will need to access the full dataset, so it has to be persistent somehow. This might still lead to OOM.
        - I should check whether I can just subsample the evaluation corpus, e.g. use a smallish dictionary + subsampled test words & queries
    - Try to run on runpod
    - Set up GitHub Actions


## 11.06.2025

- Implemented subsampling of evaluation corpus
- Decided on keeping the preprocessing as-is because this makes everything easier and I cannot spend too much time on the on-the-fly preprocessing  
- Started first run on runpod
    - Needed to subsample the evaluation corpus to 1% of the full dataset to avoid OOM
    - Maybe it is better to upsample only the training data and then maybe I can use the full eval dataset during training

- TODOs:
    - Set up GitHub Actions
    - Clean up code
    - Define proper experiments
    - Time per step seems to be increasing: Do I have a leak somewhere?
    - Avg. GPU utilization is also pretty low, I have to profile that probably D:

## 12.06.2025

- Implemented profiling
- No clear issue yet, see profile.log
    - I will run a longer training run (with num_epochs=0.0075) and check which proportions grow pointing towards an accumulation or leak


## 13.06.2025

- The slowdown was because of the NoDuplicatesBatchSampler (https://github.com/UKPLab/sentence-transformers/issues/3050)
- Using a random sampler works fine and is much faster
- Started training with random sampler on runpod
- Hits@3 are already above 0.7. I should ask Alex what his results were, what the baseline achieves and what he'd consider to be useful.

- TODOs:
    - Clean up code
    - Define proper experiments
    - Set up GitHub Actions
    - Talk to Alex about results

## 14.06.2025

- TODOs:
    - Clean up code
    - Define proper experiments
    - Set up GitHub Actions
    - Talk to Alex about results
    - Create challenge dataset  (informal language, etc.)
    - Sanity check current model
    - Implement pushing to hub after training
    - Change main metric to recall@3
    - Implement matroyshka eval after training

## 15.06.2025

- Implemented matroyshka eval
- Test set results seem fine, but I have to check whether test overlaps with train or something

- TODOs:
    - Clean up code
    - Define proper experiments
    - Set up GitHub Actions
    - Talk to Alex about results
    - Create challenge dataset  (informal language, etc.)
    - Sanity check current model
    - Implement pushing to hub after training
    - ~Change main metric to recall@3~
    - Fix experiment naming

## 16.06.2025

- Implemented sanity check
    - Found smallish overlap between train and test/dev:
    ```
    Train-Val overlap: 399/5809 sentences
    Train-Test overlap: 365/5362 sentences
    Val-Test overlap: 27/5809 sentences
    ```
    
- TODOs:
    - Clean up code
    - Define proper experiments
    - Set up GitHub Actions
    - Talk to Alex about results
    - Create challenge dataset  (informal language, etc.)
    - Fix experiment naming
    - Fix overlap issue during preprocessing
    - Fix failing tests


## 17.06.2025

- Wrote to Alex about results
- Fixed experiment and run naming
- Started to clean up code:
    - Write docs for all functions
    - Refactor where needed
    - run.py looks good, rest is still TODO

- TODOs:
    - Clean up code
    - Define proper experiments
    - Set up GitHub Actions
    - Create challenge dataset  (informal language, etc.)
    - Fix overlap issue during preprocessing
    - Fix failing tests

## 08.07.2025

- Trying whether image embedding models can be trained with sentence-transformers
    - If yes, we can try the following combinations for embedding swipe / target word:
        - image / word
        - word / word
        - image / image
        - and maybe even add context
    - We could then encode features of the swipe with color / greyscale / etc. 
    - Seems to run fine on a small subsample, but let's check on the full data

    - TODOs:
        - Check image / word on full data
         
    




