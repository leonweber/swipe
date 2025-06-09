# sanity_check_overlap.py
import polars as pl
from datasets import load_dataset


def check_sentence_overlap():
    # Load dataset
    dataset = load_dataset("leonweber/swipe")

    # Convert splits to Polars DataFrames
    train = dataset["train"].with_format("polars")
    val = dataset["validation"].with_format("polars")
    test = dataset["test"].with_format("polars")

    # Get unique sentences from each split
    train_sentences = train["sentence"].unique()
    val_sentences = val["sentence"].unique()
    test_sentences = test["sentence"].unique()

    # Check overlaps
    num_train_val_overlap = train_sentences.is_in(
        val_sentences.implode()
    ).sum()
    num_train_test_overlap = train_sentences.is_in(
        test_sentences.implode()
    ).sum()
    num_val_test_overlap = val_sentences.is_in(test_sentences.implode()).sum()

    # Print results
    print(
        f"Train-Val overlap: {num_train_val_overlap}/{len(val_sentences)} sentences"
    )
    print(
        f"Train-Test overlap: {num_train_test_overlap}/{len(test_sentences)} sentences"
    )
    print(
        f"Val-Test overlap: {num_val_test_overlap}/{len(val_sentences)} sentences"
    )


if __name__ == "__main__":
    check_sentence_overlap()
