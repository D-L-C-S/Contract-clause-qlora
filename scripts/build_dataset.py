"""Build train/val/test JSONL datasets for Phi-3 clause classification from raw CUAD.

Pipeline: load CUAD -> build positive + negative ("None") examples -> split by
contract (no leakage) -> cap dominant categories in train only -> format as
Phi-3-mini-4k-instruct chat prompts -> write JSONL.

Run from anywhere via:
    uv run scripts/build_dataset.py
"""

from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer

import pandas as pd

from contract_clause_qlora.cuad import (
    PROJECT_ROOT,
    cap_categories,
    format_split,
    load_cuad,
    load_negative_examples,
    load_positive_examples,
    write_jsonl,
)

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MIN_LEN = 20
MAX_LEN = 600
TRAIN_SIZE = 0.8
CAP_PER_CATEGORY = 600
SEED = 42

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    print("Loading CUAD...")
    cuad_json = load_cuad()

    positives = load_positive_examples(cuad_json)
    negatives = load_negative_examples(cuad_json, min_len=MIN_LEN, max_len=MAX_LEN, seed=SEED)
    df = pd.DataFrame(positives + negatives)
    print(f"  {len(df)} examples ({len(positives)} positive, {len(negatives)} negative) "
          f"across {df['contract_id'].nunique()} contracts")

    print("Splitting by contract (no clause-level leakage)...")
    gss_train = GroupShuffleSplit(n_splits=1, train_size=TRAIN_SIZE, random_state=SEED)
    train_idx, temp_idx = next(gss_train.split(df, groups=df["contract_id"]))
    train_df, temp_df = df.iloc[train_idx], df.iloc[temp_idx]

    gss_val_test = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=SEED)
    val_idx, test_idx = next(gss_val_test.split(temp_df, groups=temp_df["contract_id"]))
    val_df, test_df = temp_df.iloc[val_idx], temp_df.iloc[test_idx]

    train_contracts = set(train_df["contract_id"])
    val_contracts = set(val_df["contract_id"])
    test_contracts = set(test_df["contract_id"])
    assert not (train_contracts & val_contracts), "contract leakage: train/val overlap"
    assert not (train_contracts & test_contracts), "contract leakage: train/test overlap"
    assert not (val_contracts & test_contracts), "contract leakage: val/test overlap"
    print(f"  train: {len(train_df)} rows / {len(train_contracts)} contracts")
    print(f"  val:   {len(val_df)} rows / {len(val_contracts)} contracts")
    print(f"  test:  {len(test_df)} rows / {len(test_contracts)} contracts")

    print(f"Capping dominant categories in train at {CAP_PER_CATEGORY} (val/test left uncapped)...")
    capped_train_df = cap_categories(train_df, CAP_PER_CATEGORY, seed=SEED)
    print(f"  train: {len(train_df)} -> {len(capped_train_df)} rows after capping")

    print(f"Loading tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print("Formatting splits as Phi-3 chat prompts...")
    train_examples = format_split(capped_train_df, tokenizer)
    val_examples = format_split(val_df, tokenizer)
    test_examples = format_split(test_df, tokenizer)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = {"train": train_examples, "val": val_examples, "test": test_examples}
    for name, examples in splits.items():
        path = OUTPUT_DIR / f"{name}.jsonl"
        write_jsonl(examples, path)
        print(f"  wrote {len(examples)} examples -> {path.relative_to(PROJECT_ROOT)}")

    assert len(train_examples) == len(capped_train_df)
    assert len(val_examples) == len(val_df)
    assert len(test_examples) == len(test_df)
    print("Done.")


if __name__ == "__main__":
    main()
