"""Load and process the CUAD (Contract Understanding Atlas Dataset) SQuAD-format JSON
into (contract_id, category, text) example rows suitable for clause classification.

Positive examples come from CUAD's annotated answer spans (one row per span, per
category). Negative ("None") examples are constructed by finding the stretches of
each contract's text that no category claims, then chunking those stretches into
clause-length pieces on natural boundaries (paragraph breaks, then sentence breaks).
"""

import json
import random
import re
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUAD_PATH = PROJECT_ROOT / "data" / "cuad-raw" / "CUADv1.json"


def load_cuad(path: Path = DEFAULT_CUAD_PATH) -> dict:
    """Load CUAD's SQuAD-format JSON from disk.

    Args:
        path: Path to CUADv1.json. Defaults to data/cuad-raw/CUADv1.json under
            the project root.

    Returns:
        The parsed JSON as a dict, with a top-level "data" key holding one
        entry per contract.
    """
    with open(path) as f:
        return json.load(f)


# --- Interval helpers -------------------------------------------------------
# Used to figure out which stretches of a contract's text are NOT claimed by
# any category's annotated answer spans, so those stretches can become
# "None"-class negative examples.

def get_claimed_intervals(contract: dict) -> list[tuple[int, int]]:
    """Collect every annotated answer span's character range for one contract.

    Walks all 41 categories' qas (not just a specific one), so the result
    reflects everything any category has claimed as clause text.

    Args:
        contract: One entry from cuad_json["data"], with a "paragraphs" list.

    Returns:
        A list of (start, end) character-offset tuples into that contract's
        context string, one per answer span, unsorted and possibly
        overlapping.
    """
    intervals = []
    for paragraph in contract["paragraphs"]:
        for qa in paragraph["qas"]:
            for answer in qa["answers"]:
                start = answer["answer_start"]
                end = start + len(answer["text"])
                intervals.append((start, end))
    return intervals


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching (start, end) intervals into a sorted,
    non-overlapping set.

    Assumes `intervals` is non-empty.

    Args:
        intervals: Unsorted list of (start, end) tuples, possibly overlapping.

    Returns:
        Sorted list of merged, non-overlapping (start, end) tuples.
    """
    sorted_intervals = sorted(intervals)
    merged = []
    cur_start, cur_end = sorted_intervals[0]

    for start, end in sorted_intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end

    merged.append((cur_start, cur_end))
    return merged


def find_gaps(merged_intervals: list[tuple[int, int]], text_length: int) -> list[tuple[int, int]]:
    """Find the uncovered stretches between (and around) a set of merged intervals.

    Args:
        merged_intervals: Sorted, non-overlapping (start, end) tuples, e.g.
            from merge_intervals().
        text_length: Total length of the text the intervals index into.

    Returns:
        List of (start, end) gap tuples covering everything not claimed by
        merged_intervals, bounded by [0, text_length].
    """
    gaps = []
    cursor = 0

    for start, end in merged_intervals:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = end

    if cursor < text_length:
        gaps.append((cursor, text_length))

    return gaps


# --- Chunking helpers --------------------------------------------------------
# Turn a raw gap of unclaimed text into clause-length chunks, discarding
# fragments too short to be meaningful and splitting stretches too long to be
# a plausible single clause, preferring natural boundaries (paragraph, then
# sentence) over arbitrary character cutoffs.

def split_long_piece(piece: str, min_len: int, max_len: int) -> list[str]:
    """Fallback splitter for a piece still over max_len after paragraph splitting.

    Splits on sentence boundaries (approximate — not a full sentence
    tokenizer, which is unnecessary precision for filler negative examples).
    Sentences still over max_len after that are hard-truncated as a last
    resort; sentences under min_len are discarded.

    Args:
        piece: A single (already paragraph-split) chunk of text, longer than
            max_len.
        min_len: Minimum character length for a chunk to be kept.
        max_len: Maximum character length for a chunk before truncation.

    Returns:
        List of sentence-level chunks, each within [min_len, max_len].
    """
    sentences = re.split(r"(?<=[.!?])\s+", piece)
    res = []
    for sentence in sentences:
        sentence_stripped = sentence.strip()
        if len(sentence_stripped) < min_len:
            continue
        elif len(sentence_stripped) <= max_len:
            res.append(sentence_stripped)
        else:
            res.append(sentence_stripped[:max_len])
    return res


def chunk_gap_text(gap_text: str, min_len: int, max_len: int) -> list[str]:
    """Turn one raw gap of unclaimed contract text into clause-length chunks.

    Discards gaps that are just connective junk (too short after stripping),
    keeps gaps already in a reasonable size range as-is, and splits
    oversized gaps on paragraph breaks (falling back to split_long_piece for
    any resulting piece still too long).

    Args:
        gap_text: Raw text of one gap, e.g. context[start:end] from find_gaps.
        min_len: Minimum character length for a chunk to be kept.
        max_len: Maximum character length for a chunk before it gets split
            (or, as a last resort inside split_long_piece, truncated).

    Returns:
        List of text chunks, each within [min_len, max_len].
    """
    stripped = gap_text.strip()

    if len(stripped) < min_len:
        return []

    elif len(stripped) <= max_len:
        return [stripped]

    else:
        chunks = []
        for piece in gap_text.split("\n\n"):
            piece_stripped = piece.strip()

            if len(piece_stripped) < min_len:
                continue

            elif len(piece_stripped) <= max_len:
                chunks.append(piece_stripped)

            else:
                chunks.extend(split_long_piece(piece_stripped, min_len, max_len))

        return chunks


# --- Example loaders ----------------------------------------------------------

def load_positive_examples(cuad_json: dict) -> list[dict]:
    """Build one training example per annotated answer span.

    Args:
        cuad_json: Parsed CUAD JSON, e.g. from load_cuad().

    Returns:
        List of {"contract_id", "category", "text"} dicts, one per answer
        span across all non-impossible qas in all contracts.
    """
    examples = []
    for contract in cuad_json["data"]:
        for paragraph in contract["paragraphs"]:
            for qa in paragraph["qas"]:
                if not qa["is_impossible"]:
                    match = re.search(r'"([^"]+)"', qa["question"])
                    category = match.group(1)
                    for answer in qa["answers"]:
                        examples.append({
                            "contract_id": contract["title"],
                            "category": category,
                            "text": answer["text"],
                        })
    return examples


def load_negative_examples(
    cuad_json: dict, min_len: int, max_len: int, seed: int = 42
) -> list[dict]:
    """Build "None"-class training examples from each contract's unclaimed text.

    Per contract, finds the gaps between annotated answer spans, chunks them
    into clause-length pieces, and randomly samples down to (at most) that
    contract's positive-example count, to keep the None class roughly
    balanced against the 41 real categories combined.

    Args:
        cuad_json: Parsed CUAD JSON, e.g. from load_cuad().
        min_len: Minimum character length for a candidate chunk to be kept.
        max_len: Maximum character length before a chunk is split/truncated.
        seed: Seed for the sampling RNG, for reproducible dataset builds.

    Returns:
        List of {"contract_id", "category": "None", "text"} dicts.
    """
    rng = random.Random(seed)
    examples = []

    for contract in cuad_json["data"]:
        context = contract["paragraphs"][0]["context"]

        intervals = get_claimed_intervals(contract)
        merged = merge_intervals(intervals)
        gaps = find_gaps(merged, len(context))

        all_chunks = []
        for start, end in gaps:
            gap_text = context[start:end]
            all_chunks.extend(chunk_gap_text(gap_text, min_len, max_len))

        cap = len(intervals)
        sample_size = min(cap, len(all_chunks))
        sample_chunks = rng.sample(all_chunks, sample_size)
        for chunk in sample_chunks:
            examples.append({
                "contract_id": contract["title"],
                "category": "None",
                "text": chunk,
            })

    return examples

data = load_cuad()
pos = load_positive_examples(data)
neg = load_negative_examples(data, min_len=20, max_len=600)
combined_list = pos + neg
df = pd.DataFrame(combined_list)

print(len(df))
print(df["contract_id"].nunique())
print(df["category"].value_counts())

gss1 = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, temp_idx = next(gss1.split(df, groups=df["contract_id"]))
train_df = df.iloc[train_idx]
temp_df = df.iloc[temp_idx]

gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["contract_id"]))
val_df = temp_df.iloc[val_idx]
test_df = temp_df.iloc[test_idx]

print(len(train_df), len(val_df), len(test_df))
print(train_df["contract_id"].nunique(), val_df["contract_id"].nunique(), test_df["contract_id"].nunique())

train_contracts = set(train_df["contract_id"])
val_contracts = set(val_df["contract_id"])
test_contracts = set(test_df["contract_id"])

print(train_contracts & val_contracts)
print(train_contracts & test_contracts)
print(val_contracts & test_contracts)

print(train_df[train_df["category"]=="Price Restrictions"].shape[0])
print(val_df[val_df["category"]=="Price Restrictions"].shape[0])
print(test_df[test_df["category"]=="Price Restrictions"].shape[0])

def cap_categories(df, max_per_category, seed=42):
    capped_groups = []
    for category, group in df.groupby("category"):
        if len(group) <= max_per_category:
            capped_groups.append(group)
        else:
            capped_groups.append(group.sample(n=max_per_category, random_state=seed))
    return pd.concat(capped_groups)

capped_train_df = cap_categories(train_df, 600, seed=42)
print(capped_train_df["category"].value_counts())

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)

messages = [
    {"role": "user", "content": "PLACEHOLDER_CLAUSE_TEXT"},
    {"role": "assistant", "content": "PLACEHOLDER_CATEGORY"},
]

full_format = tokenizer.apply_chat_template(messages, tokenize=False)
print(repr(full_format))

inference_format = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
print(repr(inference_format))

INSTRUCTION_TEMPLATE = """Classify the following contract clause into its category, or respond with "None" if it does not match any category. Respond with only the category name and nothing else.

Clause:
{clause_text}"""

combined_df = df.copy()
longest = combined_df.sort_values("text", key=lambda col: col.str.len(), ascending=False).head(5)

for _, row in longest.iterrows():
    full_text = INSTRUCTION_TEMPLATE.format(clause_text=row["text"])
    token_count = len(tokenizer.encode(full_text))
    print(len(row["text"]), token_count)