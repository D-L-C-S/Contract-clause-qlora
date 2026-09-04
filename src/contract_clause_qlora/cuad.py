"""Load and process the CUAD (Contract Understanding Atlas Dataset) SQuAD-format JSON
into (contract_id, category, text) example rows suitable for clause classification.

Positive examples come from CUAD's annotated answer spans (one row per span, per
category). Negative ("None") examples are constructed by finding the stretches of
each contract's text that no category claims, then chunking those stretches into
clause-length pieces on natural boundaries (paragraph breaks, then sentence breaks).

Once positive/negative examples are combined, split, and (for train) balanced,
format_example()/format_split() render them into Phi-3-mini-4k-instruct's chat
format for supervised fine-tuning, and write_jsonl() serializes a split to disk.

See scripts/build_dataset.py for the orchestration that runs this end to end.
"""

import json
import random
import re
from pathlib import Path

import pandas as pd

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


# --- Category balancing ------------------------------------------------------

def cap_categories(df: pd.DataFrame, max_per_category: int, seed: int = 42) -> pd.DataFrame:
    """Subsample dominant categories down to a per-category ceiling.

    Categories already at or below max_per_category are left untouched;
    larger ones are randomly downsampled to exactly max_per_category rows.
    Intended for the train split only — val/test should keep their natural,
    uncapped distribution so evaluation reflects real-world class frequencies.

    Args:
        df: DataFrame with a "category" column.
        max_per_category: Ceiling on rows per category after capping.
        seed: Seed for the sampling RNG, for reproducible results.

    Returns:
        A new DataFrame with dominant categories subsampled down to the cap.
    """
    capped_groups = []
    for _category, group in df.groupby("category"):
        if len(group) <= max_per_category:
            capped_groups.append(group)
        else:
            capped_groups.append(group.sample(n=max_per_category, random_state=seed))
    return pd.concat(capped_groups)


# --- Prompt formatting + serialization ---------------------------------------

# Fixed instruction wording for the <|user|> turn. This must stay identical
# across every training example (and match inference-time prompts exactly),
# since fine-tuning teaches the model this exact input pattern rather than
# the model being prompted zero-shot. Deliberately does not enumerate the 41
# category names — supervised fine-tuning learns the label space from the
# training data itself.
INSTRUCTION_TEMPLATE = """Classify the following contract clause into its category, or respond with "None" if it does not match any category. Respond with only the category name and nothing else.

Clause:
{clause_text}"""


def format_example(example: dict, tokenizer) -> dict:
    """Format one (contract_id, category, text) row into a Phi-3 training example.

    Wraps the clause text in INSTRUCTION_TEMPLATE as the user turn and the
    category (verbatim, including "None") as the assistant turn, then
    renders both through the tokenizer's chat template to produce the exact
    string the model will be trained on.

    Also produces separate "prompt" and "completion" strings (rather than
    only the merged "text"), so training can use completion-only loss
    masking — computing loss only on the assistant's response tokens,
    not the (long, identical-every-time) instruction tokens. "prompt" is
    the user-turn rendering with add_generation_prompt=True (i.e. what an
    inference-time prompt looks like); "completion" is everything the full
    two-turn rendering adds after that point (the category, plus the
    <|end|>/<|endoftext|> tokens the model needs to learn to stop on).

    Args:
        example: A {"contract_id", "category", "text"} dict, e.g. one row
            from load_positive_examples()/load_negative_examples().
        tokenizer: A loaded tokenizer exposing apply_chat_template(), e.g.
            AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct").

    Returns:
        {"contract_id", "category", "clause_text" (original, unformatted
        clause text, kept for debugging), "prompt", "completion", "text"
        (prompt + completion, kept for convenience/debugging)}.
    """
    messages = [
        {"role": "user", "content": INSTRUCTION_TEMPLATE.format(clause_text=example["text"])},
        {"role": "assistant", "content": example["category"]},
    ]

    prompt = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    completion = full_text[len(prompt):]

    return {
        "contract_id": example["contract_id"],
        "category": example["category"],
        "clause_text": example["text"],
        "prompt": prompt,
        "completion": completion,
        "text": full_text,
    }


def format_split(df: pd.DataFrame, tokenizer) -> list[dict]:
    """Apply format_example() to every row of a split's DataFrame.

    Args:
        df: DataFrame with "contract_id", "category", "text" columns, e.g.
            one of train/val/test after splitting (and, for train, capping).
        tokenizer: A loaded tokenizer, as in format_example().

    Returns:
        List of formatted example dicts; see format_example().
    """
    records = df.to_dict(orient="records")
    return [format_example(record, tokenizer) for record in records]


def write_jsonl(examples: list[dict], path: Path) -> None:
    """Write a list of dicts to a JSON Lines file, one JSON object per line.

    Uses ensure_ascii=False so contract text with non-ASCII characters
    (curly quotes, em-dashes, redaction markers, etc.) stays human-readable
    in the output file rather than being escaped to \\uXXXX sequences.

    Args:
        examples: List of JSON-serializable dicts.
        path: Output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")