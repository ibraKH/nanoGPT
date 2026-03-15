"""
Prepare ROCStories dataset for nanoGPT training.

Supports --variant=forward|reverse|structured|llama|all
Handles both 'text' column (HuggingFace mintujupally/ROCStories) and
pre-parsed CSV with sentence1-sentence5 columns.

Tokenizer: GPT-2 BPE via tiktoken.
Split: 90/5/5 with seed=42.
meta.pkl: vocab_size only (no stoi/itos for BPE models).
"""

import os
import sys
import argparse
import pickle
import numpy as np

import tiktoken
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare ROCStories for nanoGPT')
    parser.add_argument('--variant', type=str, default='all',
                        choices=['forward', 'reverse', 'structured', 'llama', 'all'],
                        help='Which variant(s) to prepare')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split')
    parser.add_argument('--csv_path', type=str, default='',
                        help='Path to CSV file. If empty, downloads from HuggingFace.')
    return parser.parse_args()


def load_stories(csv_path=''):
    """Load stories as a DataFrame with sentence1..sentence5 columns."""
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Try loading from HuggingFace
        print("Loading ROCStories from HuggingFace (mintujupally/ROCStories)...")
        try:
            from datasets import load_dataset
            ds = load_dataset('mintujupally/ROCStories', split='train')
            df = ds.to_pandas()
        except Exception as e:
            print(f"Error loading from HuggingFace: {e}")
            # Fallback: look for local CSV
            local_paths = ['rocstories.csv', 'ROCStories.csv', 'rocstories_parsed.csv']
            for p in local_paths:
                full_p = os.path.join(os.path.dirname(__file__), p)
                if os.path.exists(full_p):
                    print(f"Found local CSV: {full_p}")
                    df = pd.read_csv(full_p)
                    break
            else:
                raise FileNotFoundError("Could not load ROCStories dataset. "
                                        "Provide --csv_path or install datasets library.")

    # Check if we have pre-parsed sentence columns
    if 'sentence1' in df.columns:
        print("Found pre-parsed sentence columns (sentence1..sentence5)")
        # Ensure all 5 sentence columns exist
        for i in range(1, 6):
            col = 'sentence' + str(i)
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        return df

    # Otherwise, parse from 'text' column using NLTK
    if 'text' not in df.columns:
        # Check for other common column names
        for candidate in ['story', 'content', 'narrative']:
            if candidate in df.columns:
                df['text'] = df[candidate]
                break
        else:
            raise ValueError(f"CSV has neither 'sentence1' nor 'text' column. "
                             f"Available columns: {list(df.columns)}")

    print("Parsing 'text' column into 5 sentences using NLTK sent_tokenize...")
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import sent_tokenize

    rows = []
    for idx, row in df.iterrows():
        text = str(row['text']).strip()
        if not text:
            continue
        sents = sent_tokenize(text)
        # Pad or truncate to exactly 5 sentences
        if len(sents) < 5:
            # Pad with empty strings (will be filtered or handled)
            sents = sents + [''] * (5 - len(sents))
        elif len(sents) > 5:
            sents = sents[:5]
        rows.append({
            'sentence1': sents[0],
            'sentence2': sents[1],
            'sentence3': sents[2],
            'sentence4': sents[3],
            'sentence5': sents[4],
        })

    parsed_df = pd.DataFrame(rows)
    # Save parsed CSV for reuse
    parsed_path = os.path.join(os.path.dirname(__file__), 'rocstories_parsed.csv')
    parsed_df.to_csv(parsed_path, index=False)
    print(f"Saved parsed CSV to {parsed_path} ({len(parsed_df)} stories)")
    return parsed_df


def format_forward(row):
    """M1/M4: plain text, sentences concatenated."""
    sents = []
    for i in range(1, 6):
        col = 'sentence' + str(i)
        s = str(row[col]).strip()
        if s:
            sents.append(s)
    return ' '.join(sents)


def format_reverse(row):
    """M2: sentences in reverse order."""
    sents = []
    for i in range(5, 0, -1):
        col = 'sentence' + str(i)
        s = str(row[col]).strip()
        if s:
            sents.append(s)
    return ' '.join(sents)


def format_structured(row):
    """M3: <S1> S1 <S2> S2 ... <S5> S5"""
    parts = []
    for i in range(1, 6):
        col = 'sentence' + str(i)
        s = str(row[col]).strip()
        if s:
            parts.append(f'<S{i}> {s}')
    return ' '.join(parts)


def encode_and_save(stories, out_dir, seed, enc):
    """Tokenize stories, split 90/5/5, and save train.bin, val.bin, test.bin, meta.pkl."""
    os.makedirs(out_dir, exist_ok=True)

    # Tokenize all stories with separator
    eot = enc.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})
    all_tokens = []
    for story in stories:
        tokens = enc.encode(story, allowed_special={'<|endoftext|>'})
        all_tokens.extend(tokens)
        all_tokens.extend(eot)

    all_tokens = np.array(all_tokens, dtype=np.uint16)
    print(f"  Total tokens: {len(all_tokens):,}")

    # Split by stories first, then tokenize
    n = len(stories)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)

    n_train = int(0.9 * n)
    n_val = int(0.05 * n)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    def tokenize_split(indices):
        tokens = []
        for i in indices:
            t = enc.encode(stories[i], allowed_special={'<|endoftext|>'})
            tokens.extend(t)
            tokens.extend(eot)
        return np.array(tokens, dtype=np.uint16)

    train_tokens = tokenize_split(train_idx)
    val_tokens = tokenize_split(val_idx)
    test_tokens = tokenize_split(test_idx)

    print(f"  Train: {len(train_tokens):,} tokens ({len(train_idx)} stories)")
    print(f"  Val:   {len(val_tokens):,} tokens ({len(val_idx)} stories)")
    print(f"  Test:  {len(test_tokens):,} tokens ({len(test_idx)} stories)")

    train_tokens.tofile(os.path.join(out_dir, 'train.bin'))
    val_tokens.tofile(os.path.join(out_dir, 'val.bin'))
    test_tokens.tofile(os.path.join(out_dir, 'test.bin'))

    # meta.pkl: vocab_size only (no stoi/itos for GPT-2 BPE)
    meta = {'vocab_size': 50304}  # GPT-2 vocab padded to multiple of 64
    with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"  Saved to {out_dir}/")


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load stories
    df = load_stories(args.csv_path)
    print(f"Loaded {len(df)} stories")

    # Init tokenizer
    enc = tiktoken.get_encoding("gpt2")

    variants = [args.variant] if args.variant != 'all' else ['forward', 'reverse', 'structured']

    for variant in variants:
        print(f"\n--- Preparing {variant} variant ---")

        if variant == 'forward' or variant == 'llama':
            stories = [format_forward(row) for _, row in df.iterrows()]
            out_dir = os.path.join(script_dir, '')  # data/rocstories/
        elif variant == 'reverse':
            stories = [format_reverse(row) for _, row in df.iterrows()]
            out_dir = os.path.join(script_dir, '..', 'rocstories_reverse')
        elif variant == 'structured':
            stories = [format_structured(row) for _, row in df.iterrows()]
            out_dir = os.path.join(script_dir, '..', 'rocstories_struct')
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Filter out empty stories
        stories = [s for s in stories if s.strip()]
        print(f"  {len(stories)} non-empty stories")

        encode_and_save(stories, out_dir, args.seed, enc)

    print("\nDone!")


if __name__ == '__main__':
    main()
