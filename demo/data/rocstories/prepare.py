"""
ROCStories Data Preparation — All 3 Variants

This single script prepares data for all 3 model variants:

  M1 (forward):    Standard story order S1 → S2 → S3 → S4 → S5  (default)
  M2 (reverse):    Reversed story order S5 → S4 → S3 → S2 → S1
  M3 (structured): Structured tokens    <S1> S1 <S2> S2 ... <S5> S5

Usage (run from demo/ directory):
  python data/rocstories/prepare.py --variant=forward     # M1
  python data/rocstories/prepare.py --variant=reverse     # M2
  python data/rocstories/prepare.py --variant=structured  # M3

Or run all 3 at once:
  python data/rocstories/prepare.py --variant=all

Output directories (relative to demo/):
  M1  →  data/rocstories/         (train.bin, val.bin, test.bin, meta.pkl)
  M2  →  data/rocstories_reverse/ (train.bin, val.bin, test.bin, meta.pkl)
  M3  →  data/rocstories_struct/  (train.bin, val.bin, test.bin, meta.pkl)

Requirements:
  pip install tiktoken pandas numpy

CSV file:
  Place rocstories_train.csv in demo/data/rocstories/
  Download from: https://huggingface.co/datasets/mintujupally/ROCStories
  Expected columns: storyid, storytitle, sentence1, sentence2, sentence3, sentence4, sentence5
"""

import os
import sys
import argparse
import pickle
import random

import numpy as np

# ---------------------------------------------------------------------------
# Helper: resolve paths relative to this script file
# script lives at demo/data/rocstories/prepare.py
# data_root is demo/data/
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # demo/data/rocstories/
DATA_ROOT  = os.path.dirname(SCRIPT_DIR)                   # demo/data/
CSV_PATH   = os.path.join(SCRIPT_DIR, 'rocstories_train.csv')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare ROCStories data for nanoGPT training.'
    )
    parser.add_argument(
        '--variant',
        type=str,
        default='forward',
        choices=['forward', 'reverse', 'structured', 'all'],
        help=(
            'Which variant to prepare: '
            '"forward" (M1), "reverse" (M2), "structured" (M3), or "all" (all three).'
        ),
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument(
        '--csv_path',
        type=str,
        default=CSV_PATH,
        help='Path to rocstories_train.csv (default: data/rocstories/rocstories_train.csv)',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Output directory mapping
# ---------------------------------------------------------------------------
VARIANT_TO_OUTDIR = {
    'forward':    os.path.join(DATA_ROOT, 'rocstories'),
    'reverse':    os.path.join(DATA_ROOT, 'rocstories_reverse'),
    'structured': os.path.join(DATA_ROOT, 'rocstories_struct'),
}


def assemble_story(row, variant):
    """Combine 5 sentences into a single string based on the variant."""
    s1 = str(row['sentence1']).strip()
    s2 = str(row['sentence2']).strip()
    s3 = str(row['sentence3']).strip()
    s4 = str(row['sentence4']).strip()
    s5 = str(row['sentence5']).strip()

    if variant == 'forward':
        # M1: standard narrative order
        return ' '.join([s1, s2, s3, s4, s5])
    elif variant == 'reverse':
        # M2: reversed order — tests causal asymmetry hypothesis
        return ' '.join([s5, s4, s3, s2, s1])
    elif variant == 'structured':
        # M3: explicit sentence position markers
        # tiktoken will encode <S1> as multi-token sequence: <, S, 1, >
        # This is fine — the model learns the pattern even without special token status.
        return f"<S1> {s1} <S2> {s2} <S3> {s3} <S4> {s4} <S5> {s5}"
    else:
        raise ValueError(f"Unknown variant: {variant}")


def prepare_variant(df, variant, seed):
    """Tokenize and save one variant of the ROCStories dataset."""
    import tiktoken

    out_dir = VARIANT_TO_OUTDIR[variant]
    os.makedirs(out_dir, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    sep_tokens = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})

    print(f"\n{'='*60}")
    print(f"Preparing variant: {variant.upper()}")
    print(f"Output dir: {out_dir}")
    print(f"{'='*60}")

    # -----------------------------------------------------------------------
    # Shuffle indices with fixed seed for reproducible 90 / 5 / 5 split
    # -----------------------------------------------------------------------
    n = len(df)
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    n_train = int(0.90 * n)
    n_val   = int(0.05 * n)
    # remainder goes to test

    split_indices = {
        'train': indices[:n_train],
        'val':   indices[n_train : n_train + n_val],
        'test':  indices[n_train + n_val:],
    }
    print(f"Split sizes → train: {len(split_indices['train'])}, "
          f"val: {len(split_indices['val'])}, test: {len(split_indices['test'])}")

    # -----------------------------------------------------------------------
    # Encode each split and save as uint16 binary
    # -----------------------------------------------------------------------
    split_token_counts = {}
    for split_name, idx_list in split_indices.items():
        tokens = []
        for i in idx_list:
            story_text = assemble_story(df.iloc[i], variant)
            story_tokens = enc.encode(story_text, allowed_special={"<|endoftext|>"})
            tokens.extend(story_tokens + sep_tokens)

        arr = np.array(tokens, dtype=np.uint16)
        out_path = os.path.join(out_dir, f'{split_name}.bin')
        arr.tofile(out_path)
        split_token_counts[split_name] = len(tokens)
        print(f"  [{split_name}] {len(tokens):>10,} tokens  →  {out_path}  ({arr.nbytes/1e6:.1f} MB)")

    # -----------------------------------------------------------------------
    # Save meta.pkl for compatibility with train.py / sample.py
    # (vocab_size is what matters; the tokenizer is always GPT-2 BPE)
    # -----------------------------------------------------------------------
    meta = {
        'vocab_size':   enc.n_vocab,   # 50257
        'variant':      variant,
        'n_stories':    n,
        'n_train':      len(split_indices['train']),
        'n_val':        len(split_indices['val']),
        'n_test':       len(split_indices['test']),
        'seed':         seed,
        'token_counts': split_token_counts,
    }
    meta_path = os.path.join(out_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"  [meta] Saved → {meta_path}")
    print(f"  Total train tokens: {split_token_counts['train']:,}  |  "
          f"vocab_size: {meta['vocab_size']}")

    # -----------------------------------------------------------------------
    # Quick sanity check: decode first story of train split
    # -----------------------------------------------------------------------
    sample_data = np.fromfile(os.path.join(out_dir, 'train.bin'), dtype=np.uint16)
    # Find first <|endoftext|> to isolate the first story
    eos_id = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    eos_positions = np.where(sample_data == eos_id)[0]
    if len(eos_positions) > 0:
        first_story_tokens = sample_data[:eos_positions[0]].tolist()
        decoded = enc.decode(first_story_tokens)
        print(f"\n  Sanity check — first story decoded:")
        print(f"  {decoded[:200]}{'...' if len(decoded) > 200 else ''}")

    print(f"\n  Done. Run training with:")
    cmd_map = {'forward': 'train_m1_forward', 'reverse': 'train_m2_reverse', 'structured': 'train_m3_structured'}
    print(f"  python train.py config/{cmd_map[variant]}.py")


def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # Load CSV
    # -----------------------------------------------------------------------
    if not os.path.exists(args.csv_path):
        print(f"\nERROR: CSV file not found at: {args.csv_path}")
        print("\nPlease download the ROCStories dataset:")
        print("  1. Go to https://huggingface.co/datasets/mintujupally/ROCStories")
        print("  2. Download rocstories_train.csv")
        print(f"  3. Place it at: {args.csv_path}")
        print("\nOr in the Colab notebook, run the download cell (Step 3).")
        sys.exit(1)

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not installed. Run: pip install pandas")
        sys.exit(1)

    print(f"Loading {args.csv_path} ...")
    df = pd.read_csv(args.csv_path)
    print(f"  Loaded {len(df):,} stories")
    print(f"  Columns: {list(df.columns)}")

    # Verify expected columns exist
    required = ['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        print(f"Found columns: {list(df.columns)}")
        sys.exit(1)

    # Drop rows with any missing sentence
    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with missing sentences. Using {len(df):,} stories.")

    # -----------------------------------------------------------------------
    # Run preparation for selected variant(s)
    # -----------------------------------------------------------------------
    variants = ['forward', 'reverse', 'structured'] if args.variant == 'all' else [args.variant]
    for v in variants:
        prepare_variant(df, v, args.seed)

    print(f"\n{'='*60}")
    print("All done! Next steps:")
    if 'forward' in variants:
        print("  python train.py config/train_m1_forward.py     # Train M1")
    if 'reverse' in variants:
        print("  python train.py config/train_m2_reverse.py     # Train M2")
    if 'structured' in variants:
        print("  python train.py config/train_m3_structured.py  # Train M3")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
