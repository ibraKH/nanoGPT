"""
Local Evaluation Script — All 3 Models (demo2 — FIXED)
=======================================================
Loads the 3 trained checkpoints from the downloaded Drive folder,
computes perplexity on eval_stories.txt, generates story samples,
and saves everything to demo2/results/.

DEMO1 BUG FIXED:
  M3 was evaluated on plain text → PPL 93.86 instead of ~11.
  Fix: format each story to match the training variant before computing PPL.
    M1/M2: plain text (matches tutor's evaluation method)
    M3:    <S1>...<S5> structured format (matches its training data)

Run from the demo2/ directory:
    python evaluate_local.py

Requirements:
    pip install torch tiktoken matplotlib numpy nltk
"""

import os
import sys
import math
import json
import pickle
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import tiktoken
import matplotlib
matplotlib.use('Agg')  # no display needed
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
DEMO_DIR        = Path(__file__).parent.resolve()
# Checkpoints downloaded from Google Drive are expected here:
CKPT_BASE       = DEMO_DIR / 'nanoGPT_rocstories 3'
EVAL_STORIES    = DEMO_DIR / 'data' / 'rocstories' / 'eval_stories.txt'
EVAL_PROMPTS    = DEMO_DIR / 'data' / 'rocstories' / 'eval_prompts.txt'
RESULTS_DIR     = DEMO_DIR / 'results'
SAMPLES_DIR     = RESULTS_DIR / 'samples'

# Model registry — (display name, out_dir_name, format description, variant)
MODELS = [
    ('M1_BEST Forward',  'out-m1-best',       'S1→S2→S3→S4→S5 (plain text)', 'forward'),
    ('M2 Reverse',       'out-m2-reverse',    'S5→S4→S3→S2→S1 (plain text)', 'reverse'),
    ('M3 Structured',    'out-m3-structured', '<Si> tokens (structured)',      'structured'),
]

# ── Sampling params ────────────────────────────────────────────────────────
NUM_SAMPLES     = 5
MAX_NEW_TOKENS  = 150
TEMPERATURE     = 0.8
TOP_K           = 100

# ── Setup ──────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Add demo2/ to sys.path so model.py is importable
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from model import GPT, GPTConfig


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_model(out_dir_name):
    """Load a checkpoint and return (model, device, ctx, encode, decode, info)."""
    device = get_device()
    ckpt_path = CKPT_BASE / out_dir_name / 'ckpt.pt'

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"  Loading {ckpt_path} ...")
    checkpoint = torch.load(str(ckpt_path), map_location=device)

    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted):
            state_dict[k[len(unwanted):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    # Dtype / autocast context
    if device == 'cuda':
        dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
        ptdtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float16
        ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    else:
        ctx = nullcontext()

    # Tokenizer — GPT-2 BPE for all ROCStories models
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda ids: enc.decode(ids)

    info = {
        'iter_num':      checkpoint.get('iter_num', 0),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'n_params':      model.get_num_params(),
    }
    return model, device, ctx, encode, decode, info


# ── Story formatting ───────────────────────────────────────────────────────

def split_5_sentences(paragraph):
    """Split a paragraph into exactly 5 sentences using NLTK."""
    try:
        import nltk
        nltk.download('punkt',     quiet=True)
        nltk.download('punkt_tab', quiet=True)
        from nltk.tokenize import sent_tokenize
        sents = [s.strip() for s in sent_tokenize(paragraph) if s.strip()]
    except Exception:
        sents = [s.strip() + '.' for s in paragraph.split('. ') if s.strip()]

    while len(sents) < 5:
        sents.append(sents[-1] if sents else '')
    return sents[:5]


def format_story(paragraph, variant):
    """
    Format a plain-text story to match each model's training format.

    KEY FIX (demo1 bug #5):
      M1/M2: tutor feeds plain text → evaluate in plain/reversed format.
      M3:    trained on structured text → must format with <S1>...<S5> markers
             so PPL reflects its actual capability (~11), not ~94 on plain text.

    Args:
        paragraph: raw plain-text story string
        variant:   'forward' | 'reverse' | 'structured'
    """
    if variant == 'forward':
        # M1_BEST: plain text — no change needed
        return paragraph
    sents = split_5_sentences(paragraph)
    if variant == 'reverse':
        # M2: reverse the sentence order
        return ' '.join(reversed(sents))
    if variant == 'structured':
        # M3: add <S1>...<S5> position markers
        return ' '.join(f'<S{i+1}> {s}' for i, s in enumerate(sents))
    return paragraph


# ── Perplexity computation ─────────────────────────────────────────────────

def compute_ppl(model, device, ctx, encode, variant='forward'):
    """
    Compute average perplexity over eval_stories.txt paragraphs.
    Stories are formatted to match the training format of each model variant.
    """
    if not EVAL_STORIES.exists():
        print(f"  WARNING: {EVAL_STORIES} not found, skipping PPL.")
        return None

    text = EVAL_STORIES.read_text(encoding='utf-8')
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    block_size = model.config.block_size
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        with ctx:
            for para in paragraphs:
                formatted = format_story(para, variant)
                ids = encode(formatted)
                if len(ids) < 2:
                    continue
                for pos in range(0, len(ids) - 1, block_size):
                    inp = ids[pos: pos + block_size]
                    tgt = ids[pos + 1: pos + 1 + block_size]
                    if not tgt:
                        break
                    if len(inp) != len(tgt):
                        inp = inp[:len(tgt)]
                    x = torch.tensor(inp, dtype=torch.long, device=device)[None, :]
                    y = torch.tensor(tgt, dtype=torch.long, device=device)[None, :]
                    _, loss = model(x, y)
                    total_nll += loss.item() * len(tgt)
                    total_tokens += len(tgt)

    if total_tokens == 0:
        return None
    return math.exp(total_nll / total_tokens)


# ── Sample generation ──────────────────────────────────────────────────────

def generate_samples(model, device, ctx, encode, decode,
                     start='\n', n=NUM_SAMPLES,
                     max_new_tokens=MAX_NEW_TOKENS,
                     temperature=TEMPERATURE, top_k=TOP_K):
    """Return a list of generated text strings."""
    start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, :]
    samples = []

    with torch.no_grad():
        with ctx:
            for _ in range(n):
                y = model.generate(x, max_new_tokens,
                                   temperature=temperature, top_k=top_k)
                samples.append(decode(y[0].tolist()))
    return samples


# ── Plots and reports ──────────────────────────────────────────────────────

def save_ppl_plot(results):
    names = [r['name'] for r in results if r['ppl'] is not None]
    ppls  = [r['ppl']  for r in results if r['ppl'] is not None]
    if not names:
        print("  No PPL values to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#2196F3', '#F44336', '#4CAF50']
    bars = ax.bar(names, ppls, color=colors[:len(names)],
                  edgecolor='black', linewidth=0.7, width=0.5)

    # PPL ≤ 25.0 threshold line (tutor passing line for M1/M2)
    ax.axhline(25.0, color='orange', linestyle='--', linewidth=1.5,
               label='PPL = 25.0 (tutor threshold)')

    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f'{ppl:.1f}', ha='center', va='bottom',
                fontweight='bold', fontsize=12)

    ax.set_ylabel('Perplexity on eval_stories.txt  (lower = better)', fontsize=11)
    ax.set_title('M1_BEST vs M2 vs M3 — Evaluation Perplexity', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(ppls) * 1.3)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    out = RESULTS_DIR / 'ppl_comparison.png'
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot → {out}")


def save_report(results):
    lines = [
        '# Demo2 Evaluation Results\n',
        '## Perplexity Table\n',
        '> Note: M1/M2 evaluated on plain text (matches tutor eval).',
        '> M3 evaluated on structured <S1>...<S5> format (matches its training).\n',
        '| Model | Format | Eval PPL | Best Val PPL | Params | Final Iter |',
        '|-------|--------|----------|--------------|--------|-----------|',
    ]
    for r in results:
        ppl_str = f"{r['ppl']:.2f}"      if r['ppl']     is not None else 'N/A'
        vl_str  = f"{r['val_ppl']:.2f}"  if r['val_ppl'] is not None else 'N/A'
        par_str = f"{r['n_params']/1e6:.1f}M"
        lines.append(
            f"| {r['name']} | {r['format']} | {ppl_str} | {vl_str} "
            f"| {par_str} | {r['iter_num']} |"
        )

    lines += [
        '',
        '## Hypotheses',
        '- **H1:** M2 (Reverse) PPL > M1_BEST (Forward) PPL → causal asymmetry in narrative',
        '- **H2:** M3 (Structured) in-format PPL ≤ M1_BEST (Forward) PPL → structure helps',
        '',
        '## Submission',
        '- Submit `out-m1-best/ckpt.pt` (plain-text model) for Task 3.',
        '- `sampling_params.json`: temperature=0.8, top_k=100, start="\\n"',
        '',
        '## PPL Interpretation',
        '- < 25.0 = passes tutor threshold (M1/M2 target)',
        '- < 50   = excellent',
        '- 50-100 = good',
        '- > 200  = undertrained',
        '',
        '---',
        '_Generated by evaluate_local.py (demo2)_',
    ]

    out = RESULTS_DIR / 'perplexity_table.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  Saved report → {out}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print('=' * 65)
    print('  NanoGPT ROCStories — Local Evaluation (demo2)')
    print(f'  Checkpoints : {CKPT_BASE}')
    print(f'  Results     : {RESULTS_DIR}')
    print('=' * 65)

    results = []

    for name, out_dir, fmt, variant in MODELS:
        print(f'\n[{name}]')

        ckpt_path = CKPT_BASE / out_dir / 'ckpt.pt'
        if not ckpt_path.exists():
            print(f'  SKIP — checkpoint not found: {ckpt_path}')
            results.append({'name': name, 'format': fmt, 'variant': variant,
                            'ppl': None, 'val_ppl': None, 'n_params': 0, 'iter_num': 0})
            continue

        model, device, ctx, encode, decode, info = load_model(out_dir)
        print(f'  Device      : {device}')
        print(f'  Params      : {info["n_params"]/1e6:.2f}M')
        print(f'  Final iter  : {info["iter_num"]}')
        best_val_ppl = math.exp(info['best_val_loss'])
        print(f'  Best val PPL: {best_val_ppl:.2f}')

        # Perplexity — format stories to match each model's training format
        print(f'  Computing perplexity (variant={variant}) ...')
        ppl = compute_ppl(model, device, ctx, encode, variant=variant)
        if ppl is not None:
            print(f'  Eval PPL    : {ppl:.2f}')
            if variant in ('forward', 'reverse'):
                status = 'PASSES' if ppl <= 25.0 else 'FAILS'
                print(f'  Tutor PPL ≤ 25.0: {status}')
        else:
            print('  Eval PPL    : N/A')

        # Start token for generation
        start = '<S1>' if variant == 'structured' else '\n'

        # Free generation samples
        print(f'  Generating {NUM_SAMPLES} samples (start={repr(start)}) ...')
        samples = generate_samples(model, device, ctx, encode, decode, start=start)
        safe_name = out_dir.replace('-', '_')
        sample_file = SAMPLES_DIR / f'{safe_name}_samples.txt'
        with open(sample_file, 'w', encoding='utf-8') as f:
            for i, s in enumerate(samples, 1):
                f.write(f'=== Sample {i} ===\n{s}\n\n')
        print(f'  Samples saved → {sample_file}')

        # M3 only: guided completions from eval_prompts.txt
        if variant == 'structured' and EVAL_PROMPTS.exists():
            print('  Generating guided completions (M3 only) ...')
            prompts = [p.strip() for p in
                       EVAL_PROMPTS.read_text(encoding='utf-8').splitlines()
                       if p.strip()]
            comp_file = SAMPLES_DIR / 'm3_structured_completions.txt'
            with open(comp_file, 'w', encoding='utf-8') as f:
                for i, prompt in enumerate(prompts, 1):
                    guided_start = f'<S1> {prompt} <S2>'
                    completions = generate_samples(
                        model, device, ctx, encode, decode,
                        start=guided_start, n=2, max_new_tokens=120)
                    f.write(f'=== Prompt {i} ===\n')
                    f.write(f'Given: {guided_start}\n\n')
                    for j, c in enumerate(completions, 1):
                        f.write(f'-- Completion {j} --\n{c}\n\n')
            print(f'  Completions saved → {comp_file}')

        results.append({
            'name':     name,
            'format':   fmt,
            'variant':  variant,
            'ppl':      ppl,
            'val_ppl':  best_val_ppl,
            'n_params': info['n_params'],
            'iter_num': info['iter_num'],
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary ─────────────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('RESULTS SUMMARY')
    print('=' * 65)
    print(f"{'Model':<22} {'Variant':<12} {'Eval PPL':>10} {'Val PPL':>10}")
    print('-' * 58)
    for r in results:
        ppl_str = f"{r['ppl']:.2f}"     if r['ppl']     is not None else 'N/A'
        vl_str  = f"{r['val_ppl']:.2f}" if r['val_ppl'] is not None else 'N/A'
        print(f"{r['name']:<22} {r['variant']:<12} {ppl_str:>10} {vl_str:>10}")

    trained = [r for r in results if r['ppl'] is not None]
    if len(trained) >= 2:
        m1 = next((r for r in trained if 'BEST' in r['name'] or 'Forward' in r['name']), None)
        m2 = next((r for r in trained if 'Reverse' in r['name']), None)
        m3 = next((r for r in trained if 'Struct' in r['name']), None)
        print()
        if m1 and m2:
            h1 = 'CONFIRMED' if m2['ppl'] > m1['ppl'] else 'NOT confirmed'
            print(f"H1 (M2 > M1): {h1}  (M2={m2['ppl']:.1f} vs M1={m1['ppl']:.1f})")
        if m1 and m3:
            h2 = 'CONFIRMED' if m3['ppl'] <= m1['ppl'] else 'NOT confirmed'
            print(f"H2 (M3 ≤ M1): {h2}  (M3={m3['ppl']:.1f} vs M1={m1['ppl']:.1f})")

    print()
    save_ppl_plot(results)
    save_report(results)
    print('\nAll done! Results saved to:', RESULTS_DIR)


if __name__ == '__main__':
    main()
