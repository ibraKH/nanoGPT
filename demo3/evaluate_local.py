"""
Local evaluation script for all 4 model variants.
Computes PPL, generates samples, runs sampling parameter sweep for M1.

Usage:
    python evaluate_local.py

Expects checkpoints in:
    out-m1/ckpt.pt, out-m2-reverse/ckpt.pt, out-m3-structured/ckpt.pt, out-m4-llama/ckpt.pt
"""

import os
import sys
import math
import json
import pickle
from contextlib import nullcontext

import torch
import tiktoken
import numpy as np

# Import both model classes
from model import GPTConfig, GPT
from model_llama import GPTConfig as GPTConfigLlama, GPTLlama

# -----------------------------------------------------------------------------
# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
seed = 1337
eval_stories_path = 'data/rocstories/eval_stories.txt'
eval_prompts_path = 'data/rocstories/eval_prompts.txt'
results_dir = 'results'
samples_dir = 'results/samples'
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

os.makedirs(results_dir, exist_ok=True)
os.makedirs(samples_dir, exist_ok=True)

# Tokenizer
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)


# --- Model loading ---
def load_model(ckpt_dir, model_type='gpt'):
    """Load a checkpoint. model_type='gpt' for M1/M2/M3, 'llama' for M4."""
    ckpt_path = os.path.join(ckpt_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        print(f"  WARNING: checkpoint not found at {ckpt_path}")
        return None

    checkpoint = torch.load(ckpt_path, map_location=device)
    if model_type == 'llama':
        gptconf = GPTConfigLlama(**checkpoint['model_args'])
        model = GPTLlama(gptconf)
    else:
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


# --- PPL computation ---
def compute_ppl(model, texts):
    """Compute perplexity over a list of text strings."""
    total_nll = 0.0
    total_tokens = 0
    block_size = model.config.block_size

    with torch.no_grad():
        with ctx:
            for text in texts:
                token_ids = encode(text)
                if len(token_ids) < 2:
                    continue

                pos = 0
                para_pred_tokens = len(token_ids) - 1
                while pos < para_pred_tokens:
                    inp = token_ids[pos: pos + block_size]
                    tgt = token_ids[pos + 1: pos + 1 + block_size]
                    if len(tgt) == 0:
                        break
                    if len(inp) != len(tgt):
                        inp = inp[:len(tgt)]

                    x = torch.tensor(inp, dtype=torch.long, device=device)[None, :]
                    y = torch.tensor(tgt, dtype=torch.long, device=device)[None, :]
                    _, loss = model(x, y)

                    n_tok = len(tgt)
                    total_nll += loss.item() * n_tok
                    total_tokens += n_tok
                    pos += n_tok

    if total_tokens == 0:
        return float('inf')
    avg_loss = total_nll / total_tokens
    return math.exp(avg_loss)


# --- Text formatting per variant ---
def load_eval_stories():
    """Load eval stories from file."""
    with open(eval_stories_path, 'r', encoding='utf-8') as f:
        content = f.read()
    stories = [p.strip() for p in content.split('\n\n') if p.strip()]
    return stories


def format_stories_for_variant(stories, variant):
    """Format stories according to the variant's training format."""
    if variant == 'forward' or variant == 'llama':
        return stories  # plain text, no change

    if variant == 'reverse':
        reversed_stories = []
        for story in stories:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                nltk.download('punkt_tab', quiet=True)
            from nltk.tokenize import sent_tokenize
            sents = sent_tokenize(story)
            # Pad/truncate to 5
            if len(sents) < 5:
                sents = sents + [''] * (5 - len(sents))
            elif len(sents) > 5:
                sents = sents[:5]
            sents = [s for s in reversed(sents) if s.strip()]
            reversed_stories.append(' '.join(sents))
        return reversed_stories

    if variant == 'structured':
        structured_stories = []
        for story in stories:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                nltk.download('punkt_tab', quiet=True)
            from nltk.tokenize import sent_tokenize
            sents = sent_tokenize(story)
            if len(sents) < 5:
                sents = sents + [''] * (5 - len(sents))
            elif len(sents) > 5:
                sents = sents[:5]
            parts = []
            for i, s in enumerate(sents, 1):
                if s.strip():
                    parts.append(f'<S{i}> {s.strip()}')
            structured_stories.append(' '.join(parts))
        return structured_stories

    return stories


# --- Sample generation ---
def generate_samples(model, start_text, num_samples=5, max_new_tokens=200,
                     temperature=0.8, top_k=100):
    """Generate text samples from a model."""
    start_ids = encode(start_text)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    samples = []
    with torch.no_grad():
        with ctx:
            for _ in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                text = decode(y[0].tolist())
                samples.append(text)
    return samples


def save_samples(samples, filepath):
    """Save samples to file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, s in enumerate(samples):
            f.write(f"--- Sample {i+1} ---\n")
            f.write(s + '\n\n')
    print(f"  Saved {len(samples)} samples to {filepath}")


# --- Main evaluation ---
def main():
    print("=" * 60)
    print("  ROCStories nanoGPT — Local Evaluation (Demo 3)")
    print("=" * 60)

    # Load eval stories
    stories = load_eval_stories()
    print(f"\nLoaded {len(stories)} eval stories from {eval_stories_path}")

    # Load eval prompts
    prompts = []
    if os.path.exists(eval_prompts_path):
        with open(eval_prompts_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} eval prompts from {eval_prompts_path}")

    # Define models
    models_config = [
        ('M1-Forward',    'out-m1',            'forward',    'gpt'),
        ('M2-Reverse',    'out-m2-reverse',    'reverse',    'gpt'),
        ('M3-Structured', 'out-m3-structured', 'structured', 'gpt'),
        ('M4-LLaMA',      'out-m4-llama',      'llama',      'llama'),
    ]

    ppl_results = {}

    # --- Step 1: Compute PPL for all models ---
    print("\n" + "=" * 60)
    print("  PERPLEXITY EVALUATION")
    print("=" * 60)

    for name, ckpt_dir, variant, model_type in models_config:
        print(f"\n--- {name} ({ckpt_dir}) ---")
        model = load_model(ckpt_dir, model_type)
        if model is None:
            ppl_results[name] = float('inf')
            continue

        formatted_stories = format_stories_for_variant(stories, variant)
        ppl = compute_ppl(model, formatted_stories)
        ppl_results[name] = ppl
        print(f"  Eval PPL: {ppl:.2f}")

        if name == 'M1-Forward':
            if ppl <= 25.0:
                print(f"  >>> PASS (PPL {ppl:.2f} <= 25.0)")
            else:
                print(f"  >>> FAIL (PPL {ppl:.2f} > 25.0)")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Step 2: Generate samples ---
    print("\n" + "=" * 60)
    print("  SAMPLE GENERATION")
    print("=" * 60)

    for name, ckpt_dir, variant, model_type in models_config:
        print(f"\n--- {name} ---")
        model = load_model(ckpt_dir, model_type)
        if model is None:
            continue

        # Standard samples
        if variant == 'structured':
            start_text = '<S1>'
        else:
            start_text = '\n'

        samples = generate_samples(model, start_text, num_samples=5,
                                   max_new_tokens=200, temperature=0.8, top_k=100)
        safe_name = name.lower().replace('-', '_')
        save_samples(samples, os.path.join(samples_dir, f'{safe_name}_samples.txt'))

        # M3 guided completions from prompts
        if variant == 'structured' and prompts:
            print(f"\n  Guided completions for {name}:")
            guided = []
            for prompt in prompts:
                guided_start = f'<S1> {prompt}'
                g_samples = generate_samples(model, guided_start, num_samples=1,
                                             max_new_tokens=200, temperature=0.8, top_k=100)
                guided.extend(g_samples)
            save_samples(guided, os.path.join(samples_dir, f'{safe_name}_guided.txt'))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Step 3: Sampling parameter sweep for M1 ---
    print("\n" + "=" * 60)
    print("  SAMPLING PARAMETER SWEEP (M1)")
    print("=" * 60)

    model = load_model('out-m1', 'gpt')
    if model is not None and prompts:
        temperatures = [0.7, 0.8, 0.9]
        top_ks = [50, 100, 200]

        print(f"\n  Grid: temperature x top_k")
        print(f"  Temperatures: {temperatures}")
        print(f"  Top-k values: {top_ks}")
        print()

        sweep_results = []
        for temp in temperatures:
            for tk in top_ks:
                print(f"  --- temp={temp}, top_k={tk} ---")
                sweep_samples = []
                for prompt in prompts[:3]:
                    s = generate_samples(model, prompt, num_samples=1,
                                         max_new_tokens=200, temperature=temp, top_k=tk)
                    sweep_samples.extend(s)

                # Print first sample as preview
                preview = sweep_samples[0][:200].replace('\n', ' ')
                print(f"    Preview: {preview}...")
                sweep_results.append({
                    'temperature': temp,
                    'top_k': tk,
                    'samples': sweep_samples
                })

                save_samples(sweep_samples,
                             os.path.join(samples_dir, f'm1_sweep_t{temp}_k{tk}.txt'))

        # Print sweep summary table
        print("\n  SWEEP SUMMARY:")
        print(f"  {'Temp':>6} {'Top-k':>6}  {'First sample preview':>50}")
        print("  " + "-" * 65)
        for r in sweep_results:
            preview = r['samples'][0][:50].replace('\n', ' ')
            print(f"  {r['temperature']:>6.1f} {r['top_k']:>6}  {preview}...")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Step 4: Results summary ---
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    # Perplexity table
    print(f"\n  {'Model':<20} {'PPL':>10} {'Status':>10}")
    print("  " + "-" * 42)
    for name in ['M1-Forward', 'M2-Reverse', 'M3-Structured', 'M4-LLaMA']:
        ppl = ppl_results.get(name, float('inf'))
        status = ''
        if name == 'M1-Forward':
            status = 'PASS' if ppl <= 25.0 else 'FAIL'
        print(f"  {name:<20} {ppl:>10.2f} {status:>10}")

    # Hypothesis testing
    m1_ppl = ppl_results.get('M1-Forward', float('inf'))
    m2_ppl = ppl_results.get('M2-Reverse', float('inf'))
    m3_ppl = ppl_results.get('M3-Structured', float('inf'))
    m4_ppl = ppl_results.get('M4-LLaMA', float('inf'))

    print("\n  HYPOTHESIS RESULTS:")
    if m1_ppl < float('inf') and m2_ppl < float('inf'):
        h1 = "SUPPORTED" if m2_ppl > m1_ppl else "NOT SUPPORTED"
        print(f"  H1 (Reverse PPL > Forward PPL): {h1}")
        print(f"      M1={m1_ppl:.2f}, M2={m2_ppl:.2f}")

    if m1_ppl < float('inf') and m3_ppl < float('inf'):
        h2 = "SUPPORTED" if m3_ppl < m1_ppl else "NOT SUPPORTED"
        print(f"  H2 (Structured PPL < Forward PPL): {h2}")
        print(f"      M1={m1_ppl:.2f}, M3={m3_ppl:.2f}")

    if m1_ppl < float('inf') and m4_ppl < float('inf'):
        if m4_ppl < m1_ppl:
            h3 = "LLaMA BETTER"
        elif abs(m4_ppl - m1_ppl) / m1_ppl < 0.05:
            h3 = "APPROXIMATELY EQUAL (data bottleneck)"
        else:
            h3 = "GPT-2 BETTER"
        print(f"  H3 (LLaMA vs GPT-2): {h3}")
        print(f"      M1={m1_ppl:.2f}, M4={m4_ppl:.2f}")

    # Save perplexity table as markdown
    md_path = os.path.join(results_dir, 'perplexity_table.md')
    with open(md_path, 'w') as f:
        f.write("# Perplexity Results\n\n")
        f.write("| Model | PPL | Status |\n")
        f.write("|-------|-----|--------|\n")
        for name in ['M1-Forward', 'M2-Reverse', 'M3-Structured', 'M4-LLaMA']:
            ppl = ppl_results.get(name, float('inf'))
            status = ''
            if name == 'M1-Forward':
                status = 'PASS' if ppl <= 25.0 else 'FAIL'
            f.write(f"| {name} | {ppl:.2f} | {status} |\n")
    print(f"\n  Saved perplexity table to {md_path}")

    # Generate PPL comparison chart
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        names = ['M1-Forward', 'M2-Reverse', 'M3-Structured', 'M4-LLaMA']
        ppls = [ppl_results.get(n, 0) for n in names]
        colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, ppls, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(y=25.0, color='orange', linestyle='--', linewidth=2, label='PPL = 25.0 threshold')
        ax.set_ylabel('Perplexity', fontsize=12)
        ax.set_title('ROCStories Perplexity Comparison (Demo 3)', fontsize=14)
        ax.legend(fontsize=10)

        for bar, ppl in zip(bars, ppls):
            if ppl < float('inf') and ppl > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                        f'{ppl:.1f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        chart_path = os.path.join(results_dir, 'ppl_comparison.png')
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"  Saved PPL chart to {chart_path}")
    except ImportError:
        print("  matplotlib not available, skipping chart generation")

    print("\n" + "=" * 60)
    print("  Evaluation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
