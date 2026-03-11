#!/usr/bin/env python3
"""Extract computational-mode feature vectors from gpt-oss-20b.

Implements the extended Tier-2 feature system from the NeurIPS 2026
activation clustering paper, adapted for MoE architectures. Produces
per-token feature vectors and geometric analysis.

Usage:
    # Single prompt
    python scripts/run_feature_extraction.py \
        --model openai/gpt-oss-20b \
        --prompt "The trophy would not fit in the suitcase because the suitcase was too" \
        --output runs/features_demo/

    # Multiple prompts from file (one per line)
    python scripts/run_feature_extraction.py \
        --model openai/gpt-oss-20b \
        --prompts-file prompts.txt \
        --output runs/features_corpus/

    # Logit-lens only (no attention capture — faster, smaller feature vector)
    python scripts/run_feature_extraction.py \
        --model openai/gpt-oss-20b \
        --prompt "The keys to the cabinet" \
        --no-attention \
        --output runs/features_fast/
"""

import argparse
import json
import os
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser(description="Feature extraction for gpt-oss-20b")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model name or path")
    parser.add_argument("--prompt", type=str, help="Single prompt to analyze")
    parser.add_argument("--prompts-file", type=str, help="File with one prompt per line")
    parser.add_argument("--output", default="runs/features/", help="Output directory")
    parser.add_argument("--no-attention", action="store_true", help="Skip attention capture")
    parser.add_argument("--dtype", default="auto", help="Model dtype")
    args = parser.parse_args()

    # Collect prompts
    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts.extend(line.strip() for line in f if line.strip())
    if not prompts:
        parser.error("Provide --prompt or --prompts-file")

    os.makedirs(args.output, exist_ok=True)

    # Load model
    from gpt_oss_interp.backends.transformers_gpt_oss import GPTOSSTransformersBackend
    from gpt_oss_interp.features.extractor import (
        FeatureConfig,
        MoEFeatureExtractor,
        extract_features_from_backend,
    )
    from gpt_oss_interp.features.geometry import (
        analyze_geometry,
        format_geometric_report,
    )

    print(f"Loading model: {args.model}")
    backend = GPTOSSTransformersBackend(model_name=args.model, dtype=args.dtype)

    config = FeatureConfig(
        n_layers=backend.structure.num_layers,
        n_query_heads=backend.NUM_HEADS,
        n_kv_heads=backend.NUM_KV_HEADS,
        n_experts=backend.NUM_EXPERTS,
        top_k_experts=backend.TOP_K,
        head_dim=backend.HEAD_DIM,
        include_head_activation=not args.no_attention,
        include_head_entropy=not args.no_attention,
        include_attention_scale=not args.no_attention,
    )

    print(f"\nFeature configuration:")
    print(f"  Feature dimensions: {config.feature_dim}")
    print(f"  Attention capture: {not args.no_attention}")
    print(f"  Expert routing: {config.include_expert_routing}")

    # Extract features for each prompt
    all_features = []
    all_depths = []
    all_tokens = []
    all_results = []

    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i+1}/{len(prompts)} ---")
        print(f"  {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

        t0 = time.time()
        result = extract_features_from_backend(backend, prompt, config)
        elapsed = time.time() - t0

        print(f"  Tokens: {result.num_tokens}, Features: {result.feature_dim}, Time: {elapsed:.1f}s")
        print(f"  Depth range: {result.processing_depth.min().item():.0f} - {result.processing_depth.max().item():.0f}")
        print(f"  Confidence range: {result.confidence.min().item():.3f} - {result.confidence.max().item():.3f}")
        print(f"  Has attention: {result.metadata.get('has_attention', False)}")
        print(f"  Has routing: {result.metadata.get('has_routing', False)}")

        all_features.append(result.feature_vectors)
        all_depths.append(result.processing_depth)
        all_tokens.extend(result.token_strings)
        all_results.append(result)

        # Save per-prompt results
        prompt_dir = os.path.join(args.output, f"prompt_{i:03d}")
        os.makedirs(prompt_dir, exist_ok=True)

        torch.save(result.feature_vectors, os.path.join(prompt_dir, "features.pt"))
        torch.save(result.processing_depth, os.path.join(prompt_dir, "depth.pt"))
        with open(os.path.join(prompt_dir, "tokens.json"), "w") as f:
            json.dump({
                "prompt": prompt,
                "tokens": result.token_strings,
                "processing_depth": result.processing_depth.tolist(),
                "confidence": result.confidence.tolist(),
                "metadata": result.metadata,
            }, f, indent=2)

    # Geometric analysis over all tokens
    if all_features:
        print("\n=== Geometric Analysis ===")
        combined_features = torch.cat(all_features, dim=0)
        combined_depths = torch.cat(all_depths, dim=0)

        summary = analyze_geometry(combined_features, combined_depths)
        report = format_geometric_report(summary, all_tokens)
        print(report)

        # Save report
        with open(os.path.join(args.output, "geometric_analysis.md"), "w") as f:
            f.write(report)

        # Save combined features for downstream analysis
        torch.save(combined_features, os.path.join(args.output, "all_features.pt"))
        torch.save(combined_depths, os.path.join(args.output, "all_depths.pt"))
        with open(os.path.join(args.output, "all_tokens.json"), "w") as f:
            json.dump(all_tokens, f)

        print(f"\nResults saved to {args.output}")
        print(f"  Feature tensor: {combined_features.shape}")
        print(f"  Intrinsic dimension: {summary.intrinsic_dim_pca:.0f} / {summary.feature_dim}")
        print(f"  Depth stratification: {summary.depth_stratification:.3f}")


if __name__ == "__main__":
    main()
