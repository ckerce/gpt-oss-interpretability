#!/usr/bin/env python3
"""Extract semantic decision trajectories from logit-lens artifacts.

For each tracked position, identifies "decision layers" where the model's
top-1 prediction changes, and reports which tokens gain/lose probability.

This demonstrates the CASCADE insight: the model's own computation trajectory
reveals steering directions, without requiring curated contrast pairs.

Output: runs/decision_vectors/ with per-prompt markdown reports.
"""

import json
import math
from pathlib import Path


RUNS = Path("runs")
OUTPUT = RUNS / "decision_vectors"


def _label_semantic_category(token: str) -> str:
    """Rough semantic label for a token."""
    t = token.strip()
    if t in ("<|endoftext|>", ""):
        return "padding"
    if t in (",", ".", ";", ":", "!", "?", "'", '"', "-", "—", "(", ")"):
        return "punctuation"
    if t.isdigit():
        return "number"
    if len(t) <= 2 and t.isalpha():
        return "letter/short"
    return "content"


def extract_decisions_for_position(pos_data: dict, prompt_label: str) -> dict:
    """Analyze one tracked position's trajectory for decision transitions."""
    trajectory = pos_data["trajectory"]
    target_token = pos_data["target_token"]
    conv_layer = pos_data["convergence_layer"]
    position = pos_data["position"]

    decisions = []
    prev_top1 = None
    prev_top1_token = None

    for i, entry in enumerate(trajectory):
        layer = entry["layer"]
        top1_id = entry["top_token_ids"][0]
        top1_token = entry["top_tokens"][0]
        top1_logprob = entry["top_logprobs"][0]

        # Target token tracking
        target_rank = entry.get("target_rank")
        target_logprob = entry.get("target_logprob")

        if prev_top1 is not None and top1_id != prev_top1:
            # Decision transition detected
            prev_entry = trajectory[i - 1]

            # Find tokens that gained/lost between layers
            prev_tokens = dict(zip(prev_entry["top_token_ids"], prev_entry["top_logprobs"]))
            curr_tokens = dict(zip(entry["top_token_ids"], entry["top_logprobs"]))

            # Top-5 gainers: tokens with biggest logprob increase
            all_ids = set(prev_tokens.keys()) | set(curr_tokens.keys())
            deltas = []
            for tid in all_ids:
                prev_lp = prev_tokens.get(tid, -20.0)  # assume low if absent
                curr_lp = curr_tokens.get(tid, -20.0)
                delta = curr_lp - prev_lp
                # Find the token string
                if tid in dict(zip(prev_entry["top_token_ids"], prev_entry["top_tokens"])):
                    tok_str = dict(zip(prev_entry["top_token_ids"], prev_entry["top_tokens"]))[tid]
                elif tid in dict(zip(entry["top_token_ids"], entry["top_tokens"])):
                    tok_str = dict(zip(entry["top_token_ids"], entry["top_tokens"]))[tid]
                else:
                    tok_str = f"[id={tid}]"
                deltas.append((tid, tok_str, delta, curr_lp))

            deltas.sort(key=lambda x: x[2], reverse=True)
            gainers = [(t[1], round(t[2], 2)) for t in deltas[:5]]
            losers = [(t[1], round(t[2], 2)) for t in deltas[-5:]]

            # Semantic label for the transition
            prev_cat = _label_semantic_category(prev_top1_token)
            curr_cat = _label_semantic_category(top1_token)
            semantic_label = f"{prev_cat}→{curr_cat}"
            if curr_cat == "content":
                semantic_label = f"{prev_cat}→'{top1_token.strip()}'"

            decisions.append({
                "from_layer": layer - 1,
                "to_layer": layer,
                "from_token": prev_top1_token,
                "to_token": top1_token,
                "semantic_label": semantic_label,
                "gainers": gainers,
                "losers": losers,
            })

        prev_top1 = top1_id
        prev_top1_token = top1_token

    # Build target trajectory (probability of final answer at each layer)
    target_trajectory = []
    for entry in trajectory:
        target_logprob = entry.get("target_logprob")
        if target_logprob is not None:
            prob = math.exp(target_logprob) if target_logprob > -50 else 0.0
            target_trajectory.append({
                "layer": entry["layer"],
                "target_logprob": round(target_logprob, 3),
                "target_prob": round(prob, 6),
                "target_rank": entry.get("target_rank"),
            })

    return {
        "position": position,
        "target_token": target_token,
        "convergence_layer": conv_layer,
        "num_decisions": len(decisions),
        "decisions": decisions,
        "target_trajectory": target_trajectory,
    }


def format_position_report(pos_result: dict) -> str:
    """Format a single position's decision analysis as markdown."""
    lines = []
    lines.append(f"### Position {pos_result['position']}: target = `{pos_result['target_token']}`")
    lines.append(f"- Convergence layer: **{pos_result['convergence_layer']}**")
    lines.append(f"- Decision transitions: **{pos_result['num_decisions']}**")
    lines.append("")

    if not pos_result["decisions"]:
        lines.append("No top-1 changes detected (stable from layer 0).\n")
        return "\n".join(lines)

    lines.append("| Transition | From → To | Semantic | Top gainers | Top losers |")
    lines.append("|---|---|---|---|---|")

    for d in pos_result["decisions"]:
        fr = f"L{d['from_layer']}→L{d['to_layer']}"
        tokens = f"`{d['from_token']}` → `{d['to_token']}`"
        sem = d["semantic_label"]
        gainers_str = ", ".join(f"`{t}` (+{v})" for t, v in d["gainers"][:3])
        losers_str = ", ".join(f"`{t}` ({v})" for t, v in d["losers"][:3])
        lines.append(f"| {fr} | {tokens} | {sem} | {gainers_str} | {losers_str} |")

    lines.append("")

    # Narrative summary of the decision arc
    if pos_result["decisions"]:
        arc = " → ".join(
            f"L{d['from_layer']}: `{d['from_token'].strip()}`"
            for d in pos_result["decisions"]
        )
        final = pos_result["decisions"][-1]
        arc += f" → L{final['to_layer']}: `{final['to_token'].strip()}`"
        lines.append(f"**Decision arc:** {arc}")
        lines.append("")

    return "\n".join(lines)


def process_prompt(prompt_name: str, data: dict) -> dict:
    """Process all tracked positions for one prompt."""
    results = []
    for tp in data["tracked_positions"]:
        result = extract_decisions_for_position(tp, prompt_name)
        results.append(result)
    return {
        "prompt": data["prompt"],
        "model": data["model"],
        "num_layers": data["num_layers"],
        "positions": results,
    }


def write_report(prompt_name: str, analysis: dict, output_dir: Path):
    """Write markdown and JSON reports for one prompt."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / f"{prompt_name}_decisions.json"
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)

    # Markdown report
    md_lines = []
    md_lines.append(f"# Decision Trajectory: {prompt_name}")
    md_lines.append(f"\n**Prompt:** `{analysis['prompt']}`")
    md_lines.append(f"**Model:** {analysis['model']}")
    md_lines.append(f"**Layers:** {analysis['num_layers']}")
    md_lines.append("")

    # Focus on key decision positions
    key_positions = [
        p for p in analysis["positions"]
        if p["num_decisions"] >= 2 and p["convergence_layer"] is not None
    ]
    key_positions.sort(key=lambda p: p["convergence_layer"] or 999)

    md_lines.append(f"## Key Decision Positions ({len(key_positions)} positions with 2+ transitions)")
    md_lines.append("")

    for pos in key_positions:
        md_lines.append(format_position_report(pos))

    # Summary statistics
    md_lines.append("## Summary")
    conv_layers = [p["convergence_layer"] for p in analysis["positions"]
                   if p["convergence_layer"] is not None]
    if conv_layers:
        md_lines.append(f"- Mean convergence layer: {sum(conv_layers)/len(conv_layers):.1f}")
        md_lines.append(f"- Range: L{min(conv_layers)} – L{max(conv_layers)}")

    total_decisions = sum(p["num_decisions"] for p in analysis["positions"])
    md_lines.append(f"- Total decision transitions across all positions: {total_decisions}")
    md_lines.append("")

    # Key insight callout
    md_lines.append("## CASCADE Insight")
    md_lines.append("")
    md_lines.append("Each decision transition (top-1 change) identifies a layer where the model's")
    md_lines.append("prediction shifts. In CASCADE mode, the logit-space difference `Δz = z^(l+1) − z^(l)`")
    md_lines.append("at a decision layer is a **self-supervised steering direction**: it captures the")
    md_lines.append("semantic decision the model makes, without requiring curated contrast pairs.")
    md_lines.append("")
    md_lines.append("The gauge-safe projection `v_steer = (CW)⁺ · C·Δz` maps this direction into the")
    md_lines.append("student's embedding space, yielding a closed-form steering vector. This is")
    md_lines.append("fundamentally different from contrastive activation addition (CAA), which requires")
    md_lines.append("100+ positive/negative example pairs per concept.")
    md_lines.append("")

    md_path = output_dir / f"{prompt_name}_decisions.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"  {prompt_name}: {len(key_positions)} key positions, "
          f"{total_decisions} total transitions → {md_path.name}")


def main():
    print("Extracting decision vectors from logit-lens artifacts...")

    prompts = {
        "recency": RUNS / "logit_lens_recency" / "logit_lens_data.json",
        "syntax": RUNS / "logit_lens_syntax" / "logit_lens_data.json",
        "induction": RUNS / "logit_lens_induction" / "logit_lens_data.json",
    }

    for name, path in prompts.items():
        if not path.exists():
            print(f"  SKIP {name}: {path} not found")
            continue
        data = json.loads(path.read_text())
        analysis = process_prompt(name, data)
        write_report(name, analysis, OUTPUT)

    print(f"\nAll reports saved to {OUTPUT}/")


if __name__ == "__main__":
    main()
