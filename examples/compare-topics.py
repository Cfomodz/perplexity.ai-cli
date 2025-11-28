#!/usr/bin/env python3
"""
Compare Topics Script

Uses multiple Perplexity queries to create a comprehensive comparison
between two or more topics/technologies/options.

Usage:
    python compare-topics.py "Topic A" "Topic B"
    python compare-topics.py "Python" "Rust" "Go"

Example:
    python compare-topics.py "React" "Vue" "Svelte"
"""

import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
CLI_PATH = SCRIPT_DIR.parent / "perplexity.ai-cli.py"
VENV_PYTHON = SCRIPT_DIR.parent / ".venv" / "bin" / "python"


def run_query(query: str, mode: str = None, model: str = None) -> str:
    """Run a CLI query and return the output."""
    cmd = [str(VENV_PYTHON), str(CLI_PATH), "--no-typing"]
    
    if mode == "research":
        cmd.append("--research")
    if model:
        cmd.extend(["--model", model])
    
    cmd.append(query)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return result.stdout
    except Exception as e:
        return f"Error: {e}"


def compare_topics(topics: list[str], output_file: str = None):
    """Generate a comprehensive comparison of topics."""
    
    topics_str = " vs ".join(topics)
    topics_list = ", ".join(topics)
    
    print(f"\n{'═' * 60}")
    print(f"COMPARING: {topics_str}")
    print(f"{'═' * 60}\n")
    
    results = []
    
    # Step 1: Overview of each topic
    print("[1/5] Getting overview of each topic...")
    for topic in topics:
        print(f"  → {topic}")
        overview = run_query(f"Give a brief overview of {topic}. What is it and what are its main characteristics?")
        results.append(f"\n## Overview: {topic}\n{overview}")
    
    # Step 2: Direct comparison
    print("\n[2/5] Creating direct comparison...")
    comparison = run_query(
        f"Create a detailed comparison between {topics_list}. "
        f"Compare their strengths, weaknesses, use cases, and target audiences. "
        f"Use a structured format.",
        mode="research"
    )
    results.append(f"\n## Direct Comparison\n{comparison}")
    
    # Step 3: Pros and cons table
    print("[3/5] Generating pros and cons...")
    pros_cons = run_query(
        f"Create a pros and cons table for {topics_list}. "
        f"List 5 pros and 5 cons for each option.",
        model="claude"
    )
    results.append(f"\n## Pros and Cons\n{pros_cons}")
    
    # Step 4: Use case recommendations
    print("[4/5] Getting use case recommendations...")
    use_cases = run_query(
        f"When should someone choose each of these: {topics_list}? "
        f"Give specific scenarios and use cases where each option excels.",
        model="gpt"
    )
    results.append(f"\n## Use Case Recommendations\n{use_cases}")
    
    # Step 5: Final verdict
    print("[5/5] Generating final verdict...")
    verdict = run_query(
        f"If you had to recommend one of {topics_list} for a general audience, "
        f"which would you pick and why? Also mention scenarios where the others might be better.",
        model="gemini"
    )
    results.append(f"\n## Final Verdict\n{verdict}")
    
    # Combine results
    full_report = f"""
{'=' * 70}
COMPARISON REPORT: {topics_str}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 70}

{''.join(results)}

{'=' * 70}
END OF REPORT
{'=' * 70}
"""
    
    # Save or print
    if output_file:
        Path(output_file).write_text(full_report)
        print(f"\n✓ Report saved to: {output_file}")
    else:
        print(full_report)
    
    return full_report


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple topics using Perplexity",
        epilog="""
Examples:
  %(prog)s "Python" "JavaScript"
  %(prog)s "React" "Vue" "Svelte" -o comparison.txt
  %(prog)s "AWS" "GCP" "Azure"
"""
    )
    
    parser.add_argument("topics", nargs="+", help="Topics to compare (2 or more)")
    parser.add_argument("-o", "--output", help="Output file (optional)")
    
    args = parser.parse_args()
    
    if len(args.topics) < 2:
        print("Error: Please provide at least 2 topics to compare")
        sys.exit(1)
    
    compare_topics(args.topics, args.output)


if __name__ == "__main__":
    main()

