#!/usr/bin/env python3
"""
Research Workflow Script

This script demonstrates using the Perplexity CLI programmatically to accomplish
a multi-step research and analysis process.

Usage:
    python research-workflow.py "Your research topic"
    python research-workflow.py --quick "Quick topic"  # Abbreviated workflow

Example:
    python research-workflow.py "Impact of AI on healthcare in 2025"
"""

import subprocess
import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List

# Find the CLI path relative to this script
SCRIPT_DIR = Path(__file__).parent
CLI_PATH = SCRIPT_DIR.parent / "perplexity.ai-cli.py"
VENV_PYTHON = SCRIPT_DIR.parent / ".venv" / "bin" / "python"


@dataclass
class ResearchStep:
    """A single step in the research workflow."""
    name: str
    description: str
    query: str
    mode: Optional[str] = None  # "research", "labs", None for standard
    model: Optional[str] = None
    focus: Optional[str] = None
    output_file: Optional[str] = None


class ResearchWorkflow:
    """Orchestrates multi-step research using the Perplexity CLI."""
    
    def __init__(self, topic: str, output_dir: Optional[Path] = None):
        self.topic = topic
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        safe_topic = "".join(c if c.isalnum() or c in "_-" else "_" for c in topic)[:50]
        self.output_dir = output_dir or Path(f"./research-output/{safe_topic}_{self.timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
    
    def run_cli(self, query: str, mode: str = None, model: str = None, 
                focus: str = None, timeout: int = 120) -> str:
        """Run the Perplexity CLI with the given parameters."""
        cmd = [str(VENV_PYTHON), str(CLI_PATH), "--no-typing"]
        
        if mode == "research":
            cmd.append("--research")
        elif mode == "labs":
            cmd.append("--labs")
        
        if model:
            cmd.extend(["--model", model])
        
        if focus:
            cmd.extend(["--focus", focus])
        
        cmd.append(query)
        
        print(f"\n{'─' * 60}")
        print(f"Running: {' '.join(cmd[:4])} ... \"{query[:50]}...\"")
        print(f"{'─' * 60}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout * 3 if mode in ("research", "labs") else timeout
            )
            
            output = result.stdout
            if result.returncode != 0:
                print(f"Warning: CLI returned non-zero exit code: {result.returncode}")
                if result.stderr:
                    print(f"Stderr: {result.stderr}")
            
            return output
            
        except subprocess.TimeoutExpired:
            print(f"Warning: Request timed out after {timeout}s")
            return ""
        except Exception as e:
            print(f"Error running CLI: {e}")
            return ""
    
    def execute_step(self, step: ResearchStep) -> str:
        """Execute a single research step."""
        print(f"\n{'═' * 60}")
        print(f"STEP: {step.name}")
        print(f"{'═' * 60}")
        print(f"Description: {step.description}")
        
        # Substitute topic into query
        query = step.query.format(topic=self.topic)
        
        result = self.run_cli(
            query=query,
            mode=step.mode,
            model=step.model,
            focus=step.focus
        )
        
        # Save to file
        if step.output_file:
            output_path = self.output_dir / step.output_file
            with open(output_path, "w") as f:
                f.write(f"# {step.name}\n")
                f.write(f"# {step.description}\n")
                f.write(f"# Query: {query}\n")
                f.write(f"# Mode: {step.mode or 'standard'}, Model: {step.model or 'auto'}\n")
                f.write(f"{'=' * 60}\n\n")
                f.write(result)
            print(f"Saved to: {output_path}")
        
        self.results[step.name] = result
        return result
    
    def run_full_workflow(self):
        """Run the complete research workflow."""
        steps = [
            ResearchStep(
                name="Deep Research",
                description="Comprehensive analysis using Research mode",
                query="Provide a comprehensive analysis of: {topic}. Include current state, key players, recent developments, challenges, and future outlook.",
                mode="research",
                output_file="01_deep_research.txt"
            ),
            ResearchStep(
                name="Claude Perspective",
                description="Ethical and societal analysis from Claude",
                query="Regarding '{topic}': What are the most significant ethical considerations and potential societal impacts? Be specific and analytical.",
                model="claude",
                output_file="02a_claude_perspective.txt"
            ),
            ResearchStep(
                name="GPT Perspective", 
                description="Technical innovations from GPT",
                query="Regarding '{topic}': What are the key technical innovations and breakthroughs driving progress? Focus on concrete examples.",
                model="gpt",
                output_file="02b_gpt_perspective.txt"
            ),
            ResearchStep(
                name="Gemini Perspective",
                description="Economic analysis from Gemini",
                query="Regarding '{topic}': What are the economic implications and market opportunities? Include data points where possible.",
                model="gemini",
                output_file="02c_gemini_perspective.txt"
            ),
            ResearchStep(
                name="Academic Research",
                description="Academic papers and citations",
                query="What are the most cited recent academic papers and research studies about: {topic}? Summarize their key findings.",
                focus="academic",
                output_file="03_academic_research.txt"
            ),
            ResearchStep(
                name="Visual Timeline",
                description="Generate timeline using Labs mode",
                query="Create a timeline showing the major milestones and developments in: {topic}. Include dates and brief descriptions.",
                mode="labs",
                output_file="04_labs_timeline.txt"
            ),
            ResearchStep(
                name="Critical Analysis",
                description="Devil's advocate perspective from Grok",
                query="Play devil's advocate on '{topic}'. What are the strongest arguments against the mainstream narrative? What risks are being overlooked?",
                model="grok",
                output_file="05_critical_analysis.txt"
            ),
            ResearchStep(
                name="Executive Summary",
                description="Final executive summary",
                query="Create an executive summary about '{topic}' that includes: 1) Key findings, 2) Opportunities and challenges, 3) Expert perspectives, 4) Recommendations. Format professionally.",
                mode="research",
                output_file="06_executive_summary.txt"
            ),
        ]
        
        print(f"\n{'╔' + '═' * 58 + '╗'}")
        print(f"║{'PERPLEXITY MULTI-STEP RESEARCH WORKFLOW':^58}║")
        print(f"{'╚' + '═' * 58 + '╝'}")
        print(f"\nTopic: {self.topic}")
        print(f"Output: {self.output_dir}")
        print(f"Steps: {len(steps)}")
        
        for i, step in enumerate(steps, 1):
            print(f"\n[{i}/{len(steps)}] ", end="")
            self.execute_step(step)
        
        self._generate_report()
        
        print(f"\n{'╔' + '═' * 58 + '╗'}")
        print(f"║{'RESEARCH COMPLETE':^58}║")
        print(f"{'╚' + '═' * 58 + '╝'}")
        print(f"\nAll files saved to: {self.output_dir}")
        print(f"Full report: {self.output_dir / 'FULL_REPORT.txt'}")
    
    def run_quick_workflow(self):
        """Run an abbreviated workflow for quick research."""
        steps = [
            ResearchStep(
                name="Quick Research",
                description="Fast comprehensive analysis",
                query="Give me a quick but thorough overview of: {topic}. Include key facts, current state, and main considerations.",
                mode="research",
                output_file="01_quick_research.txt"
            ),
            ResearchStep(
                name="Key Insights",
                description="Most important insights",
                query="What are the 5 most important things to know about: {topic}? Be concise but informative.",
                model="claude",
                output_file="02_key_insights.txt"
            ),
            ResearchStep(
                name="Action Items",
                description="Practical next steps",
                query="Based on '{topic}', what are the key action items or next steps someone should consider? Be practical and specific.",
                output_file="03_action_items.txt"
            ),
        ]
        
        print(f"\n{'╔' + '═' * 58 + '╗'}")
        print(f"║{'QUICK RESEARCH WORKFLOW':^58}║")
        print(f"{'╚' + '═' * 58 + '╝'}")
        print(f"\nTopic: {self.topic}")
        print(f"Output: {self.output_dir}")
        
        for i, step in enumerate(steps, 1):
            print(f"\n[{i}/{len(steps)}] ", end="")
            self.execute_step(step)
        
        self._generate_report()
        
        print(f"\n✓ Quick research complete: {self.output_dir}")
    
    def _generate_report(self):
        """Generate a combined report from all results."""
        report_path = self.output_dir / "FULL_REPORT.txt"
        
        with open(report_path, "w") as f:
            f.write(f"{'=' * 70}\n")
            f.write(f"RESEARCH REPORT: {self.topic}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 70}\n\n")
            
            # Combine all output files in order
            for file_path in sorted(self.output_dir.glob("*.txt")):
                if file_path.name != "FULL_REPORT.txt":
                    f.write(f"\n{'─' * 70}\n")
                    f.write(file_path.read_text())
                    f.write("\n")
        
        # Also save metadata as JSON
        metadata = {
            "topic": self.topic,
            "timestamp": self.timestamp,
            "output_dir": str(self.output_dir),
            "steps_completed": list(self.results.keys()),
        }
        
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-step research workflow using Perplexity CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Impact of AI on healthcare"
  %(prog)s --quick "Best practices for Python testing"
  %(prog)s --output ./my-research "Climate change solutions"
"""
    )
    
    parser.add_argument("topic", help="The research topic to investigate")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Run abbreviated workflow (3 steps instead of 8)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory for research files")
    
    args = parser.parse_args()
    
    # Verify CLI exists
    if not CLI_PATH.exists():
        print(f"Error: CLI not found at {CLI_PATH}")
        sys.exit(1)
    
    if not VENV_PYTHON.exists():
        print(f"Error: Python venv not found at {VENV_PYTHON}")
        print("Trying system python...")
        global VENV_PYTHON
        VENV_PYTHON = Path(sys.executable)
    
    # Run workflow
    workflow = ResearchWorkflow(args.topic, args.output)
    
    if args.quick:
        workflow.run_quick_workflow()
    else:
        workflow.run_full_workflow()


if __name__ == "__main__":
    main()

