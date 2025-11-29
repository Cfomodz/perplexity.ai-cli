#!/usr/bin/env python3
"""
Task Orchestrator

Uses a planner prompt to decompose a high-level goal into isolated sub-tasks,
then executes each sub-task with its designated model. Sub-tasks receive only
general context - no results from sibling tasks - producing independent fragments
that assemble into a larger whole.

Usage:
    python orchestrator.py "Your high-level goal or question"

Example:
    python orchestrator.py "Create a comprehensive business plan for a SaaS startup"
"""

import subprocess
import sys
import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List

SCRIPT_DIR = Path(__file__).parent
CLI_PATH = SCRIPT_DIR / "perplexity.ai-cli.py"
VENV_PYTHON = SCRIPT_DIR / ".venv" / "bin" / "python"
TEMP_DIR = SCRIPT_DIR / "temp"  # Persistent state directory


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

MAX_DEPTH = 5                    # Maximum levels of task decomposition
MAX_TOTAL_TASKS = 100            # Maximum total atomic tasks allowed
TASKS_UNTIL_CONFIRM = 20         # Ask for human confirmation after N tasks


# ═══════════════════════════════════════════════════════════════════════════
# PLANNER PROMPT
# ═══════════════════════════════════════════════════════════════════════════

# Planner prompt - multi-line is OK now with --paste mode
PLANNER_PROMPT = '''You are a task decomposition planner. Break down the following goal into 3-5 independent sub-tasks.

GOAL: {goal}

Return a valid JSON object with this structure:
- goal: (string) The goal being analyzed
- context: (string) Brief context summary
- subtasks: (array of objects), each containing:
  - id: (string) "1", "2", etc.
  - title: (string) Short descriptive title
  - model: (string) One of: claude, gpt, gemini, grok, sonar
  - focus: (string or null) Optional focus mode (academic, youtube, etc)
  - mode: (string or null) "research" for deep dives, "labs" for visual/code, or null
  - prompt: (string) Detailed, self-contained prompt for the agent to execute this task
  - contribution: (string) How this task contributes to the final goal
  - is_atomic: (boolean) true if simple, false if it needs further breakdown

Generate actual content for the goal. Return ONLY the JSON.'''


# JSON Fixer prompt
FIXER_PROMPT = '''You are a JSON repair expert. The following JSON is invalid. Please fix it and return ONLY the corrected, valid JSON object.

INVALID JSON:
{broken_json}

ERROR:
{error_msg}

Return ONLY the fixed JSON string. No other text.'''


@dataclass
class SubTask:
    """A decomposed sub-task to execute."""
    id: str  # Changed to string for hierarchical IDs (e.g., "1.2")
    title: str
    model: str
    prompt: str
    contribution: str
    is_atomic: bool = True
    focus: Optional[str] = None
    mode: Optional[str] = None
    result: Optional[str] = None
    subtasks: List['SubTask'] = field(default_factory=list)


class TaskOrchestrator:
    """Orchestrates multi-step task execution with isolated sub-tasks."""
    
    def __init__(self, goal: str, output_dir: Optional[Path] = None):
        self.goal = goal
        self.context = ""
        self.subtasks: list[SubTask] = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Task tracking
        self.total_tasks_created = 0
        self.tasks_executed = 0
        self.user_cancelled = False
        self.phase = "init"  # Track current phase: init, planning, executing, assembling, complete
        
        safe_goal = "".join(c if c.isalnum() or c in "_-" else "_" for c in goal)[:40]
        self.run_id = f"{safe_goal}_{self.timestamp}"
        self.output_dir = output_dir or Path(f"./orchestrator-output/{self.run_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up temp directory for incremental state
        self.temp_dir = TEMP_DIR / self.run_id
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.temp_dir / "state.json"
        
        # Save initial state
        self.save_state()
    
    def _subtask_to_dict(self, task: SubTask) -> dict:
        """Convert a SubTask to a serializable dictionary (including results)."""
        return {
            "id": task.id,
            "title": task.title,
            "model": task.model,
            "prompt": task.prompt,
            "contribution": task.contribution,
            "is_atomic": task.is_atomic,
            "focus": task.focus,
            "mode": task.mode,
            "result": task.result,
            "status": "completed" if task.result else ("pending" if not task.subtasks else "decomposed"),
            "subtasks": [self._subtask_to_dict(st) for st in task.subtasks]
        }
    
    def _dict_to_subtask(self, data: dict) -> SubTask:
        """Restore a SubTask from a dictionary."""
        task = SubTask(
            id=data["id"],
            title=data["title"],
            model=data["model"],
            prompt=data["prompt"],
            contribution=data["contribution"],
            is_atomic=data.get("is_atomic", True),
            focus=data.get("focus"),
            mode=data.get("mode"),
            result=data.get("result")
        )
        task.subtasks = [self._dict_to_subtask(st) for st in data.get("subtasks", [])]
        return task
    
    def save_state(self):
        """Save current orchestration state to temp directory."""
        state = {
            "meta": {
                "run_id": self.run_id,
                "goal": self.goal,
                "context": self.context,
                "timestamp": self.timestamp,
                "last_updated": datetime.now().isoformat(),
                "phase": self.phase
            },
            "config": {
                "max_depth": MAX_DEPTH,
                "max_total_tasks": MAX_TOTAL_TASKS,
                "tasks_until_confirm": TASKS_UNTIL_CONFIRM
            },
            "progress": {
                "total_tasks_created": self.total_tasks_created,
                "tasks_executed": self.tasks_executed,
                "user_cancelled": self.user_cancelled
            },
            "tasks": [self._subtask_to_dict(t) for t in self.subtasks]
        }
        
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load_state(cls, state_file: Path) -> Optional['TaskOrchestrator']:
        """Load orchestrator state from a state file for resumption."""
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            
            meta = state["meta"]
            orchestrator = cls(meta["goal"])
            orchestrator.run_id = meta["run_id"]
            orchestrator.context = meta["context"]
            orchestrator.timestamp = meta["timestamp"]
            orchestrator.phase = meta["phase"]
            
            progress = state["progress"]
            orchestrator.total_tasks_created = progress["total_tasks_created"]
            orchestrator.tasks_executed = progress["tasks_executed"]
            orchestrator.user_cancelled = progress["user_cancelled"]
            
            orchestrator.subtasks = [orchestrator._dict_to_subtask(t) for t in state["tasks"]]
            
            # Restore paths
            safe_goal = "".join(c if c.isalnum() or c in "_-" else "_" for c in meta["goal"])[:40]
            orchestrator.output_dir = Path(f"./orchestrator-output/{meta['run_id']}")
            orchestrator.temp_dir = TEMP_DIR / meta["run_id"]
            orchestrator.state_file = orchestrator.temp_dir / "state.json"
            
            return orchestrator
        except Exception as e:
            print(f"Error loading state: {e}")
            return None
    
    def run_cli(self, query: str, model: str = None, mode: str = None, 
                focus: str = None, timeout: int = 120) -> str:
        """Execute a CLI query."""
        cmd = [str(VENV_PYTHON), str(CLI_PATH), "--no-typing", "--paste"]
        
        if mode == "research":
            cmd.append("--research")
        elif mode == "labs":
            cmd.append("--labs")
        
        if model and model != "auto":
            cmd.extend(["--model", model])
        
        if focus:
            cmd.extend(["--focus", focus])
        
        cmd.append(query)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout * 3 if mode in ("research", "labs") else timeout
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return "[TIMEOUT]"
        except Exception as e:
            return f"[ERROR: {e}]"
    
    def repair_json(self, broken_json: str, error_msg: str, max_retries: int = 3) -> Optional[dict]:
        """Attempt to repair invalid JSON using an LLM, with retries."""
        print(f"  Repairing invalid JSON (max retries: {max_retries})...")
        
        current_error = error_msg
        current_json = broken_json
        
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"  Retry {attempt+1}/{max_retries}...")
                
            prompt = FIXER_PROMPT.format(broken_json=current_json, error_msg=current_error)
            response = self.run_cli(prompt, model="gpt", timeout=60)  # GPT is good at syntax
            
            # Extract JSON from response
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"  Repair attempt {attempt+1} failed: {e}")
                current_error = str(e)
                current_json = response # Feed the failed repair back as the input
            except Exception as e:
                print(f"  JSON repair error: {e}")
                return None
                
        print("  All repair attempts failed.")
        return None

    def decompose(self, task_description: str, parent_id: str = "", depth: int = 0, max_depth: int = None) -> list[SubTask]:
        """Recursively decompose a task into atomic sub-tasks."""
        if max_depth is None:
            max_depth = MAX_DEPTH
            
        # If we've reached max depth, treat as atomic
        if depth >= max_depth:
            print(f"  Max depth ({max_depth}) reached, treating remaining tasks as atomic")
            return []
        
        # Check if we've hit the task limit
        if self.total_tasks_created >= MAX_TOTAL_TASKS:
            print(f"  Max task limit ({MAX_TOTAL_TASKS}) reached, stopping decomposition")
            return []
            
        print(f"\nDecomposing task: {task_description[:50]}... (Depth {depth})")
        
        # Format planner prompt for this specific task
        prompt = PLANNER_PROMPT.format(goal=task_description)
        response = self.run_cli(prompt, model="claude", timeout=90)
        
        plan = None
        try:
            # Find JSON
            json_start = response.find('{"goal"')
            if json_start == -1:
                json_start = response.find('{\n  "goal"')
            if json_start == -1:
                json_start = response.find('{\"goal\"')
            
            if json_start == -1:
                print("  Warning: No JSON found in planner response, treating as atomic")
                return []
                
            # Extract JSON
            json_str = response[json_start:]
            brace_count = 0
            json_end = 0
            for i, char in enumerate(json_str):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end == 0:
                # If we couldn't find the end, maybe the model didn't close it properly
                # Try to repair the whole response string
                raise json.JSONDecodeError("Could not find closing brace", json_str, 0)
                
            plan = json.loads(json_str[:json_end])
            
        except json.JSONDecodeError as e:
            print(f"  Error parsing plan JSON: {e}")
            # Try to repair it
            plan = self.repair_json(response, str(e))
            if not plan:
                print(f"Response excerpt: {response[:500]}...")
                return []
        except Exception as e:
            print(f"  Error in decomposition: {e}")
            return []
            
        if not plan:
            return []

        subtasks = []
        try:
            # Set context if top level
            if depth == 0:
                self.context = plan.get("context", "")
            
            # Process subtasks
            for task_data in plan.get("subtasks", []):
                # Check task limit before creating more
                if self.total_tasks_created >= MAX_TOTAL_TASKS:
                    print(f"  Task limit ({MAX_TOTAL_TASKS}) reached during processing")
                    break
                    
                # Generate hierarchical ID (e.g. "1", "1.1", "1.1.1")
                task_id = str(task_data["id"])
                if parent_id:
                    task_id = f"{parent_id}.{task_id}"
                
                subtask = SubTask(
                    id=task_id,
                    title=task_data["title"],
                    model=task_data.get("model", "auto"),
                    prompt=task_data["prompt"],
                    contribution=task_data.get("contribution", ""),
                    is_atomic=task_data.get("is_atomic", True),
                    focus=task_data.get("focus"),
                    mode=task_data.get("mode")
                )
                
                self.total_tasks_created += 1
                
                # If not atomic and we have depth allowance, decompose further
                if not subtask.is_atomic and depth < max_depth - 1:
                    print(f"  Task {task_id} is not atomic, decomposing...")
                    children = self.decompose(subtask.prompt, task_id, depth + 1, max_depth)
                    if children:
                        subtask.subtasks = children
                        # If successfully decomposed, this task becomes a container
                        # and its children will be executed instead
                
                subtasks.append(subtask)
                
                # Save state after each task is created/decomposed
                if depth == 0:
                    self.subtasks = subtasks  # Update for state saving
                    self.save_state()
                
            return subtasks
            
        except Exception as e:
            print(f"  Error processing plan: {e}")
            return []

    def plan(self) -> bool:
        """Use the planner to decompose the goal into sub-tasks."""
        self.phase = "planning"
        self.save_state()
        
        print(f"\n{'═' * 70}")
        print("PHASE 1: TASK DECOMPOSITION")
        print(f"{'═' * 70}")
        print(f"\nGoal: {self.goal}")
        print("\nGenerating hierarchical execution plan...")
        
        self.subtasks = self.decompose(self.goal)
        
        # Save the full plan
        if self.subtasks:
            # Helper to serialize SubTask objects
            def to_dict(obj):
                if isinstance(obj, SubTask):
                    return {k: v for k, v in obj.__dict__.items() if k != 'result'}
                if isinstance(obj, list):
                    return [to_dict(x) for x in obj]
                return obj
                
            plan_data = {
                "goal": self.goal,
                "context": self.context,
                "tasks": [to_dict(t) for t in self.subtasks]
            }
            
            plan_path = self.output_dir / "00_execution_plan.json"
            with open(plan_path, "w") as f:
                json.dump(plan_data, f, indent=2, default=str)
            
            count = sum(1 for _ in self._flatten_tasks(self.subtasks))
            print(f"\n✓ Plan generated with {count} atomic sub-tasks total")
            print(f"  Plan saved to: {plan_path}")
            return True
            
        return False

    def _flatten_tasks(self, tasks: list[SubTask]) -> list[SubTask]:
        """Return a flat list of all atomic tasks to be executed."""
        flat = []
        for task in tasks:
            if task.subtasks:
                # If task has children, execute children
                flat.extend(self._flatten_tasks(task.subtasks))
            else:
                # Atomic task
                flat.append(task)
        return flat

    def execute_subtask(self, subtask: SubTask) -> str:
        """Execute a single sub-task in isolation."""
        print(f"\n{'─' * 60}")
        print(f"SUB-TASK {subtask.id}: {subtask.title}")
        print(f"{'─' * 60}")
        print(f"  Model: {subtask.model}")
        print(f"  Mode: {subtask.mode or 'standard'}")
        print(f"  Focus: {subtask.focus or 'default'}")
        print(f"  Contribution: {subtask.contribution}")
        
        # Build the isolated prompt with only general context (single line to avoid Enter key issues)
        isolated_prompt = f"CONTEXT: {self.context}. TASK: {subtask.prompt}. Provide a thorough response focused specifically on this aspect."
        
        result = self.run_cli(
            isolated_prompt,
            model=subtask.model,
            mode=subtask.mode,
            focus=subtask.focus
        )
        
        subtask.result = result
        
        # Save individual result to output dir
        result_path = self.output_dir / f"{subtask.id.replace('.', '_')}_{subtask.title.replace(' ', '_')[:30]}.txt"
        with open(result_path, "w") as f:
            f.write(f"# Sub-Task {subtask.id}: {subtask.title}\n")
            f.write(f"# Model: {subtask.model}\n")
            f.write(f"# Contribution: {subtask.contribution}\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(result)
        
        # Save state after each task execution (incremental progress)
        self.save_state()
        
        print(f"  ✓ Saved to: {result_path}")
        print(f"  ✓ State updated: {self.state_file}")
        
        return result

    def confirm_continue(self) -> bool:
        """Ask user for confirmation to continue execution."""
        print(f"\n{'─' * 70}")
        print(f"  CHECKPOINT: {self.tasks_executed} tasks completed")
        print(f"{'─' * 70}")
        try:
            response = input("  Continue execution? [Y/n]: ").strip().lower()
            if response in ('n', 'no', 'q', 'quit', 'exit'):
                return False
            return True
        except (KeyboardInterrupt, EOFError):
            return False

    def execute_all(self):
        """Execute all atomic sub-tasks with periodic human confirmation."""
        self.phase = "executing"
        self.save_state()
        
        print(f"\n{'═' * 70}")
        print("PHASE 2: ISOLATED SUB-TASK EXECUTION")
        print(f"{'═' * 70}")
        
        # Get all atomic tasks (leaves of the tree)
        execution_list = self._flatten_tasks(self.subtasks)
        
        print(f"\nExecuting {len(execution_list)} atomic sub-tasks...")
        print(f"(Will pause for confirmation every {TASKS_UNTIL_CONFIRM} tasks)")
        
        for i, subtask in enumerate(execution_list, 1):
            # Check if user has cancelled
            if self.user_cancelled:
                print("\n  Execution cancelled by user.")
                break
                
            # Human-in-the-loop checkpoint
            if self.tasks_executed > 0 and self.tasks_executed % TASKS_UNTIL_CONFIRM == 0:
                if not self.confirm_continue():
                    self.user_cancelled = True
                    print("\n  User chose to stop. Assembling partial results...")
                    break
            
            print(f"\n[{i}/{len(execution_list)}]", end="")
            self.execute_subtask(subtask)
            self.tasks_executed += 1

    def _format_results_recursive(self, tasks: list[SubTask], level: int = 0) -> str:
        """Recursively format results for the report."""
        output = ""
        indent = "  " * level
        
        for task in tasks:
            output += f"\n{'─' * (70-len(indent))}\n"
            output += f"{indent}SECTION {task.id}: {task.title.upper()}\n"
            
            if task.subtasks:
                output += f"{indent}(Composite Task - Decomposed into {len(task.subtasks)} parts)\n"
                output += self._format_results_recursive(task.subtasks, level + 1)
            else:
                output += f"{indent}Model: {task.model} | {task.contribution}\n"
                output += f"{'─' * (70-len(indent))}\n\n"
                # Indent the content slightly
                content = task.result or "[No result]"
                output += "\n".join(f"{indent}  {line}" for line in content.splitlines())
                output += "\n"
                
        return output

    def assemble(self):
        """Assemble all fragments into the final deliverable."""
        self.phase = "assembling"
        self.save_state()
        
        print(f"\n{'═' * 70}")
        print("PHASE 3: ASSEMBLY")
        print(f"{'═' * 70}")
        
        # Save manifest (recursive structure preserved in subtasks)
        def to_dict(obj):
            if isinstance(obj, SubTask):
                d = {k: v for k, v in obj.__dict__.items() if k != 'result'}
                d['result_length'] = len(obj.result) if obj.result else 0
                return d
            return obj

        manifest = {
            "goal": self.goal,
            "context": self.context,
            "timestamp": self.timestamp,
            "structure": [to_dict(t) for t in self.subtasks]
        }
        
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        
        # Assemble full report
        report_path = self.output_dir / "ASSEMBLED_REPORT.txt"
        with open(report_path, "w") as f:
            f.write("╔" + "═" * 68 + "╗\n")
            f.write(f"║{'ASSEMBLED REPORT':^68}║\n")
            f.write("╚" + "═" * 68 + "╝\n\n")
            f.write(f"GOAL: {self.goal}\n")
            f.write(f"GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            flat_count = len(self._flatten_tasks(self.subtasks))
            f.write(f"ATOMIC TASKS: {flat_count}\n")
            
            f.write("\n" + "═" * 70 + "\n")
            f.write("STRUCTURE\n")
            f.write("═" * 70 + "\n\n")
            
            def write_structure(tasks, level=0):
                for t in tasks:
                    indent = "  " * level
                    status = " (Composite)" if t.subtasks else f" ({t.model})"
                    f.write(f"{indent}{t.id}. {t.title}{status}\n")
                    if t.subtasks:
                        write_structure(t.subtasks, level + 1)
            
            write_structure(self.subtasks)
            
            f.write("\n" + "═" * 70 + "\n")
            f.write("CONTENT\n")
            f.write("═" * 70 + "\n")
            
            f.write(self._format_results_recursive(self.subtasks))
            
            f.write("\n\n" + "═" * 70 + "\n")
            f.write("END OF REPORT\n")
            f.write("═" * 70 + "\n")
        
        print(f"\n✓ Manifest saved to: {manifest_path}")
        print(f"✓ Full report saved to: {report_path}")
        
        return report_path
    
    def run(self):
        """Execute the full orchestration pipeline."""
        print("\n" + "╔" + "═" * 68 + "╗")
        print(f"║{'TASK ORCHESTRATOR':^68}║")
        print("╚" + "═" * 68 + "╝")
        
        # Phase 1: Plan
        if not self.plan():
            print("\n✗ Planning failed. Aborting.")
            return False
        
        # Display the plan
        print(f"\n{'─' * 70}")
        print("EXECUTION PLAN:")
        print(f"{'─' * 70}")
        for st in self.subtasks:
            print(f"\n  [{st.id}] {st.title}")
            print(f"      Model: {st.model}")
            print(f"      Mode: {st.mode or 'standard'}")
            print(f"      Contributes: {st.contribution}")
        
        # Phase 2: Execute
        self.execute_all()
        
        # Phase 3: Assemble
        report_path = self.assemble()
        
        # Mark complete
        self.phase = "complete"
        self.save_state()
        
        print("\n" + "╔" + "═" * 68 + "╗")
        print(f"║{'ORCHESTRATION COMPLETE':^68}║")
        print("╚" + "═" * 68 + "╝")
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Full report: {report_path}")
        print(f"State file: {self.state_file}")
        
        return True


def main():
    global MAX_DEPTH, MAX_TOTAL_TASKS, TASKS_UNTIL_CONFIRM
    
    parser = argparse.ArgumentParser(
        description="Orchestrate multi-step task execution with isolated sub-tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The orchestrator:
1. Uses a planner to decompose your goal into independent sub-tasks
2. Assigns each sub-task to the most appropriate model
3. Executes sub-tasks in isolation (no cross-contamination of results)
4. Assembles fragments into a cohesive final report
5. Saves state incrementally to temp/ for resumption

State files are saved to: temp/<run_id>/state.json

Examples:
  %(prog)s "Create a business plan for a mobile app startup"
  %(prog)s --max-depth 3 --max-tasks 50 "Design a curriculum for learning ML"
  %(prog)s --resume temp/My_Goal_20251128_120000/state.json
  %(prog)s --list-runs
"""
    )
    
    parser.add_argument("goal", nargs="?", help="The high-level goal or task to accomplish")
    parser.add_argument("-o", "--output", type=Path, help="Output directory")
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH,
                        help=f"Maximum decomposition depth (default: {MAX_DEPTH})")
    parser.add_argument("--max-tasks", type=int, default=MAX_TOTAL_TASKS,
                        help=f"Maximum total tasks (default: {MAX_TOTAL_TASKS})")
    parser.add_argument("--confirm-interval", type=int, default=TASKS_UNTIL_CONFIRM,
                        help=f"Tasks between confirmations (default: {TASKS_UNTIL_CONFIRM})")
    parser.add_argument("--no-confirm", action="store_true",
                        help="Disable human-in-the-loop confirmations")
    parser.add_argument("--resume", type=Path, metavar="STATE_FILE",
                        help="Resume from a previous state file")
    parser.add_argument("--list-runs", action="store_true",
                        help="List available runs in temp directory")
    
    args = parser.parse_args()
    
    # Handle --list-runs
    if args.list_runs:
        print("\nAvailable runs in temp directory:")
        print(f"{'─' * 70}")
        if TEMP_DIR.exists():
            runs = sorted(TEMP_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            for run_dir in runs:
                state_file = run_dir / "state.json"
                if state_file.exists():
                    try:
                        with open(state_file) as f:
                            state = json.load(f)
                        meta = state["meta"]
                        progress = state["progress"]
                        print(f"\n  {run_dir.name}")
                        print(f"    Goal: {meta['goal'][:50]}...")
                        print(f"    Phase: {meta['phase']}")
                        print(f"    Progress: {progress['tasks_executed']}/{progress['total_tasks_created']} tasks")
                        print(f"    Last updated: {meta['last_updated']}")
                    except:
                        print(f"\n  {run_dir.name} (corrupted state)")
            if not runs:
                print("  No runs found.")
        else:
            print("  Temp directory does not exist yet.")
        sys.exit(0)
    
    # Apply CLI overrides to global config
    MAX_DEPTH = args.max_depth
    MAX_TOTAL_TASKS = args.max_tasks
    TASKS_UNTIL_CONFIRM = args.confirm_interval if not args.no_confirm else float('inf')
    
    if not CLI_PATH.exists():
        print(f"Error: CLI not found at {CLI_PATH}")
        sys.exit(1)
    
    # Handle --resume
    if args.resume:
        if not args.resume.exists():
            print(f"Error: State file not found: {args.resume}")
            sys.exit(1)
        
        print(f"\nResuming from: {args.resume}")
        orchestrator = TaskOrchestrator.load_state(args.resume)
        if not orchestrator:
            print("Error: Failed to load state file")
            sys.exit(1)
        
        print(f"  Goal: {orchestrator.goal}")
        print(f"  Phase: {orchestrator.phase}")
        print(f"  Progress: {orchestrator.tasks_executed}/{orchestrator.total_tasks_created} tasks")
        
        # Resume from appropriate phase
        if orchestrator.phase == "planning":
            print("\n  Resuming from planning phase...")
            success = orchestrator.run()
        elif orchestrator.phase == "executing":
            print("\n  Resuming from execution phase...")
            orchestrator.execute_all()
            orchestrator.assemble()
            orchestrator.phase = "complete"
            orchestrator.save_state()
            success = True
        elif orchestrator.phase == "assembling":
            print("\n  Resuming from assembly phase...")
            orchestrator.assemble()
            orchestrator.phase = "complete"
            orchestrator.save_state()
            success = True
        elif orchestrator.phase == "complete":
            print("\n  Run already complete. Nothing to resume.")
            success = True
        else:
            print(f"\n  Unknown phase: {orchestrator.phase}. Starting fresh.")
            success = orchestrator.run()
        
        sys.exit(0 if success else 1)
    
    # Require goal for new runs
    if not args.goal:
        parser.error("the following arguments are required: goal (or use --resume)")
    
    # Display configuration
    print(f"\n  Config: max_depth={MAX_DEPTH}, max_tasks={MAX_TOTAL_TASKS}, ", end="")
    if args.no_confirm:
        print("confirmations=disabled")
    else:
        print(f"confirm_every={TASKS_UNTIL_CONFIRM}")
    
    orchestrator = TaskOrchestrator(args.goal, args.output)
    print(f"  State file: {orchestrator.state_file}")
    
    success = orchestrator.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

