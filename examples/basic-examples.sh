#!/bin/bash
#
# Basic Examples - Quick reference for common CLI usage patterns
#

CLI="../perplexity.ai-cli.py"
PYTHON="../.venv/bin/python"

echo "═══════════════════════════════════════════════════════════════"
echo "  Perplexity CLI - Basic Usage Examples"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ─────────────────────────────────────────────────────────────────
# Example 1: Simple query
# ─────────────────────────────────────────────────────────────────
example_simple() {
    echo "Example 1: Simple Query"
    echo "─────────────────────────────────────────────────────────────"
    $PYTHON $CLI --no-typing "What is the capital of France?"
}

# ─────────────────────────────────────────────────────────────────
# Example 2: Research mode for deep analysis
# ─────────────────────────────────────────────────────────────────
example_research() {
    echo "Example 2: Research Mode (Deep Analysis)"
    echo "─────────────────────────────────────────────────────────────"
    $PYTHON $CLI --no-typing --research "Explain the current state of quantum computing"
}

# ─────────────────────────────────────────────────────────────────
# Example 3: Labs mode for creative tasks
# ─────────────────────────────────────────────────────────────────
example_labs() {
    echo "Example 3: Labs Mode (Creative/Visual)"
    echo "─────────────────────────────────────────────────────────────"
    $PYTHON $CLI --no-typing --labs "Create a comparison table of major cloud providers"
}

# ─────────────────────────────────────────────────────────────────
# Example 4: Specific model selection
# ─────────────────────────────────────────────────────────────────
example_models() {
    echo "Example 4: Model Selection"
    echo "─────────────────────────────────────────────────────────────"
    
    echo ""
    echo "Using Claude:"
    $PYTHON $CLI --no-typing -m claude "Write a haiku about programming"
    
    echo ""
    echo "Using GPT:"
    $PYTHON $CLI --no-typing -m gpt "Explain recursion simply"
    
    echo ""
    echo "Using Grok:"
    $PYTHON $CLI --no-typing -m grok "What's a hot take on cryptocurrency?"
}

# ─────────────────────────────────────────────────────────────────
# Example 5: Focus modes
# ─────────────────────────────────────────────────────────────────
example_focus() {
    echo "Example 5: Focus Modes"
    echo "─────────────────────────────────────────────────────────────"
    
    echo ""
    echo "Academic focus:"
    $PYTHON $CLI --no-typing -f academic "Recent papers on transformer architectures"
    
    echo ""
    echo "Reddit focus:"
    $PYTHON $CLI --no-typing -f reddit "Best mechanical keyboards for programming"
    
    echo ""
    echo "YouTube focus:"
    $PYTHON $CLI --no-typing -f youtube "Best tutorials for learning Rust"
}

# ─────────────────────────────────────────────────────────────────
# Example 6: Combining options
# ─────────────────────────────────────────────────────────────────
example_combined() {
    echo "Example 6: Combined Options"
    echo "─────────────────────────────────────────────────────────────"
    
    echo ""
    echo "Research mode + Claude + Academic focus:"
    $PYTHON $CLI --no-typing --research -m claude -f academic \
        "Latest breakthroughs in CRISPR gene editing"
}

# ─────────────────────────────────────────────────────────────────
# Example 7: Piping output
# ─────────────────────────────────────────────────────────────────
example_piping() {
    echo "Example 7: Piping Output to Files"
    echo "─────────────────────────────────────────────────────────────"
    
    $PYTHON $CLI --no-typing "List 5 interesting facts about space" > space_facts.txt
    echo "Output saved to space_facts.txt"
    cat space_facts.txt
    rm space_facts.txt
}

# ─────────────────────────────────────────────────────────────────
# Run selected example or show menu
# ─────────────────────────────────────────────────────────────────
if [ -n "$1" ]; then
    case "$1" in
        1|simple) example_simple ;;
        2|research) example_research ;;
        3|labs) example_labs ;;
        4|models) example_models ;;
        5|focus) example_focus ;;
        6|combined) example_combined ;;
        7|piping) example_piping ;;
        all)
            example_simple
            echo ""; example_research
            echo ""; example_labs
            echo ""; example_models
            echo ""; example_focus
            echo ""; example_combined
            echo ""; example_piping
            ;;
        *)
            echo "Unknown example: $1"
            echo "Usage: $0 [1-7|all]"
            ;;
    esac
else
    echo "Usage: $0 [example_number]"
    echo ""
    echo "Available examples:"
    echo "  1, simple    - Basic query"
    echo "  2, research  - Research mode"
    echo "  3, labs      - Labs mode"
    echo "  4, models    - Model selection"
    echo "  5, focus     - Focus modes"
    echo "  6, combined  - Combined options"
    echo "  7, piping    - Output to files"
    echo "  all          - Run all examples"
    echo ""
    echo "Example: $0 2"
fi

