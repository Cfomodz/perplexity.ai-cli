#!/bin/bash
#
# Research Workflow Script
# 
# This script demonstrates using the Perplexity CLI to accomplish a
# multi-step research and analysis process.
#
# Usage: ./research-workflow.sh "Your research topic"
#
# Example: ./research-workflow.sh "Impact of AI on healthcare in 2025"
#

set -e

# Configuration
CLI_PATH="${CLI_PATH:-$(dirname "$0")/../perplexity.ai-cli.py}"
VENV_PYTHON="${VENV_PYTHON:-$(dirname "$0")/../.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-./research-output}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check arguments
if [ -z "$1" ]; then
    echo -e "${RED}Error: Please provide a research topic${NC}"
    echo "Usage: $0 \"Your research topic\""
    echo "Example: $0 \"Impact of AI on healthcare in 2025\""
    exit 1
fi

TOPIC="$1"
SAFE_TOPIC=$(echo "$TOPIC" | tr ' ' '_' | tr -cd '[:alnum:]_-')
RESEARCH_DIR="${OUTPUT_DIR}/${SAFE_TOPIC}_${TIMESTAMP}"

# Create output directory
mkdir -p "$RESEARCH_DIR"

echo -e "${PURPLE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║          PERPLEXITY MULTI-STEP RESEARCH WORKFLOW              ║${NC}"
echo -e "${PURPLE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Topic:${NC} $TOPIC"
echo -e "${CYAN}Output:${NC} $RESEARCH_DIR"
echo ""

# Helper function to run CLI
run_cli() {
    local args="$1"
    local output_file="$2"
    
    echo -e "${YELLOW}Running:${NC} perplexity.ai-cli.py $args"
    
    if [ -n "$output_file" ]; then
        $VENV_PYTHON "$CLI_PATH" --no-typing $args 2>&1 | tee "$output_file"
    else
        $VENV_PYTHON "$CLI_PATH" --no-typing $args 2>&1
    fi
    
    # Small delay between requests to avoid rate limiting
    sleep 2
}

# ═══════════════════════════════════════════════════════════════════
# STEP 1: Deep Research
# ═══════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}STEP 1: Deep Research Analysis${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

run_cli "--research \"Provide a comprehensive analysis of: $TOPIC. Include current state, key players, recent developments, challenges, and future outlook.\"" \
    "$RESEARCH_DIR/01_deep_research.txt"

# ═══════════════════════════════════════════════════════════════════
# STEP 2: Get Multiple Model Perspectives
# ═══════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}STEP 2: Multi-Model Perspectives${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${BLUE}Getting Claude's perspective...${NC}"
run_cli "-m claude \"Regarding '$TOPIC': What are the most significant ethical considerations and potential societal impacts? Be specific and analytical.\"" \
    "$RESEARCH_DIR/02a_claude_perspective.txt"

echo ""
echo -e "${BLUE}Getting GPT's perspective...${NC}"
run_cli "-m gpt \"Regarding '$TOPIC': What are the key technical innovations and breakthroughs driving progress in this area? Focus on concrete examples.\"" \
    "$RESEARCH_DIR/02b_gpt_perspective.txt"

echo ""
echo -e "${BLUE}Getting Gemini's perspective...${NC}"
run_cli "-m gemini \"Regarding '$TOPIC': What are the economic implications and market opportunities? Include data points where possible.\"" \
    "$RESEARCH_DIR/02c_gemini_perspective.txt"

# ═══════════════════════════════════════════════════════════════════
# STEP 3: Academic Focus
# ═══════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}STEP 3: Academic Research${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

run_cli "-f academic \"What are the most cited recent academic papers and research studies about: $TOPIC? Summarize their key findings.\"" \
    "$RESEARCH_DIR/03_academic_research.txt"

# ═══════════════════════════════════════════════════════════════════
# STEP 4: Practical Applications (Labs Mode)
# ═══════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}STEP 4: Generate Visual Summary (Labs Mode)${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

run_cli "--labs \"Create a timeline chart showing the major milestones and developments in: $TOPIC. Include dates and brief descriptions.\"" \
    "$RESEARCH_DIR/04_labs_timeline.txt"

# ═══════════════════════════════════════════════════════════════════
# STEP 5: Critical Analysis
# ═══════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}STEP 5: Critical Analysis & Contrarian Views${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

run_cli "-m grok \"Play devil's advocate on '$TOPIC'. What are the strongest arguments against the mainstream narrative? What risks are being overlooked?\"" \
    "$RESEARCH_DIR/05_critical_analysis.txt"

# ═══════════════════════════════════════════════════════════════════
# STEP 6: Executive Summary
# ═══════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}STEP 6: Generate Executive Summary${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Combine key points for final summary
run_cli "--research \"Create an executive summary about '$TOPIC' that includes: 1) Key findings and current state, 2) Main opportunities and challenges, 3) Different expert perspectives, 4) Actionable recommendations. Format it professionally with clear sections.\"" \
    "$RESEARCH_DIR/06_executive_summary.txt"

# ═══════════════════════════════════════════════════════════════════
# COMPLETION
# ═══════════════════════════════════════════════════════════════════
echo ""
echo -e "${PURPLE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║                    RESEARCH COMPLETE                           ║${NC}"
echo -e "${PURPLE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}All research files saved to:${NC} $RESEARCH_DIR"
echo ""
echo "Generated files:"
ls -la "$RESEARCH_DIR"
echo ""
echo -e "${CYAN}You can view the combined research with:${NC}"
echo "  cat $RESEARCH_DIR/*.txt"
echo ""
echo -e "${CYAN}Or create a combined report:${NC}"
echo "  cat $RESEARCH_DIR/*.txt > $RESEARCH_DIR/FULL_REPORT.txt"

