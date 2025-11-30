# Perplexity AI CLI

Browser automation bridge to Perplexity AI with CLI, HTTP API, and Task Orchestration.

<div align="center"> <img src="ppl-ai.gif" width=500> </div>

## Features

- 🌐 **Browser Automation** - Automates the Perplexity web UI via Playwright
- 💬 **CLI Interface** - Ask questions directly from the terminal
- 🔌 **HTTP API** - RESTful API for integration with other projects
- 🤖 **Task Orchestrator** - Multi-step task execution with sub-task decomposition
- 🔄 **Model Selection** - Choose from GPT, Claude, Gemini, Grok, and more
- 🔬 **Research Mode** - Deep, multi-step research capabilities
- 🧪 **Labs Mode** - Access experimental features

## Installation

### From GitHub (recommended for development)

```bash
# Clone the repository
git clone https://github.com/Cfomodz/perplexity.ai-cli.git
cd perplexity.ai-cli

# Install as editable package
pip install -e .

# Install Playwright browsers
playwright install chromium
```

### As a dependency in other projects

```bash
# In your project's requirements.txt or pip install:
pip install git+https://github.com/Cfomodz/perplexity.ai-cli.git
```

## Prerequisites

1. **Perplexity Pro Account** - Required for model selection and advanced features
2. **Browser with CDP** (recommended) - For using your existing logged-in session:
   ```bash
   brave --remote-debugging-port=9222
   # or
   google-chrome --remote-debugging-port=9222
   ```

## Quick Start

### First-time Login

```bash
# Login and save session (one-time setup)
perplexity-ai --login
# or
ppl --login
```

### CLI Usage

```bash
# Ask a question
ppl "What is quantum computing?"

# Use a specific model
ppl --model claude "Explain relativity"

# Research mode (deep, multi-step)
ppl --research "History of quantum computing"

# Labs mode (experimental features)
ppl --labs "Build me a chart"

# Interactive mode
ppl -i

# Connect to existing browser (uses your logged-in session)
ppl --cdp "Your question"
```

### HTTP API

```bash
# Start the API server
ppl --serve

# Query via HTTP
curl "http://localhost:8000/ask?q=What+is+AI"
curl "http://localhost:8000/ask?q=Topic&model=claude"
curl "http://localhost:8000/ask?q=Deep+topic&research=true"
```

### Task Orchestrator

```bash
# Run orchestrated multi-step task
ppl --orchestrate "Create a business plan for a SaaS startup"

# List previous runs
ppl --orchestrate-list

# Resume a previous run
ppl --orchestrate-resume path/to/state.json
```

## Programmatic Usage

```python
import asyncio
from perplexity_ai_cli import PerplexityBrowser, TaskOrchestrator, AVAILABLE_MODELS

async def main():
    # Simple query
    async with PerplexityBrowser(cdp_url="http://localhost:9222") as browser:
        response = await browser.ask(
            "What is quantum computing?",
            model="claude",
            research_mode=False,
        )
        print(response.answer)
        print(response.references)

    # Task orchestration
    async with PerplexityBrowser() as browser:
        orchestrator = TaskOrchestrator(
            goal="Create a marketing plan",
            browser=browser,
            auto_confirm=True,
        )
        await orchestrator.run_async()

asyncio.run(main())
```

## Available Models

| Key | Model |
|-----|-------|
| `auto` / `best` | Best (auto-select) |
| `sonar` | Sonar |
| `gpt` / `gpt-5` | GPT-5.1 |
| `o3-pro` | o3-pro |
| `claude` / `claude-sonnet` | Claude Sonnet 4.5 |
| `claude-opus` | Claude Opus 4.5 |
| `gemini` / `gemini-pro` | Gemini 3 Pro |
| `grok` | Grok 4.1 |
| `kimi` | Kimi K2 Thinking |

## Project Structure

```
perplexity.ai-cli/
├── perplexity_ai_cli/          # Python package
│   ├── __init__.py             # Package exports
│   └── core.py                 # Main implementation
├── perplexity.ai-cli.py        # Standalone script (legacy)
├── examples/                   # Usage examples
├── pyproject.toml              # Package configuration
└── requirements.txt            # Dependencies
```

## License

MIT License

## Credits

Originally inspired by [Helpingai](https://github.com/HelpingAI/Helpingai_T2).
