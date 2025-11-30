"""
Perplexity AI CLI - Browser automation + Local API

Automates the Perplexity web UI via browser and exposes it as:
  - CLI interface for direct queries
  - HTTP API for integration with other projects
  - TaskOrchestrator for multi-step task execution

Usage:
    from perplexity_ai_cli import PerplexityBrowser, TaskOrchestrator

    # Simple query
    async with PerplexityBrowser() as browser:
        response = await browser.ask("What is quantum computing?")
        print(response.answer)

    # Orchestrated task
    orchestrator = TaskOrchestrator("Create a business plan", browser=browser)
    await orchestrator.run_async()
"""

from .core import (
    PerplexityBrowser,
    PerplexityResponse,
    TaskOrchestrator,
    SubTask,
    AVAILABLE_MODELS,
    __version__,
)

__all__ = [
    "PerplexityBrowser",
    "PerplexityResponse",
    "TaskOrchestrator",
    "SubTask",
    "AVAILABLE_MODELS",
    "__version__",
]

