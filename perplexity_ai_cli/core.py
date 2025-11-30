#!/usr/bin/env python

"""
Perplexity AI Bridge - Browser automation + Local API

Automates the Perplexity web UI via browser and exposes it as:
  - CLI interface for direct queries
  - HTTP API for integration with other projects
  - TaskOrchestrator for multi-step task execution with isolated sub-tasks

No reverse engineering, no token management - just real browser automation.

Usage as a module:
    from perplexity_ai_cli import PerplexityBrowser, TaskOrchestrator, PerplexityResponse
    
    # Simple query
    async with PerplexityBrowser() as browser:
        response = await browser.ask("What is quantum computing?")
        print(response.answer)
    
    # Orchestrated task
    orchestrator = TaskOrchestrator("Create a business plan", browser=browser)
    await orchestrator.run_async()
"""

__version__ = "1.1.0"

__all__ = [
    "PerplexityBrowser",
    "PerplexityResponse", 
    "TaskOrchestrator",
    "SubTask",
    "AVAILABLE_MODELS",
]

import asyncio
import json
import os
import sys
import re
from time import sleep
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Browser automation
try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
except ImportError:
    print("Playwright not installed. Run: pip install playwright && playwright install chromium")
    sys.exit(1)

# HTTP API
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    FastAPI = None  # HTTP server optional


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class PerplexityResponse:
    """Response from Perplexity."""
    answer: str
    references: List[Dict[str, str]]
    query: str
    model: Optional[str] = None
    raw_html: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SubTask:
    """A decomposed sub-task for the orchestrator."""
    id: str  # Hierarchical ID (e.g., "1.2")
    title: str
    model: str
    prompt: str
    contribution: str
    is_atomic: bool = True
    focus: Optional[str] = None
    mode: Optional[str] = None
    result: Optional[str] = None
    subtasks: List['SubTask'] = field(default_factory=list)


# Available models in Perplexity Pro mode
# These names must match EXACTLY what appears in the UI dropdown
AVAILABLE_MODELS = {
    # Default (auto-select best model)
    "auto": "Best",
    "best": "Best",
    # Perplexity's own model
    "sonar": "Sonar",
    # OpenAI models
    "gpt": "GPT-5.1",
    "gpt-5": "GPT-5.1",
    "gpt-5.1": "GPT-5.1",
    "o3-pro": "o3-pro",
    # Anthropic models
    "claude": "Claude Sonnet 4.5",
    "claude-sonnet": "Claude Sonnet 4.5",
    "claude-opus": "Claude Opus 4.5",
    # Google models
    "gemini": "Gemini 3 Pro",
    "gemini-pro": "Gemini 3 Pro",
    # xAI
    "grok": "Grok 4.1",
    # Moonshot
    "kimi": "Kimi K2 Thinking",
}


# ---------------------------------------------------------------------------
# Browser Automation
# ---------------------------------------------------------------------------

# Common browser profile locations
BROWSER_PROFILES = {
    "brave-flatpak": Path.home() / ".var/app/com.brave.Browser/config/BraveSoftware/Brave-Browser",
    "brave": Path.home() / ".config/BraveSoftware/Brave-Browser",
    "chrome-flatpak": Path.home() / ".var/app/com.google.Chrome/config/google-chrome",
    "chrome": Path.home() / ".config/google-chrome",
    "chromium": Path.home() / ".config/chromium",
}

CLONED_PROFILE_DIR = Path.home() / ".perplexity-cli" / "browser-profile"


def find_browser_profile() -> Optional[Path]:
    """Find an existing browser profile to use."""
    for name, path in BROWSER_PROFILES.items():
        if path.exists() and (path / "Default").exists():
            return path
    return None


def clone_browser_profile(source: Path, dest: Path) -> bool:
    """
    Clone essential browser profile data (cookies, storage) to a new location.
    This allows using the profile while the original browser is running.
    """
    import shutil
    
    dest.mkdir(parents=True, exist_ok=True)
    
    # Files/folders to copy for session data
    items_to_copy = [
        "Default/Cookies",
        "Default/Login Data", 
        "Default/Web Data",
        "Default/Local Storage",
        "Default/Session Storage",
        "Default/IndexedDB",
        "Default/Preferences",
        "Default/Secure Preferences",
        "Local State",
    ]
    
    # Ensure Default directory exists
    (dest / "Default").mkdir(parents=True, exist_ok=True)
    
    copied = 0
    for item in items_to_copy:
        src_path = source / item
        dst_path = dest / item
        
        if src_path.exists():
            try:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                if src_path.is_dir():
                    if dst_path.exists():
                        shutil.rmtree(dst_path)
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
                copied += 1
            except Exception as e:
                print(f"  Warning: Could not copy {item}: {e}")
    
    return copied > 0


class PerplexityBrowser:
    """
    Automates Perplexity AI via browser.
    
    Can either:
    1. Connect to a running browser via CDP (recommended - uses existing session)
    2. Launch a new browser with its own profile
    
    Usage as async context manager:
        async with PerplexityBrowser() as browser:
            response = await browser.ask("What is AI?")
    """
    
    PERPLEXITY_URL = "https://www.perplexity.ai"
    
    def __init__(
        self, 
        cdp_url: Optional[str] = None,
        profile_path: Optional[str] = None,
    ):
        """
        Initialize browser automation.
        
        Args:
            cdp_url: Chrome DevTools Protocol URL to connect to running browser.
                    e.g., "http://localhost:9222"
            profile_path: Path to browser profile (used if cdp_url not provided).
        """
        self.cdp_url = cdp_url
        self.profile_path = Path(profile_path) if profile_path else CLONED_PROFILE_DIR
        
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._initialized = False
        self._owns_browser = True  # Whether we launched the browser (vs connected)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        return False
    
    async def start(self):
        """Start or connect to browser and navigate to Perplexity."""
        if self._initialized:
            return
        
        self._playwright = await async_playwright().start()
        
        if self.cdp_url:
            # Connect to existing browser via CDP
            print(f"  Connecting to browser at: {self.cdp_url}")
            try:
                self._browser = await self._playwright.chromium.connect_over_cdp(self.cdp_url)
                self._context = self._browser.contexts[0] if self._browser.contexts else await self._browser.new_context()
                self._owns_browser = False
                print("  Connected to existing browser!")
            except Exception as e:
                print(f"  Could not connect to CDP: {e}")
                print("  Falling back to launching new browser...")
                self.cdp_url = None
        
        if not self.cdp_url:
            # Launch new browser with profile
            self.profile_path.mkdir(parents=True, exist_ok=True)
            print(f"  Launching browser with profile: {self.profile_path}")
            
            self._context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.profile_path),
                headless=False,
                viewport={"width": 1280, "height": 900},
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-first-run",
                    "--no-default-browser-check",
                ],
            )
            self._owns_browser = True
        
        # Get or create page
        if self._context.pages:
            self._page = self._context.pages[0]
        else:
            self._page = await self._context.new_page()
        
        # Navigate to Perplexity
        print("  Navigating to Perplexity...")
        await self._page.goto(self.PERPLEXITY_URL, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(3)
        
        self._initialized = True
    
    async def stop(self):
        """Close or disconnect from browser."""
        if self._owns_browser:
            # We launched it, so close it
            if self._context:
                await self._context.close()
        else:
            # We connected to it, just close our page
            if self._page:
                await self._page.close()
        
        if self._playwright:
            await self._playwright.stop()
        self._initialized = False
    
    async def is_logged_in(self) -> bool:
        """Check if user is logged in."""
        if not self._initialized:
            await self.start()
        
        try:
            # Give page time to render
            await asyncio.sleep(1)
            
            # Check URL for login redirect
            current_url = self._page.url
            if "login" in current_url or "signin" in current_url or "auth" in current_url:
                return False
            
            # DEFINITIVE: Look for the signed-in sidebar trigger
            # This element only appears when logged in
            signed_in_indicator = await self._page.query_selector('[data-testid="sidebar-popover-trigger-signed-in"]')
            if signed_in_indicator:
                return True
            
            # Also check for user menu indicators
            logged_in_selectors = [
                '[data-testid="user-menu"]',
                '[data-testid="profile-button"]',
                '[data-testid="sidebar-popover-trigger-signed-in"]',
            ]
            
            for selector in logged_in_selectors:
                try:
                    element = await self._page.query_selector(selector)
                    if element:
                        return True
                except:
                    continue
            
            # If none of the logged-in indicators found, we're not logged in
            return False
            
        except Exception as e:
            print(f"Login check error: {e}")
            return False
    
    async def login_interactive(self):
        """
        Open browser for interactive login.
        
        Opens a visible browser window for the user to log in manually.
        Waits patiently for Google 2FA and other auth flows.
        """
        print("\n" + "="*60)
        print("PERPLEXITY LOGIN")
        print("="*60)
        print("\nOpening browser window...")
        print("Please log in to your Perplexity account.")
        print("\nTake your time with 2FA - the browser will wait.")
        print("Press Ctrl+C here if you need to cancel.\n")
        
        # Start browser if not already
        if not self._initialized:
            await self.start()
        
        # Navigate to Perplexity
        try:
            await self._page.goto("https://www.perplexity.ai/", wait_until="domcontentloaded", timeout=60000)
        except:
            pass
        
        await asyncio.sleep(3)
        
        # Wait for login with patient status updates
        check_count = 0
        last_url = ""
        while True:
            check_count += 1
            
            # Show current status
            current_url = self._page.url
            if current_url != last_url:
                if "accounts.google" in current_url:
                    print("  📧 Google sign-in detected - take your time with 2FA...")
                elif "login" in current_url or "signin" in current_url:
                    print("  🔐 On login page...")
                elif "perplexity.ai" in current_url and "login" not in current_url:
                    print("  🌐 On Perplexity - checking login status...")
                last_url = current_url
            
            is_logged = await self.is_logged_in()
            
            if is_logged:
                print("\n✓ Login detected! Verifying...")
                await asyncio.sleep(3)
                # Double check after page settles
                if await self.is_logged_in():
                    break
            
            if check_count % 10 == 0:
                mins = (check_count * 2) // 60
                secs = (check_count * 2) % 60
                print(f"  ⏳ Still waiting... ({mins}m {secs}s)")
            
            await asyncio.sleep(2)
        
        print("\n" + "="*60)
        print("✓ LOGIN SUCCESSFUL!")
        print("="*60)
        print("\nYour session has been saved.")
        print("You can now use the CLI.\n")
    
    async def ask(
        self,
        query: str,
        focus: str = "internet",
        pro_mode: bool = False,
        model: Optional[str] = None,
        research_mode: bool = False,
        labs_mode: bool = False,
        timeout: int = 60,
        use_paste: bool = False,
    ) -> PerplexityResponse:
        """
        Ask a question and get the response.
        
        Args:
            query: The question to ask.
            focus: Search focus ('internet', 'academic', 'writing', 'wolfram', 'youtube', 'reddit').
            pro_mode: Use Pro/Copilot mode if available.
            model: Specific model to use (requires Pro). See AVAILABLE_MODELS.
            research_mode: Use Research mode for deep, multi-step research.
            labs_mode: Use Labs mode for experimental features.
            timeout: Maximum seconds to wait for response.
            use_paste: Use clipboard paste instead of typing (faster, preserves newlines).
        
        Returns:
            PerplexityResponse with answer and references.
        """
        await self.start()
        
        # Navigate to Perplexity
        await self._page.goto(self.PERPLEXITY_URL, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(3)
        
        # Select search mode (Search / Research / Labs) - this should be visible immediately
        if labs_mode:
            await self._select_search_mode("studio")  # Labs is called "studio" internally
        elif research_mode:
            await self._select_search_mode("research")
        # else: default is "search" mode, already selected
        
        # Find the input - Perplexity uses a contenteditable div, not a textarea
        # The main input has id="ask-input" and role="textbox"
        input_selectors = [
            '#ask-input',  # Primary: the contenteditable div
            '[role="textbox"][contenteditable="true"]',
            '[data-lexical-editor="true"]',
            'textarea[placeholder*="Ask"]',
            'textarea[placeholder*="ask"]',
            '[data-testid="query-input"]',
            'textarea',
        ]
        
        input_element = None
        for selector in input_selectors:
            try:
                await self._page.wait_for_selector(selector, timeout=5000)
                input_element = await self._page.query_selector(selector)
                if input_element:
                    print(f"  Found input with selector: {selector}")
                    break
            except:
                continue
        
        if not input_element:
            raise RuntimeError("Could not find input element. Page may not have loaded correctly.")
        
        # Click to focus the input (this makes the toolbar with model selector visible)
        await input_element.click()
        await asyncio.sleep(0.5)
        
        # Now select model/focus/pro mode (toolbar should be visible after input click)
        if model and model != "auto":
            await self._select_model(model)
        
        if pro_mode and not model:
            await self._enable_pro_mode()
        
        if focus and focus != "internet":
            await self._select_focus(focus)
        
        # Click input again to ensure focus for typing
        await input_element.click()
        await asyncio.sleep(0.2)
        
        if use_paste:
            # Use clipboard paste - faster and preserves newlines
            await self._page.evaluate(f"navigator.clipboard.writeText({json.dumps(query)})")
            await asyncio.sleep(0.1)
            # Ctrl+V to paste
            await self._page.keyboard.press("Control+v")
            await asyncio.sleep(0.3)
        else:
            # For contenteditable divs, we need to type character by character
            # Using fill() doesn't work well with Lexical editor
            await self._page.keyboard.type(query, delay=20)
        
        await asyncio.sleep(0.5)
        
        # Submit with Enter
        await self._page.keyboard.press("Enter")
        
        mode_str = "Labs mode" if labs_mode else "Research mode" if research_mode else f"Model: {model}" if model else "Standard"
        print(f"  Query submitted ({mode_str}), waiting for response...")
        
        # Research/Labs mode takes longer - adjust timeout
        effective_timeout = timeout * 3 if (research_mode or labs_mode) else timeout
        
        # Wait for response to complete
        response_text = await self._wait_for_response(effective_timeout)
        references = await self._extract_references()
        
        return PerplexityResponse(
            answer=response_text,
            references=references,
            query=query,
            model=model,
        )
    
    async def _select_search_mode(self, mode: str):
        """
        Select the search mode: 'search', 'research', or 'studio' (labs).
        
        Uses the segmented control with data-testid="search-mode-{mode}".
        """
        mode_names = {"search": "Search", "research": "Research", "studio": "Labs"}
        print(f"  Selecting mode: {mode_names.get(mode, mode)}")
        
        # The mode buttons have data-testid="search-mode-{mode}" inside them
        # The clickable button is the parent with role="radio"
        selectors = [
            # Direct testid on the inner div
            f'[data-testid="search-mode-{mode}"]',
            # Parent button with the value attribute
            f'button[value="{mode}"]',
            # Parent button with aria-label
            f'button[aria-label="{mode_names.get(mode, mode)}"]',
        ]
        
        for selector in selectors:
            try:
                element = await self._page.query_selector(selector)
                if element:
                    # If we got the inner div, get the parent button
                    tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
                    if tag_name != "button":
                        # Click the parent button instead
                        parent = await element.evaluate_handle("el => el.closest('button')")
                        if parent:
                            await parent.click()
                            print(f"    Mode selected via parent: {mode_names.get(mode, mode)}")
                            await asyncio.sleep(0.5)
                            return
                    else:
                        await element.click()
                        print(f"    Mode selected: {mode_names.get(mode, mode)}")
                        await asyncio.sleep(0.5)
                        return
            except Exception as e:
                continue
        
        print(f"    Warning: Could not find mode selector for '{mode}'")
    
    async def _select_model(self, model: str):
        """Select a specific model from the model selector dropdown."""
        model_display_name = AVAILABLE_MODELS.get(model, model)
        print(f"  Selecting model: {model_display_name}")
        
        # Model selector button - the aria-label shows the currently selected model name
        # So we need to look for buttons with known model names in aria-label
        known_model_labels = [
            "Best", "Sonar", "GPT", "Claude", "Gemini", "Grok", "Kimi", "o3",
            "Choose a model",  # fallback if no model selected
        ]
        
        model_button = None
        try:
            buttons = await self._page.query_selector_all('button[aria-label]')
            for btn in buttons:
                label = await btn.get_attribute('aria-label')
                if label:
                    # Check if this button's label matches any known model name
                    for model_label in known_model_labels:
                        if model_label in label:
                            model_button = btn
                            print(f"    Found model button with label: {label}")
                            break
                    if model_button:
                        break
        except Exception as e:
            print(f"    Error finding model button: {e}")
        
        if not model_button:
            print(f"    Warning: Could not find model selector. Using default model.")
            return
        
        # Click to open dropdown
        await model_button.click()
        await asyncio.sleep(0.5)
        
        # Look for the model option in the dropdown
        # The dropdown has role="menu" with role="menuitem" children
        # Each menuitem contains a span with the model name
        model_option = None
        
        # First, try to find by exact text match in menuitem
        try:
            # Get all menuitems
            menuitems = await self._page.query_selector_all('[role="menuitem"]')
            for item in menuitems:
                # Get the text content of the span inside
                text = await item.inner_text()
                # Clean up the text (remove "new", "max" badges, etc.)
                text_clean = text.split('\n')[0].strip()
                if text_clean == model_display_name:
                    model_option = item
                    print(f"    Found model option: {model_display_name}")
                    break
        except Exception as e:
            print(f"    Error searching menuitems: {e}")
        
        # Fallback selectors if direct search failed
        if not model_option:
            fallback_selectors = [
                f'[role="menuitem"]:has-text("{model_display_name}")',
                f'[role="menu"] span:text-is("{model_display_name}")',
            ]
            for selector in fallback_selectors:
                try:
                    model_option = await self._page.query_selector(selector)
                    if model_option:
                        print(f"    Found model option via fallback: {selector}")
                        break
                except:
                    continue
        
        if model_option:
            await model_option.click()
            await asyncio.sleep(0.3)
            print(f"    Model selected: {model_display_name}")
        else:
            print(f"    Warning: Could not find model option for '{model_display_name}'")
            # Press Escape to close dropdown
            await self._page.keyboard.press("Escape")
    
    async def _enable_pro_mode(self):
        """Enable Pro/Copilot mode if available."""
        pro_selectors = [
            '[data-testid="pro-toggle"]',
            '[data-testid="copilot-toggle"]',
            'button[aria-label*="Pro"]',
            'button[aria-label*="Copilot"]',
            '[class*="pro-toggle"]',
            'input[type="checkbox"][name*="pro"]',
        ]
        
        for selector in pro_selectors:
            try:
                toggle = await self._page.query_selector(selector)
                if toggle:
                    # Check if it's already enabled
                    is_checked = await toggle.get_attribute("aria-checked")
                    if is_checked != "true":
                        await toggle.click()
                        print("  Pro mode enabled")
                    return
            except:
                continue
        
        print("  Note: Pro mode toggle not found (may already be enabled or unavailable)")
    
    async def _select_focus(self, focus: str):
        """Select a specific focus mode."""
        focus_selectors = [
            f'[data-testid="focus-{focus}"]',
            f'button[aria-label*="{focus}"]',
            f'[class*="focus"][class*="{focus}"]',
        ]
        
        for selector in focus_selectors:
            try:
                focus_btn = await self._page.query_selector(selector)
                if focus_btn:
                    await focus_btn.click()
                    print(f"  Focus mode set: {focus}")
                    await asyncio.sleep(0.3)
                    return
            except:
                continue
        
        print(f"  Note: Focus mode '{focus}' selector not found")
    
    async def _wait_for_response(self, timeout: int) -> str:
        """Wait for response to complete and extract text."""
        # Wait for response to start appearing
        await asyncio.sleep(4)
        
        # Poll for completion (look for streaming to stop)
        last_text = ""
        stable_count = 0
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            current_text = await self._extract_response_text()
            
            if current_text == last_text and current_text:
                stable_count += 1
                if stable_count >= 3:  # Text stable for 3 checks
                    print(f"  Response complete ({len(current_text)} chars)")
                    break
            else:
                stable_count = 0
                last_text = current_text
                if current_text:
                    print(f"  Receiving... ({len(current_text)} chars)")
            
            await asyncio.sleep(1)
        
        return last_text.strip() if last_text else "No response received"
    
    async def _extract_response_text(self) -> str:
        """Extract the response text from the current page."""
        # The response appears in a prose block after the query
        # We need to find the actual answer content, not the whole page
        
        # First try: Look for prose elements (Perplexity's markdown container)
        prose_elements = await self._page.query_selector_all('[class*="prose"]')
        for el in prose_elements:
            try:
                text = await el.inner_text()
                # The answer prose block should have actual content
                if text and len(text.strip()) > 5:
                    # Clean up - remove common UI text
                    text = text.strip()
                    return text
            except:
                continue
        
        # Second try: Get the main content and parse it
        try:
            main_el = await self._page.query_selector('main')
            if main_el:
                full_text = await main_el.inner_text()
                
                # The response structure is typically:
                # [Navigation] [Query] [Answer] [Follow-up prompts] [Sign in stuff]
                # We want to extract just the answer portion
                
                lines = full_text.split('\n')
                answer_lines = []
                in_answer = False
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Skip navigation/UI elements
                    skip_patterns = [
                        'Home', 'Discover', 'Spaces', 'Finance', 'Answer', 
                        'Links', 'Images', 'Share', 'Ask a follow-up',
                        'Sign in', 'Unlock', 'Continue with', 'Log in',
                        'Create account', 'Google', 'Apple',
                    ]
                    
                    if any(line.startswith(p) or line == p for p in skip_patterns):
                        if in_answer:
                            break  # End of answer section
                        continue
                    
                    # If we see the query repeated, the answer follows
                    if not in_answer:
                        in_answer = True
                    
                    answer_lines.append(line)
                
                if answer_lines:
                    return '\n'.join(answer_lines)
        except:
            pass
        
        return ""
    
    async def _extract_references(self) -> List[Dict[str, str]]:
        """Extract source references from the page."""
        references = []
        
        # Try multiple selectors for sources
        selectors = [
            '[data-testid="source-item"] a',
            '.source-item a',
            '[class*="citation"] a',
            '[class*="source"] a',
            '.prose a[href^="http"]',
        ]
        
        seen_urls = set()
        for selector in selectors:
            links = await self._page.query_selector_all(selector)
            for link in links:
                try:
                    url = await link.get_attribute("href")
                    title = await link.inner_text()
                    if url and url.startswith("http") and url not in seen_urls:
                        seen_urls.add(url)
                        references.append({
                            "url": url,
                            "title": title.strip() if title else url,
                        })
                except:
                    continue
        
        return references


# ---------------------------------------------------------------------------
# Task Orchestrator
# ---------------------------------------------------------------------------

# Orchestrator configuration defaults
ORCHESTRATOR_MAX_DEPTH = 5
ORCHESTRATOR_MAX_TASKS = 100
ORCHESTRATOR_CONFIRM_INTERVAL = 20
ORCHESTRATOR_TEMP_DIR = Path.home() / ".perplexity-cli" / "orchestrator-state"

# Planner prompt for task decomposition
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

# JSON repair prompt
FIXER_PROMPT = '''You are a JSON repair expert. The following JSON is invalid. Please fix it and return ONLY the corrected, valid JSON object.

INVALID JSON:
{broken_json}

ERROR:
{error_msg}

Return ONLY the fixed JSON string. No other text.'''


class TaskOrchestrator:
    """
    Orchestrates multi-step task execution with isolated sub-tasks.
    
    Uses a planner prompt to decompose a high-level goal into independent sub-tasks,
    then executes each with its designated model. Sub-tasks receive only general 
    context - no results from sibling tasks - producing independent fragments
    that assemble into a larger whole.
    
    Usage:
        # With existing browser
        orchestrator = TaskOrchestrator("Create a business plan", browser=browser)
        await orchestrator.run_async()
        
        # Standalone (will create its own browser)
        orchestrator = TaskOrchestrator("Create a business plan")
        await orchestrator.run_async()
        
        # Synchronous usage
        orchestrator = TaskOrchestrator("Create a business plan")
        orchestrator.run()
    """
    
    def __init__(
        self,
        goal: str,
        browser: Optional[PerplexityBrowser] = None,
        output_dir: Optional[Path] = None,
        max_depth: int = ORCHESTRATOR_MAX_DEPTH,
        max_tasks: int = ORCHESTRATOR_MAX_TASKS,
        confirm_interval: int = ORCHESTRATOR_CONFIRM_INTERVAL,
        auto_confirm: bool = False,
    ):
        """
        Initialize the task orchestrator.
        
        Args:
            goal: The high-level goal or task to accomplish.
            browser: Optional PerplexityBrowser instance. If not provided, one will be created.
            output_dir: Directory for output files. Auto-generated if not provided.
            max_depth: Maximum levels of task decomposition (default: 5).
            max_tasks: Maximum total atomic tasks allowed (default: 100).
            confirm_interval: Ask for confirmation after this many tasks (default: 20).
            auto_confirm: If True, skip human confirmation prompts.
        """
        self.goal = goal
        self._browser = browser
        self._owns_browser = browser is None  # Track if we need to close the browser
        
        self.max_depth = max_depth
        self.max_tasks = max_tasks
        self.confirm_interval = confirm_interval
        self.auto_confirm = auto_confirm
        
        self.context = ""
        self.subtasks: List[SubTask] = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Task tracking
        self.total_tasks_created = 0
        self.tasks_executed = 0
        self.user_cancelled = False
        self.phase = "init"  # init, planning, executing, assembling, complete
        
        # Output paths
        safe_goal = "".join(c if c.isalnum() or c in "_-" else "_" for c in goal)[:40]
        self.run_id = f"{safe_goal}_{self.timestamp}"
        self.output_dir = output_dir or Path(f"./orchestrator-output/{self.run_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State persistence
        self.temp_dir = ORCHESTRATOR_TEMP_DIR / self.run_id
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.temp_dir / "state.json"
        
        # Save initial state
        self.save_state()
    
    def _subtask_to_dict(self, task: SubTask) -> dict:
        """Convert a SubTask to a serializable dictionary."""
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
                "max_depth": self.max_depth,
                "max_tasks": self.max_tasks,
                "confirm_interval": self.confirm_interval
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
    def load_state(cls, state_file: Path, browser: Optional[PerplexityBrowser] = None) -> Optional['TaskOrchestrator']:
        """
        Load orchestrator state from a state file for resumption.
        
        Args:
            state_file: Path to the state.json file.
            browser: Optional browser instance to use.
        
        Returns:
            TaskOrchestrator instance or None if loading failed.
        """
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            
            meta = state["meta"]
            config = state["config"]
            
            orchestrator = cls(
                goal=meta["goal"],
                browser=browser,
                max_depth=config.get("max_depth", ORCHESTRATOR_MAX_DEPTH),
                max_tasks=config.get("max_tasks", ORCHESTRATOR_MAX_TASKS),
                confirm_interval=config.get("confirm_interval", ORCHESTRATOR_CONFIRM_INTERVAL),
            )
            
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
            orchestrator.output_dir = Path(f"./orchestrator-output/{meta['run_id']}")
            orchestrator.temp_dir = ORCHESTRATOR_TEMP_DIR / meta["run_id"]
            orchestrator.state_file = orchestrator.temp_dir / "state.json"
            
            return orchestrator
        except Exception as e:
            print(f"Error loading state: {e}")
            return None
    
    @classmethod
    def list_runs(cls) -> List[Dict[str, Any]]:
        """
        List available orchestrator runs in the temp directory.
        
        Returns:
            List of run info dictionaries.
        """
        runs = []
        if ORCHESTRATOR_TEMP_DIR.exists():
            for run_dir in sorted(ORCHESTRATOR_TEMP_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                state_file = run_dir / "state.json"
                if state_file.exists():
                    try:
                        with open(state_file) as f:
                            state = json.load(f)
                        runs.append({
                            "run_id": state["meta"]["run_id"],
                            "goal": state["meta"]["goal"],
                            "phase": state["meta"]["phase"],
                            "tasks_executed": state["progress"]["tasks_executed"],
                            "total_tasks": state["progress"]["total_tasks_created"],
                            "last_updated": state["meta"]["last_updated"],
                            "state_file": str(state_file),
                        })
                    except:
                        pass
        return runs
    
    async def _ensure_browser(self):
        """Ensure we have a browser instance."""
        if self._browser is None:
            self._browser = PerplexityBrowser()
            self._owns_browser = True
        await self._browser.start()
    
    async def _query(
        self,
        query: str,
        model: Optional[str] = None,
        mode: Optional[str] = None,
        focus: Optional[str] = None,
        timeout: int = 120,
    ) -> str:
        """Execute a query via the browser."""
        await self._ensure_browser()
        
        research_mode = mode == "research"
        labs_mode = mode == "labs"
        
        try:
            response = await self._browser.ask(
                query=query,
                model=model if model and model != "auto" else None,
                research_mode=research_mode,
                labs_mode=labs_mode,
                focus=focus,
                timeout=timeout * 3 if mode in ("research", "labs") else timeout,
                use_paste=True,  # Use paste for multi-line prompts
            )
            return response.answer
        except Exception as e:
            return f"[ERROR: {e}]"
    
    async def _repair_json(self, broken_json: str, error_msg: str, max_retries: int = 3) -> Optional[dict]:
        """Attempt to repair invalid JSON using an LLM."""
        print(f"  Repairing invalid JSON (max retries: {max_retries})...")
        
        current_error = error_msg
        current_json = broken_json
        
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"  Retry {attempt+1}/{max_retries}...")
            
            prompt = FIXER_PROMPT.format(broken_json=current_json, error_msg=current_error)
            response = await self._query(prompt, model="gpt", timeout=60)
            
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"  Repair attempt {attempt+1} failed: {e}")
                current_error = str(e)
                current_json = response
            except Exception as e:
                print(f"  JSON repair error: {e}")
                return None
        
        print("  All repair attempts failed.")
        return None
    
    async def _decompose(self, task_description: str, parent_id: str = "", depth: int = 0) -> List[SubTask]:
        """Recursively decompose a task into atomic sub-tasks."""
        if depth >= self.max_depth:
            print(f"  Max depth ({self.max_depth}) reached, treating remaining tasks as atomic")
            return []
        
        if self.total_tasks_created >= self.max_tasks:
            print(f"  Max task limit ({self.max_tasks}) reached, stopping decomposition")
            return []
        
        print(f"\nDecomposing task: {task_description[:50]}... (Depth {depth})")
        
        prompt = PLANNER_PROMPT.format(goal=task_description)
        response = await self._query(prompt, model="claude", timeout=90)
        
        plan = None
        try:
            json_start = response.find('{"goal"')
            if json_start == -1:
                json_start = response.find('{\n  "goal"')
            if json_start == -1:
                json_start = response.find('{\"goal\"')
            
            if json_start == -1:
                print("  Warning: No JSON found in planner response, treating as atomic")
                return []
            
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
                raise json.JSONDecodeError("Could not find closing brace", json_str, 0)
            
            plan = json.loads(json_str[:json_end])
            
        except json.JSONDecodeError as e:
            print(f"  Error parsing plan JSON: {e}")
            plan = await self._repair_json(response, str(e))
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
            if depth == 0:
                self.context = plan.get("context", "")
            
            for task_data in plan.get("subtasks", []):
                if self.total_tasks_created >= self.max_tasks:
                    print(f"  Task limit ({self.max_tasks}) reached during processing")
                    break
                
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
                
                if not subtask.is_atomic and depth < self.max_depth - 1:
                    print(f"  Task {task_id} is not atomic, decomposing...")
                    children = await self._decompose(subtask.prompt, task_id, depth + 1)
                    if children:
                        subtask.subtasks = children
                
                subtasks.append(subtask)
                
                if depth == 0:
                    self.subtasks = subtasks
                    self.save_state()
            
            return subtasks
            
        except Exception as e:
            print(f"  Error processing plan: {e}")
            return []
    
    async def plan(self) -> bool:
        """Decompose the goal into sub-tasks."""
        self.phase = "planning"
        self.save_state()
        
        print(f"\n{'═' * 70}")
        print("PHASE 1: TASK DECOMPOSITION")
        print(f"{'═' * 70}")
        print(f"\nGoal: {self.goal}")
        print("\nGenerating hierarchical execution plan...")
        
        self.subtasks = await self._decompose(self.goal)
        
        if self.subtasks:
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
    
    def _flatten_tasks(self, tasks: List[SubTask]) -> List[SubTask]:
        """Return a flat list of all atomic tasks to be executed."""
        flat = []
        for task in tasks:
            if task.subtasks:
                flat.extend(self._flatten_tasks(task.subtasks))
            else:
                flat.append(task)
        return flat
    
    async def _execute_subtask(self, subtask: SubTask) -> str:
        """Execute a single sub-task in isolation."""
        print(f"\n{'─' * 60}")
        print(f"SUB-TASK {subtask.id}: {subtask.title}")
        print(f"{'─' * 60}")
        print(f"  Model: {subtask.model}")
        print(f"  Mode: {subtask.mode or 'standard'}")
        print(f"  Focus: {subtask.focus or 'default'}")
        print(f"  Contribution: {subtask.contribution}")
        
        isolated_prompt = f"CONTEXT: {self.context}. TASK: {subtask.prompt}. Provide a thorough response focused specifically on this aspect."
        
        result = await self._query(
            isolated_prompt,
            model=subtask.model,
            mode=subtask.mode,
            focus=subtask.focus
        )
        
        subtask.result = result
        
        result_path = self.output_dir / f"{subtask.id.replace('.', '_')}_{subtask.title.replace(' ', '_')[:30]}.txt"
        with open(result_path, "w") as f:
            f.write(f"# Sub-Task {subtask.id}: {subtask.title}\n")
            f.write(f"# Model: {subtask.model}\n")
            f.write(f"# Contribution: {subtask.contribution}\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(result)
        
        self.save_state()
        
        print(f"  ✓ Saved to: {result_path}")
        print(f"  ✓ State updated: {self.state_file}")
        
        return result
    
    def _confirm_continue(self) -> bool:
        """Ask user for confirmation to continue execution."""
        if self.auto_confirm:
            return True
        
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
    
    async def execute_all(self):
        """Execute all atomic sub-tasks."""
        self.phase = "executing"
        self.save_state()
        
        print(f"\n{'═' * 70}")
        print("PHASE 2: ISOLATED SUB-TASK EXECUTION")
        print(f"{'═' * 70}")
        
        execution_list = self._flatten_tasks(self.subtasks)
        
        print(f"\nExecuting {len(execution_list)} atomic sub-tasks...")
        if not self.auto_confirm:
            print(f"(Will pause for confirmation every {self.confirm_interval} tasks)")
        
        for i, subtask in enumerate(execution_list, 1):
            if self.user_cancelled:
                print("\n  Execution cancelled by user.")
                break
            
            if not self.auto_confirm and self.tasks_executed > 0 and self.tasks_executed % self.confirm_interval == 0:
                if not self._confirm_continue():
                    self.user_cancelled = True
                    print("\n  User chose to stop. Assembling partial results...")
                    break
            
            print(f"\n[{i}/{len(execution_list)}]", end="")
            await self._execute_subtask(subtask)
            self.tasks_executed += 1
    
    def _format_results_recursive(self, tasks: List[SubTask], level: int = 0) -> str:
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
                content = task.result or "[No result]"
                output += "\n".join(f"{indent}  {line}" for line in content.splitlines())
                output += "\n"
        
        return output
    
    def assemble(self) -> Path:
        """Assemble all fragments into the final deliverable."""
        self.phase = "assembling"
        self.save_state()
        
        print(f"\n{'═' * 70}")
        print("PHASE 3: ASSEMBLY")
        print(f"{'═' * 70}")
        
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
    
    async def run_async(self) -> bool:
        """Execute the full orchestration pipeline asynchronously."""
        print("\n" + "╔" + "═" * 68 + "╗")
        print(f"║{'TASK ORCHESTRATOR':^68}║")
        print("╚" + "═" * 68 + "╝")
        
        try:
            # Phase 1: Plan
            if not await self.plan():
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
            await self.execute_all()
            
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
        
        finally:
            # Clean up browser if we created it
            if self._owns_browser and self._browser:
                await self._browser.stop()
    
    def run(self) -> bool:
        """Execute the full orchestration pipeline synchronously."""
        return asyncio.run(self.run_async())


# ---------------------------------------------------------------------------
# HTTP API Server
# ---------------------------------------------------------------------------

if FastAPI:
    from contextlib import asynccontextmanager
    
    # Global browser instance
    _browser: Optional[PerplexityBrowser] = None
    
    @asynccontextmanager
    async def lifespan(app):
        """Manage browser lifecycle."""
        global _browser
        # Use the saved profile from login
        _browser = PerplexityBrowser()
        try:
            await _browser.start()
            if not await _browser.is_logged_in():
                print("\n⚠️  Not logged in. Run with --login first.")
            else:
                print("✓ Logged in to Perplexity")
        except Exception as e:
            print(f"Browser startup error: {e}")
        
        yield
        
        if _browser:
            await _browser.stop()
    
    app = FastAPI(
        title="Perplexity Bridge API",
        description="Local API bridge to Perplexity AI via browser automation",
        version=__version__,
        lifespan=lifespan,
    )
    
    class QueryRequest(BaseModel):
        query: str
        focus: str = "internet"
        pro_mode: bool = False
        model: Optional[str] = None  # Model selection (e.g., "gpt-4o", "claude-sonnet", "sonar")
        research_mode: bool = False  # Deep research mode
        labs_mode: bool = False  # Labs mode for experimental features
        timeout: int = 60
    
    class QueryResponse(BaseModel):
        answer: str
        references: List[Dict[str, str]]
        query: str
        model: Optional[str] = None
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        logged_in = await _browser.is_logged_in() if _browser else False
        return {
            "status": "ok",
            "logged_in": logged_in,
            "version": __version__,
        }
    
    async def _process_ask_request(request: QueryRequest) -> QueryResponse:
        """Shared logic for processing ask requests."""
        if not _browser:
            raise HTTPException(status_code=503, detail="Browser not initialized")
        
        if not await _browser.is_logged_in():
            raise HTTPException(status_code=401, detail="Not logged in. Run with --login first.")
        
        # Validate model if provided
        if request.model and request.model not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model '{request.model}'. Available: {', '.join(AVAILABLE_MODELS.keys())}"
            )
        
        try:
            response = await _browser.ask(
                query=request.query,
                focus=request.focus,
                pro_mode=request.pro_mode,
                model=request.model,
                research_mode=request.research_mode,
                labs_mode=request.labs_mode,
                timeout=request.timeout,
            )
            return QueryResponse(**response.to_dict())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/ask", response_model=QueryResponse)
    async def ask(request: QueryRequest):
        """
        Ask a question to Perplexity.
        
        Returns the answer and source references.
        """
        return await _process_ask_request(request)
    
    @app.get("/ask")
    async def ask_get(
        q: str,
        focus: str = "internet",
        pro_mode: bool = False,
        model: Optional[str] = None,
        research: bool = False,
        labs: bool = False,
        timeout: int = 60,
    ):
        """GET endpoint for simple queries."""
        request = QueryRequest(
            query=q, 
            focus=focus, 
            pro_mode=pro_mode, 
            model=model,
            research_mode=research,
            labs_mode=labs,
            timeout=timeout
        )
        return await _process_ask_request(request)
    
    @app.get("/models")
    async def list_models():
        """List available models for selection."""
        return {
            "models": list(AVAILABLE_MODELS.keys()),
            "descriptions": AVAILABLE_MODELS,
        }


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

class tColor:
    reset = '\033[0m'
    bold = '\033[1m'
    red = '\033[91m'
    green = '\033[92m'
    yellow = '\033[93m'
    blue = '\033[94m'
    purple = '\033[38;2;181;76;210m'
    lavand = '\033[38;5;140m'
    aqua = '\033[38;5;109m'
    aqua2 = '\033[38;5;158m'


def render_answer(answer: str, typing_delay: float = 0.02):
    """Display answer with typing animation."""
    print(tColor.aqua2, end='\n', flush=True)
    for char in answer:
        print(char, end='', flush=True)
        if typing_delay > 0:
            sleep(typing_delay)
    print(tColor.reset, end='\n', flush=True)


def render_references(refs: List[Dict[str, str]]):
    """Display references."""
    if not refs:
        return
    print(f"\n{tColor.lavand}References:{tColor.reset}")
    for i, ref in enumerate(refs, 1):
        title = ref.get("title", ref.get("url", ""))
        url = ref.get("url", "")
        print(f"  [{i}] {title}")
        if url != title:
            print(f"      {tColor.blue}{url}{tColor.reset}")


async def cli_ask(
    browser: PerplexityBrowser, 
    query: str, 
    show_refs: bool = True, 
    typing_delay: float = 0.02,
    model: Optional[str] = None,
    research_mode: bool = False,
    labs_mode: bool = False,
    focus: str = "internet",
    use_paste: bool = False,
):
    """CLI: Ask a single question."""
    mode_info = []
    if labs_mode:
        mode_info.append(f"{tColor.yellow}Labs{tColor.reset}")
    elif research_mode:
        mode_info.append(f"{tColor.purple}Research{tColor.reset}")
    if model:
        model_name = AVAILABLE_MODELS.get(model, model)
        mode_info.append(f"{tColor.aqua}{model_name}{tColor.reset}")
    if focus != "internet":
        mode_info.append(f"Focus: {focus}")
    
    mode_str = f" [{', '.join(mode_info)}]" if mode_info else ""
    print(f"{tColor.bold}Asking:{tColor.reset} {query}{mode_str}\n")
    
    try:
        response = await browser.ask(
            query, 
            model=model, 
            research_mode=research_mode,
            labs_mode=labs_mode,
            focus=focus,
            use_paste=use_paste,
        )
        render_answer(response.answer, typing_delay=typing_delay)
        if show_refs:
            render_references(response.references)
    except Exception as e:
        print(f"{tColor.red}Error: {e}{tColor.reset}")
        sys.exit(1)


async def cli_interactive(
    browser: PerplexityBrowser, 
    typing_delay: float = 0.02,
    model: Optional[str] = None,
    research_mode: bool = False,
    labs_mode: bool = False,
    focus: str = "internet",
    use_paste: bool = False,
):
    """CLI: Interactive mode."""
    print(f"{tColor.bold}Perplexity AI Bridge{tColor.reset} - Interactive Mode")
    
    # Show current settings
    settings = []
    if model:
        settings.append(f"Model: {AVAILABLE_MODELS.get(model, model)}")
    if labs_mode:
        settings.append("Labs Mode")
    elif research_mode:
        settings.append("Research Mode")
    if focus != "internet":
        settings.append(f"Focus: {focus}")
    if settings:
        print(f"Settings: {', '.join(settings)}")
    
    print("Type your question and press Enter. Type 'exit' to quit.")
    print("Commands: /model <name>, /research, /labs, /focus <mode>, /help\n")
    
    current_model = model
    current_research = research_mode
    current_labs = labs_mode
    current_focus = focus
    
    while True:
        try:
            prompt_parts = []
            if current_model:
                prompt_parts.append(f"{tColor.aqua}{current_model}{tColor.reset}")
            if current_labs:
                prompt_parts.append(f"{tColor.yellow}L{tColor.reset}")
            elif current_research:
                prompt_parts.append(f"{tColor.purple}R{tColor.reset}")
            prompt_prefix = f"[{'/'.join(prompt_parts)}] " if prompt_parts else ""
            
            print(f"{prompt_prefix}{tColor.bold}❯{tColor.reset} ", end="")
            query = input().strip()
            
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                break
            
            # Handle commands
            if query.startswith("/"):
                parts = query.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""
                
                if cmd == "/model":
                    if arg in AVAILABLE_MODELS:
                        current_model = arg if arg != "auto" else None
                        print(f"  Model set to: {AVAILABLE_MODELS.get(arg, arg) or 'auto'}")
                    else:
                        print(f"  Available models: {', '.join(AVAILABLE_MODELS.keys())}")
                elif cmd == "/research":
                    current_research = not current_research
                    if current_research:
                        current_labs = False  # Mutually exclusive
                    print(f"  Research mode: {'ON' if current_research else 'OFF'}")
                elif cmd == "/labs":
                    current_labs = not current_labs
                    if current_labs:
                        current_research = False  # Mutually exclusive
                    print(f"  Labs mode: {'ON' if current_labs else 'OFF'}")
                elif cmd == "/focus":
                    if arg in ["internet", "academic", "writing", "wolfram", "youtube", "reddit"]:
                        current_focus = arg
                        print(f"  Focus set to: {current_focus}")
                    else:
                        print("  Available: internet, academic, writing, wolfram, youtube, reddit")
                elif cmd == "/help":
                    print("  /model <name>  - Set model (e.g., gpt-4o, claude-sonnet)")
                    print("  /research      - Toggle research mode")
                    print("  /labs          - Toggle labs mode")
                    print("  /focus <mode>  - Set focus (internet, academic, etc.)")
                    print("  /help          - Show this help")
                else:
                    print(f"  Unknown command: {cmd}")
                continue
            
            response = await browser.ask(
                query,
                model=current_model,
                research_mode=current_research,
                labs_mode=current_labs,
                focus=current_focus,
                use_paste=use_paste,
            )
            render_answer(response.answer, typing_delay=typing_delay)
            render_references(response.references)
            print()
            
        except KeyboardInterrupt:
            break
        except EOFError:
            break
    
    print(f"\n{tColor.yellow}Goodbye!{tColor.reset}")


async def run_server(host: str, port: int):
    """Run the HTTP API server."""
    if not FastAPI:
        print(f"{tColor.red}FastAPI not installed. Run: pip install fastapi uvicorn{tColor.reset}")
        sys.exit(1)
    
    print(f"{tColor.green}Starting Perplexity Bridge API server...{tColor.reset}")
    print(f"  URL: http://{host}:{port}")
    print(f"  Docs: http://{host}:{port}/docs")
    print(f"\n{tColor.yellow}Press Ctrl+C to stop{tColor.reset}\n")
    
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Perplexity AI Bridge - Browser automation + Local API + Task Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Connect to running Brave/Chrome (recommended - uses your session)
  # First, start your browser with: brave --remote-debugging-port=9222
  %(prog)s --cdp "What is quantum computing?"
  
  # Ask a question (launches new browser)
  %(prog)s "What is quantum computing?"
  
  # Use a specific model (requires Pro subscription)
  %(prog)s --model gpt-4o "Explain relativity"
  %(prog)s -m claude-sonnet "Write a poem"
  
  # Use Research mode for deep research
  %(prog)s --research "History of quantum computing"
  %(prog)s -r "Compare modern AI architectures"
  
  # Use Labs mode for experimental features
  %(prog)s --labs "Build me a chart of stock prices"
  %(prog)s -l "Create a presentation about AI"
  
  # Combine model and focus
  %(prog)s -m sonar -f academic "Latest fusion research"
  
  # Interactive mode (with live model switching)
  %(prog)s -i
  %(prog)s -i -m gpt-4o  # Start with GPT-4o
  
  # List available models
  %(prog)s --list-models
  
  # Login in new browser profile
  %(prog)s --login
  
  # Start HTTP API server
  %(prog)s --serve
  
  # Query via HTTP:
  curl "http://localhost:8000/ask?q=What+is+AI"
  curl "http://localhost:8000/ask?q=Deep+topic&research=true"
  curl "http://localhost:8000/ask?q=Build+a+chart&labs=true"
  curl "http://localhost:8000/ask?q=Question&model=gpt-4o"

  # ORCHESTRATOR - Multi-step task execution
  %(prog)s --orchestrate "Create a business plan for a SaaS startup"
  %(prog)s --orchestrate "Design a curriculum for learning ML" --max-depth 3
  %(prog)s --orchestrate-resume temp/My_Goal_20251128/state.json
  %(prog)s --orchestrate-list

To use your existing logged-in session:
  1. Close your browser
  2. Restart it with: brave --remote-debugging-port=9222
     (or: google-chrome --remote-debugging-port=9222)
  3. Log into Perplexity in that browser
  4. Run: %(prog)s --cdp "Your question"
"""
    )
    
    parser.add_argument("query", nargs="?", help="Question to ask")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--login", action="store_true",
                        help="Open browser for login")
    parser.add_argument("--serve", action="store_true",
                        help="Start HTTP API server")
    parser.add_argument("--host", default="127.0.0.1",
                        help="API server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="API server port (default: 8000)")
    parser.add_argument("--cdp", action="store_true",
                        help="Connect to browser via CDP at http://localhost:9222")
    parser.add_argument("--cdp-url", type=str, default="http://localhost:9222",
                        help="CDP URL (default: http://localhost:9222)")
    parser.add_argument("--profile", type=str, default=None,
                        help="Path to browser profile directory")
    parser.add_argument("--no-typing", action="store_true",
                        help="Disable typing animation")
    parser.add_argument("--paste", action="store_true",
                        help="Use clipboard paste instead of typing (faster, preserves formatting)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        choices=list(AVAILABLE_MODELS.keys()),
                        help="Model to use (requires Pro). Options: " + ", ".join(AVAILABLE_MODELS.keys()))
    parser.add_argument("--research", "-r", action="store_true",
                        help="Use Research mode for deep, multi-step research")
    parser.add_argument("--labs", "-l", action="store_true",
                        help="Use Labs mode for experimental features")
    parser.add_argument("--focus", "-f", type=str, default="internet",
                        choices=["internet", "academic", "writing", "wolfram", "youtube", "reddit"],
                        help="Search focus mode (default: internet)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    
    # Orchestrator arguments
    orchestrator_group = parser.add_argument_group('Orchestrator Options')
    orchestrator_group.add_argument("--orchestrate", "-O", type=str, metavar="GOAL",
                                    help="Run the task orchestrator with the given goal")
    orchestrator_group.add_argument("--orchestrate-resume", type=Path, metavar="STATE_FILE",
                                    help="Resume orchestrator from a previous state file")
    orchestrator_group.add_argument("--orchestrate-list", action="store_true",
                                    help="List available orchestrator runs")
    orchestrator_group.add_argument("--max-depth", type=int, default=ORCHESTRATOR_MAX_DEPTH,
                                    help=f"Max decomposition depth (default: {ORCHESTRATOR_MAX_DEPTH})")
    orchestrator_group.add_argument("--max-tasks", type=int, default=ORCHESTRATOR_MAX_TASKS,
                                    help=f"Max total tasks (default: {ORCHESTRATOR_MAX_TASKS})")
    orchestrator_group.add_argument("--confirm-interval", type=int, default=ORCHESTRATOR_CONFIRM_INTERVAL,
                                    help=f"Tasks between confirmations (default: {ORCHESTRATOR_CONFIRM_INTERVAL})")
    orchestrator_group.add_argument("--no-confirm", action="store_true",
                                    help="Disable human-in-the-loop confirmations for orchestrator")
    
    args = parser.parse_args()
    
    # List models and exit
    if args.list_models:
        print(f"{tColor.bold}Available Models:{tColor.reset}")
        for key, name in AVAILABLE_MODELS.items():
            if name:
                print(f"  {tColor.aqua}{key:20}{tColor.reset} → {name}")
            else:
                print(f"  {tColor.aqua}{key:20}{tColor.reset} → (auto-select)")
        return
    
    # List orchestrator runs
    if args.orchestrate_list:
        print(f"\n{tColor.bold}Available orchestrator runs:{tColor.reset}")
        print(f"{'─' * 70}")
        runs = TaskOrchestrator.list_runs()
        if runs:
            for run in runs:
                print(f"\n  {tColor.aqua}{run['run_id']}{tColor.reset}")
                print(f"    Goal: {run['goal'][:50]}...")
                print(f"    Phase: {run['phase']}")
                print(f"    Progress: {run['tasks_executed']}/{run['total_tasks']} tasks")
                print(f"    Last updated: {run['last_updated']}")
                print(f"    Resume: --orchestrate-resume {run['state_file']}")
        else:
            print("  No runs found.")
        return
    
    # Resume orchestrator from state
    if args.orchestrate_resume:
        if not args.orchestrate_resume.exists():
            print(f"{tColor.red}Error: State file not found: {args.orchestrate_resume}{tColor.reset}")
            sys.exit(1)
        
        print(f"\n{tColor.bold}Resuming orchestrator from:{tColor.reset} {args.orchestrate_resume}")
        
        # Create browser for orchestrator
        browser = PerplexityBrowser(
            cdp_url=args.cdp_url if args.cdp else None,
            profile_path=args.profile,
        )
        
        orchestrator = TaskOrchestrator.load_state(args.orchestrate_resume, browser=browser)
        if not orchestrator:
            print(f"{tColor.red}Error: Failed to load state file{tColor.reset}")
            sys.exit(1)
        
        orchestrator.auto_confirm = args.no_confirm
        
        print(f"  Goal: {orchestrator.goal}")
        print(f"  Phase: {orchestrator.phase}")
        print(f"  Progress: {orchestrator.tasks_executed}/{orchestrator.total_tasks_created} tasks")
        
        try:
            await browser.start()
            if not await browser.is_logged_in():
                print(f"{tColor.yellow}Not logged in. Run with --login first.{tColor.reset}")
                sys.exit(1)
            
            if orchestrator.phase == "planning":
                success = await orchestrator.run_async()
            elif orchestrator.phase == "executing":
                await orchestrator.execute_all()
                orchestrator.assemble()
                orchestrator.phase = "complete"
                orchestrator.save_state()
                success = True
            elif orchestrator.phase == "assembling":
                orchestrator.assemble()
                orchestrator.phase = "complete"
                orchestrator.save_state()
                success = True
            elif orchestrator.phase == "complete":
                print(f"\n{tColor.yellow}Run already complete. Nothing to resume.{tColor.reset}")
                success = True
            else:
                success = await orchestrator.run_async()
            
            sys.exit(0 if success else 1)
        finally:
            await browser.stop()
    
    # Run orchestrator with new goal
    if args.orchestrate:
        print(f"\n{tColor.bold}Starting Task Orchestrator{tColor.reset}")
        print(f"  Config: max_depth={args.max_depth}, max_tasks={args.max_tasks}, ", end="")
        if args.no_confirm:
            print("confirmations=disabled")
        else:
            print(f"confirm_every={args.confirm_interval}")
        
        # Create browser for orchestrator
        browser = PerplexityBrowser(
            cdp_url=args.cdp_url if args.cdp else None,
            profile_path=args.profile,
        )
        
        try:
            await browser.start()
            if not await browser.is_logged_in():
                print(f"{tColor.yellow}Not logged in. Run with --login first.{tColor.reset}")
                sys.exit(1)
            
            orchestrator = TaskOrchestrator(
                goal=args.orchestrate,
                browser=browser,
                max_depth=args.max_depth,
                max_tasks=args.max_tasks,
                confirm_interval=args.confirm_interval,
                auto_confirm=args.no_confirm,
            )
            
            print(f"  State file: {orchestrator.state_file}")
            success = await orchestrator.run_async()
            sys.exit(0 if success else 1)
        finally:
            await browser.stop()
    
    # Server mode
    if args.serve:
        await run_server(args.host, args.port)
        return
    
    # Browser instance
    browser = PerplexityBrowser(
        cdp_url=args.cdp_url if args.cdp else None,
        profile_path=args.profile,
    )
    
    try:
        # Login mode
        if args.login:
            await browser.login_interactive()
            return
        
        # Check login status
        await browser.start()
        if not await browser.is_logged_in():
            print(f"{tColor.yellow}Not logged in. Run with --login first.{tColor.reset}")
            print(f"  {sys.argv[0]} --login")
            sys.exit(1)
        
        # Determine typing delay
        typing_delay = 0 if args.no_typing else 0.02
        
        # Interactive mode
        if args.interactive or not args.query:
            await cli_interactive(
                browser, 
                typing_delay=typing_delay,
                model=args.model,
                research_mode=args.research,
                labs_mode=args.labs,
                focus=args.focus,
                use_paste=args.paste,
            )
        else:
            # Single query
            await cli_ask(
                browser, 
                args.query, 
                typing_delay=typing_delay,
                model=args.model,
                research_mode=args.research,
                labs_mode=args.labs,
                focus=args.focus,
                use_paste=args.paste,
            )
    
    finally:
        await browser.stop()


def cli_main():
    """Synchronous entry point for CLI."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
