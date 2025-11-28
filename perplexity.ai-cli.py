#!/usr/bin/env python

"""
Perplexity AI Bridge - Browser automation + Local API

Automates the Perplexity web UI via browser and exposes it as:
  - CLI interface for direct queries
  - HTTP API for integration with other projects

No reverse engineering, no token management - just real browser automation.
"""

__version__ = "1.0.0"

import asyncio
import json
import os
import sys
import re
from time import sleep
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

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
        timeout: int = 60,
    ) -> PerplexityResponse:
        """
        Ask a question and get the response.
        
        Args:
            query: The question to ask.
            focus: Search focus ('internet', 'academic', 'writing', 'wolfram', 'youtube', 'reddit').
            pro_mode: Use Pro/Copilot mode if available.
            timeout: Maximum seconds to wait for response.
        
        Returns:
            PerplexityResponse with answer and references.
        """
        await self.start()
        
        # Navigate to new thread
        await self._page.goto(self.PERPLEXITY_URL, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(3)
        
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
        
        # Click to focus the input
        await input_element.click()
        await asyncio.sleep(0.3)
        
        # For contenteditable divs, we need to type character by character
        # Using fill() doesn't work well with Lexical editor
        await self._page.keyboard.type(query, delay=20)
        
        await asyncio.sleep(0.5)
        
        # Submit with Enter
        await self._page.keyboard.press("Enter")
        
        print("  Query submitted, waiting for response...")
        
        # Wait for response to complete
        response_text = await self._wait_for_response(timeout)
        references = await self._extract_references()
        
        return PerplexityResponse(
            answer=response_text,
            references=references,
            query=query,
        )
    
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
        
        try:
            response = await _browser.ask(
                query=request.query,
                focus=request.focus,
                pro_mode=request.pro_mode,
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
        timeout: int = 60,
    ):
        """GET endpoint for simple queries."""
        request = QueryRequest(query=q, focus=focus, pro_mode=pro_mode, timeout=timeout)
        return await _process_ask_request(request)


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


async def cli_ask(browser: PerplexityBrowser, query: str, show_refs: bool = True, typing_delay: float = 0.02):
    """CLI: Ask a single question."""
    print(f"{tColor.bold}Asking:{tColor.reset} {query}\n")
    
    try:
        response = await browser.ask(query)
        render_answer(response.answer, typing_delay=typing_delay)
        if show_refs:
            render_references(response.references)
    except Exception as e:
        print(f"{tColor.red}Error: {e}{tColor.reset}")
        sys.exit(1)


async def cli_interactive(browser: PerplexityBrowser, typing_delay: float = 0.02):
    """CLI: Interactive mode."""
    print(f"{tColor.bold}Perplexity AI Bridge{tColor.reset} - Interactive Mode")
    print("Type your question and press Enter. Type 'exit' to quit.\n")
    
    while True:
        try:
            print(f"{tColor.bold}❯{tColor.reset} ", end="")
            query = input().strip()
            
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                break
            
            response = await browser.ask(query)
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
        description="Perplexity AI Bridge - Browser automation + Local API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Connect to running Brave/Chrome (recommended - uses your session)
  # First, start your browser with: brave --remote-debugging-port=9222
  %(prog)s --cdp "What is quantum computing?"
  
  # Ask a question (launches new browser)
  %(prog)s "What is quantum computing?"
  
  # Interactive mode
  %(prog)s -i
  
  # Login in new browser profile
  %(prog)s --login
  
  # Start HTTP API server
  %(prog)s --serve
  
  # Then query via HTTP:
  curl "http://localhost:8000/ask?q=What+is+AI"

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
    
    args = parser.parse_args()
    
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
            await cli_interactive(browser, typing_delay=typing_delay)
        else:
            # Single query
            await cli_ask(browser, args.query, typing_delay=typing_delay)
    
    finally:
        await browser.stop()


if __name__ == "__main__":
    asyncio.run(main())
