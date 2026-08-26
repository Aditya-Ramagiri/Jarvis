"""macOS menu bar status item (spec section 1, 3).

Adrien has no dock icon and no window - it is a background service. That makes
a menu bar item genuinely important rather than decorative: without it there is
no way to tell whether the thing is listening, which provider it is on, or how
to make it stop, short of reading a log file.

The item shows state at a glance through its icon:

    ●   idle, listening for the wake word
    ◉   in conversation
    ◌   speaking
    ○   paused
    ⊘   degraded (every key of some provider is cooling down)

Runs on the main thread because AppKit insists; the orchestrator runs its
asyncio loop on a worker thread underneath. `rumps` is macOS-only and optional -
`adrien run` works perfectly well without it, just silently.
"""

from __future__ import annotations

import asyncio
import threading
import webbrowser
from typing import Any

from adrien.config import log_dir, settings
from adrien.core.conversation import WindowState
from adrien.logging_setup import get_logger, setup_logging

log = get_logger(__name__)

ICONS = {
    WindowState.IDLE: "●",
    WindowState.LISTENING: "◉",
    WindowState.THINKING: "◉",
    WindowState.SPEAKING: "◌",
    WindowState.FOLLOW_UP: "◉",
    WindowState.CONFIRMING: "◉",
}
PAUSED_ICON = "○"
DEGRADED_ICON = "⊘"


def run_menu_bar() -> None:
    """Start Adrien with a menu bar item. Falls back to headless if rumps
    is unavailable."""
    try:
        import rumps
    except ImportError:
        log.warning("rumps is not installed; running without a menu bar")
        from adrien.__main__ import main

        main(["run"])
        return

    from adrien.core.orchestrator import Orchestrator
    from adrien.server.ws_server import AdrienServer, local_ip

    class AdrienStatusItem(rumps.App):
        def __init__(self) -> None:
            super().__init__("●", quit_button=None)
            self.orchestrator = Orchestrator()
            self.orchestrator.on_state_change = self._on_state_change
            self.paused = False
            self._loop: asyncio.AbstractEventLoop | None = None
            self._server: AdrienServer | None = None

            self.menu = [
                rumps.MenuItem("Adrien", callback=None),
                rumps.separator,
                rumps.MenuItem("Pause listening", callback=self.toggle_pause),
                rumps.MenuItem("Status", callback=self.show_status),
                rumps.MenuItem("What Adrien remembers", callback=self.show_memory),
                rumps.separator,
                rumps.MenuItem("Retry cooling keys", callback=self.reset_keys),
                rumps.MenuItem("Open log", callback=self.open_log),
                rumps.MenuItem("Reload settings", callback=self.reload_settings),
                rumps.separator,
                rumps.MenuItem("Quit Adrien", callback=self.quit_app),
            ]

            threading.Thread(target=self._run_loop, daemon=True, name="adrien").start()

        # -- the asyncio side ------------------------------------------
        def _run_loop(self) -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._serve())
            except Exception:
                log.exception("the orchestrator stopped")

        async def _serve(self) -> None:
            if settings().get("server.enabled", True):
                self._server = AdrienServer(self.orchestrator)
                try:
                    await self._server.start()
                    log.info("clients: ws://%s:%d", local_ip(), self._server.port)
                except Exception as exc:
                    log.warning("client server not started: %s", exc)
                    self._server = None
            await self.orchestrator.run()

        def _submit(self, coroutine) -> Any:
            """Run a coroutine on the orchestrator's loop from the UI thread."""
            if self._loop is None:
                return None
            return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

        # -- state -----------------------------------------------------
        def _on_state_change(self, state: WindowState) -> None:
            if self.paused:
                return
            self.title = ICONS.get(state, "●")

        # -- menu actions ----------------------------------------------
        def toggle_pause(self, sender) -> None:
            self.paused = not self.paused
            if self.paused:
                if self.orchestrator.mic is not None:
                    self.orchestrator.mic.mute()
                self.orchestrator.speaker.stop()
                sender.title = "Resume listening"
                self.title = PAUSED_ICON
            else:
                if self.orchestrator.mic is not None:
                    self.orchestrator.mic.unmute()
                sender.title = "Pause listening"
                self.title = ICONS[WindowState.IDLE]

        def show_status(self, _) -> None:
            status = self.orchestrator.status()
            providers = status["providers"]["providers"]
            lines = [
                f"State: {status['state']}",
                f"Wake word: {status['wake_word']}"
                + (" (fallback - see docs/WAKE_WORD.md)"
                   if status["wake_word_is_fallback"] else ""),
                f"Tools: {status['tools']}",
                "",
            ]
            for provider in providers:
                lines.append(
                    f"{provider['name']}: {provider['available']}/{provider['keys']} keys ready"
                )
            memory = status["memory"]
            lines += ["", f"Memory: {memory['facts']} facts, {memory['sessions']} sessions"]

            if not status["providers"]["healthy"]:
                self.title = DEGRADED_ICON
            rumps.alert(title="Adrien", message="\n".join(lines), ok="Close")

        def show_memory(self, _) -> None:
            facts = self.orchestrator.memory.known_facts()
            if not facts:
                rumps.alert(title="Adrien remembers", message="Nothing yet.", ok="Close")
                return
            shown = "\n".join(f"• {fact.as_sentence()}" for fact in facts[:25])
            extra = f"\n\n...and {len(facts) - 25} more" if len(facts) > 25 else ""
            rumps.alert(title="Adrien remembers", message=shown + extra, ok="Close")

        def reset_keys(self, _) -> None:
            self.orchestrator.router.reset_cooldowns()
            self.title = ICONS[WindowState.IDLE]
            rumps.notification("Adrien", "", "Every key is back in rotation.")

        def open_log(self, _) -> None:
            webbrowser.open(f"file://{log_dir() / 'adrien.log'}")

        def reload_settings(self, _) -> None:
            from adrien.config import reload_settings

            reloaded = reload_settings()
            self.orchestrator.settings = reloaded
            self.orchestrator.permissions.settings = reloaded
            rumps.notification("Adrien", "", "Settings reloaded.")

        def quit_app(self, _) -> None:
            self.orchestrator.stop()
            if self._server is not None:
                self._submit(self._server.stop())
            self._submit(self.orchestrator.shutdown())
            rumps.quit_application()

    setup_logging()
    AdrienStatusItem().run()


if __name__ == "__main__":  # pragma: no cover
    run_menu_bar()
