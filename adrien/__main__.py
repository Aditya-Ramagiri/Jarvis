"""Adrien's command line.

    python -m adrien run             # start the assistant (what launchd runs)
    python -m adrien chat            # type instead of talking - no mic needed
    python -m adrien doctor          # check keys, devices, permissions
    python -m adrien status          # provider and memory health
    python -m adrien devices         # list audio devices
    python -m adrien discover        # find Adrien on the LAN
    python -m adrien auth-google     # one-off Google OAuth consent
    python -m adrien memory          # inspect what Adrien remembers
    python -m adrien tools           # list registered tools

`doctor` is the one to run first on a new machine: it checks every dependency
and credential and says what is missing, rather than leaving you to find out
one failed tool at a time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from adrien.config import env_key_pool, env_str, load_env, settings
from adrien.logging_setup import setup_logging

OK = "✓"
BAD = "✗"
WARN = "!"


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
async def _run(with_server: bool = True) -> None:
    from adrien.core.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    server = None

    if with_server and settings().get("server.enabled", True):
        from adrien.server.ws_server import AdrienServer, local_ip

        server = AdrienServer(orchestrator)
        try:
            await server.start()
            print(f"clients can reach Adrien at ws://{local_ip()}:{server.port}")
        except Exception as exc:
            # A broken socket must not stop the voice assistant working on the
            # Mac itself, which is the primary use.
            print(f"{WARN} the client server could not start: {exc}", file=sys.stderr)
            server = None

    try:
        await orchestrator.run()
    finally:
        if server is not None:
            await server.stop()


def cmd_run(args) -> int:
    setup_logging(args.log_level)
    try:
        asyncio.run(_run(with_server=not args.no_server))
    except KeyboardInterrupt:
        print("\nstopped")
    except Exception as exc:
        # A background service should not report failure as a stack trace in a
        # log file. Say what broke and what to do about it; the traceback is
        # still in the log at DEBUG for anyone who wants it.
        print(f"\n{BAD} Adrien could not start: {exc}", file=sys.stderr)
        print("  Run 'python -m adrien doctor' to see what is missing.\n", file=sys.stderr)
        import logging

        logging.getLogger("adrien").debug("startup failed", exc_info=True)
        return 1
    return 0


# --------------------------------------------------------------------------
# chat - the whole brain, minus the microphone
# --------------------------------------------------------------------------
async def _chat() -> None:
    from adrien.core.orchestrator import Orchestrator

    orchestrator = Orchestrator()

    async def confirm(prompt: str) -> bool:
        from adrien.tools.permissions import interpret_confirmation

        answer = await asyncio.to_thread(input, f"\n{prompt} [y/N] ")
        return interpret_confirmation(answer) is True

    orchestrator.permissions.confirm_fn = confirm

    print("Adrien, typed. Ctrl-C or 'quit' to leave.\n")
    try:
        while True:
            text = await asyncio.to_thread(input, "you > ")
            if text.strip().lower() in ("quit", "exit"):
                break
            result = await orchestrator.handle_text(text, speak=False, source="cli")
            if result.tool_calls:
                print(f"      [{', '.join(result.tool_calls)}]")
            print(f"adrien > {result.reply}  ({result.latency_ms:.0f}ms)\n")
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        await orchestrator.shutdown()


def cmd_chat(args) -> int:
    setup_logging(args.log_level, to_file=False)
    asyncio.run(_chat())
    return 0


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------
def cmd_doctor(args) -> int:
    """Check every dependency and credential, and say what is missing."""
    load_env()
    problems = 0

    def report(ok: bool, label: str, detail: str = "", fatal: bool = True) -> None:
        nonlocal problems
        mark = OK if ok else (BAD if fatal else WARN)
        print(f" {mark} {label}{f' - {detail}' if detail else ''}")
        if not ok and fatal:
            problems += 1

    print("\nKeys")
    for label, prefix, fatal in (
        ("Groq (LLM + Whisper)", "GROQ_API_KEY", True),
        ("Gemini (fallback LLM)", "GEMINI_API_KEY", False),
        ("Fish Audio (voice)", "FISH_AUDIO_API_KEY", True),
    ):
        keys = env_key_pool(prefix)
        report(bool(keys), label, f"{len(keys)} key(s)" if keys else f"no {prefix}_1 in .env",
               fatal=fatal)
    report(bool(env_str("FISH_AUDIO_VOICE_ID")), "Fish Audio voice id",
           "" if env_str("FISH_AUDIO_VOICE_ID") else "unset - the default voice will be used",
           fatal=False)
    for label, name in (("GitHub", "GITHUB_TOKEN"),
                        ("Google OAuth", "GOOGLE_OAUTH_CLIENT_ID")):
        report(bool(env_str(name)), label, "" if env_str(name) else f"{name} unset", fatal=False)

    print("\nPython packages")
    for module, label, fatal in (
        ("httpx", "httpx", True),
        ("websockets", "websockets", True),
        ("sounddevice", "sounddevice (audio I/O)", True),
        ("numpy", "numpy", True),
        ("openwakeword", "openWakeWord (wake word)", True),
        ("webrtcvad", "webrtcvad (endpointing)", False),
        ("chromadb", "chromadb (semantic memory)", False),
        ("zeroconf", "zeroconf (client discovery)", False),
        ("github", "PyGithub", False),
        ("mcstatus", "mcstatus (Minecraft)", False),
        ("googleapiclient", "Google API client", False),
    ):
        try:
            __import__(module)
            report(True, label)
        except ImportError:
            report(False, label, "pip install -r requirements.txt", fatal=fatal)
        except Exception as exc:
            # Installed but unusable - sounddevice raises OSError when
            # PortAudio is missing, for instance. A diagnostic that dies on the
            # very problem it exists to report is worse than useless, so catch
            # everything and say what actually went wrong.
            report(False, label, f"installed but not usable: {exc}", fatal=fatal)

    print("\nAudio")
    try:
        import sounddevice as sd

        default_in, default_out = sd.default.device
        devices = sd.query_devices()
        report(True, "input", str(devices[default_in]["name"]) if default_in is not None else "?")
        report(True, "output", str(devices[default_out]["name"]) if default_out is not None else "?")
    except ImportError:
        report(False, "audio devices", "sounddevice is not installed")
    except Exception as exc:
        hint = str(exc)
        if "PortAudio" in hint:
            hint = "PortAudio is missing - install it with: brew install portaudio"
        report(False, "audio devices", hint)

    print("\nWake word")
    from pathlib import Path

    from adrien.config import PROJECT_ROOT

    model_path = PROJECT_ROOT / str(settings().get("wake_word.model_path", "models/adrien.onnx"))
    if Path(model_path).exists():
        report(True, "custom 'Adrien' model", str(model_path.name))
    else:
        report(False, "custom 'Adrien' model",
               f"not at {model_path} - falling back to "
               f"'{settings().get('wake_word.fallback_model')}'. See docs/WAKE_WORD.md",
               fatal=False)

    print("\nmacOS integration")
    from adrien.tools._shell import is_macos

    if not is_macos():
        report(False, "macOS", "not running on macOS - system and Discord tools are disabled",
               fatal=False)
    else:
        from adrien.tools.discord_automation import preflight

        checks = preflight()
        report(checks["accessibility_permission"], "Accessibility permission",
               checks.get("hint", ""), fatal=False)
        report(checks["discord_installed"], "Discord installed", fatal=False)

    print("\nTools")
    from adrien.tools.registry import load_all_tools

    registry = load_all_tools()
    report(len(registry) > 0, f"{len(registry)} tools registered",
           ", ".join(f"{k}:{len(v)}" for k, v in registry.by_category().items()))

    print(f"\n{OK} ready" if problems == 0 else f"\n{BAD} {problems} problem(s) to fix")
    return 1 if problems else 0


# --------------------------------------------------------------------------
# smaller commands
# --------------------------------------------------------------------------
def cmd_models(args) -> int:
    """List the models each provider will actually serve this key.

    Groq retires ids without warning and offers no moving alias, so "which
    models can I really use" is a question worth being able to answer directly
    rather than inferring from a failed turn.
    """
    load_env()

    async def run() -> int:
        from adrien.core.providers.groq import GroqProvider, rank_candidate

        keys = env_key_pool("GROQ_API_KEY")
        if not keys:
            print(f"{BAD} no Groq keys in .env")
            return 1

        provider = GroqProvider()
        available = await provider._discover(keys[0])
        if not available:
            print(f"{BAD} could not list Groq models")
            return 1

        print(f"\nGroq offers {len(available)} model(s) to this key:\n")
        for name in sorted(available, key=rank_candidate):
            print(f"  {name}")

        print("\nAdrien would choose:")
        for tier in ("fast", "smart"):
            print(f"  {tier:6} -> {await provider.resolve_model(tier, keys[0])}")
        print("\nPin one by setting GROQ_FAST_MODEL / GROQ_SMART_MODEL in .env.\n")
        return 0

    return asyncio.run(run())


def cmd_status(args) -> int:
    load_env()
    from adrien.core.llm_router import LLMRouter
    from adrien.memory.manager import MemoryManager

    print(json.dumps({
        "providers": LLMRouter().status(),
        "memory": MemoryManager().stats(),
    }, indent=2))
    return 0


def cmd_devices(args) -> int:
    from adrien.core.audio import list_devices

    for device in list_devices():
        kind = "in " if device["inputs"] else "   "
        kind += "out" if device["outputs"] else "   "
        print(f"  [{device['index']:2}] {kind}  {device['name']}")
    print("\nSet audio.input_device / audio.output_device in config/settings.json "
          "to pin one.")
    return 0


def cmd_discover(args) -> int:
    from adrien.server.discovery import discover

    found = discover(timeout=args.timeout)
    for service in found:
        print(f"{service['host']}:{service['port']}  {service['properties']}")
    if not found:
        print("nothing found - is the service running on this network?")
    return 0 if found else 1


def cmd_auth_google(args) -> int:
    load_env()
    from adrien.tools.productivity_tools import run_google_oauth_flow

    ok, message = run_google_oauth_flow()
    print(f"{OK if ok else BAD} {message}")
    return 0 if ok else 1


def cmd_memory(args) -> int:
    load_env()
    from adrien.memory.manager import MemoryManager

    memory = MemoryManager()
    if args.query:
        recalled = memory.recall(args.query)
        print(recalled.as_prompt() or "nothing relevant")
        return 0

    facts = memory.known_facts(category=args.category)
    if not facts:
        print("no facts stored yet")
        return 0
    for fact in facts:
        print(f"  [{fact.category}] {fact.as_sentence()}")
    print(f"\n{len(facts)} fact(s); {memory.vectors.count()} vector(s)")
    return 0


def cmd_tools(args) -> int:
    load_env()
    from adrien.tools.permissions import PermissionManager
    from adrien.tools.registry import load_all_tools

    registry = load_all_tools()
    permissions = PermissionManager()
    for category, names in registry.by_category().items():
        print(f"\n{category}")
        for name in names:
            spec = registry.get(name)
            mode = permissions.mode_for(spec)
            flag = " (confirms)" if mode == "confirm" else " (off)" if mode == "deny" else ""
            print(f"  {name}{flag}")
            if args.verbose:
                print(f"      {spec.description}")
    print(f"\n{len(registry)} tools")
    return 0


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="adrien", description="Adrien voice assistant")
    parser.add_argument("--log-level", default="", help="DEBUG, INFO, WARNING, ERROR")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="start the assistant")
    run.add_argument("--no-server", action="store_true", help="skip the client server")
    run.set_defaults(func=cmd_run)

    subparsers.add_parser("chat", help="type instead of talking").set_defaults(func=cmd_chat)
    subparsers.add_parser("doctor", help="check the setup").set_defaults(func=cmd_doctor)
    subparsers.add_parser("status", help="provider and memory health").set_defaults(func=cmd_status)
    subparsers.add_parser("models", help="list the models your keys can use").set_defaults(
        func=cmd_models)
    subparsers.add_parser("devices", help="list audio devices").set_defaults(func=cmd_devices)
    subparsers.add_parser("auth-google", help="authorise Calendar and Gmail").set_defaults(
        func=cmd_auth_google)

    discover_parser = subparsers.add_parser("discover", help="find Adrien on the LAN")
    discover_parser.add_argument("--timeout", type=float, default=3.0)
    discover_parser.set_defaults(func=cmd_discover)

    memory_parser = subparsers.add_parser("memory", help="inspect long-term memory")
    memory_parser.add_argument("--query", default="", help="search instead of listing")
    memory_parser.add_argument("--category", default="", help="filter by category")
    memory_parser.set_defaults(func=cmd_memory)

    tools_parser = subparsers.add_parser("tools", help="list registered tools")
    tools_parser.add_argument("-v", "--verbose", action="store_true")
    tools_parser.set_defaults(func=cmd_tools)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
