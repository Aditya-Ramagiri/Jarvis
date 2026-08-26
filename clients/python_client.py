#!/usr/bin/env python3
"""A complete Adrien client, in one file.

Doubles as the protocol's reference implementation and as the way to test the
server without a phone in your hand:

    python clients/python_client.py --text "what's the weather"
    python clients/python_client.py --listen          # push to talk
    python clients/python_client.py --status
    python clients/python_client.py --discover

Everything the Android and iPad clients do is here, minus the platform
scaffolding: browse mDNS, hand over the token, stream 16 kHz PCM up, play
24 kHz PCM back, and answer confirmation prompts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adrien.config import env_str, load_env  # noqa: E402
from adrien.server.protocol import MessageType, decode, encode  # noqa: E402

CLIENT_RATE = 16_000
CHUNK_SAMPLES = 1600  # 100 ms


async def connect(host: str, port: int, token: str, device: str = "python-client"):
    import websockets

    url = f"ws://{host}:{port}/"
    print(f"connecting to {url}")
    websocket = await websockets.connect(url, max_size=2 ** 20)
    await websocket.send(encode(
        MessageType.HELLO, token=token, device=device, platform=sys.platform
    ))

    frame = decode(await websocket.recv())
    if frame is None or frame.type != MessageType.WELCOME:
        reason = frame.get("reason") if frame else "no reply"
        raise SystemExit(f"handshake refused: {reason}")
    print(f"connected to {frame.get('assistant')} (protocol {frame.get('protocol')})")
    return websocket


async def pump_replies(websocket, play: bool = True) -> None:
    """Read frames until the turn ends, playing any audio that arrives."""
    import numpy as np
    import sounddevice as sd

    stream = None
    try:
        while True:
            message = await websocket.recv()

            if isinstance(message, bytes):
                if stream is not None:
                    stream.write(np.frombuffer(message, dtype=np.int16))
                continue

            frame = decode(message)
            if frame is None:
                continue

            if frame.type == MessageType.REPLY:
                print(f"\nAdrien: {frame.get('text')}")
                if frame.get("tools"):
                    print(f"  (used: {', '.join(frame.get('tools'))})")

            elif frame.type == MessageType.CONFIRM:
                # Never auto-answer: this is the whole point of the layer.
                print(f"\nAdrien asks: {frame.get('prompt')}")
                answer = await asyncio.to_thread(input, "yes/no > ")
                await websocket.send(encode(MessageType.TEXT, text=answer))

            elif frame.type == MessageType.AUDIO_START and play:
                rate = int(frame.get("sample_rate", 24_000))
                stream = sd.OutputStream(samplerate=rate, channels=1, dtype="int16")
                stream.start()

            elif frame.type == MessageType.AUDIO_END and stream is not None:
                stream.stop()
                stream.close()
                stream = None

            elif frame.type == MessageType.STATE:
                state = frame.get("state")
                print(f"[{state}]", end="\r" if state == "thinking" else "\n")
                if state == "idle":
                    return

            elif frame.type == MessageType.STATUS_REPORT:
                print(json.dumps(frame.data, indent=2))
                return

            elif frame.type == MessageType.ERROR:
                print(f"error: {frame.get('reason')}")
                if frame.get("fatal"):
                    return
    finally:
        if stream is not None:
            stream.stop()
            stream.close()


async def send_text(websocket, text: str, want_audio: bool) -> None:
    await websocket.send(encode(MessageType.TEXT, text=text, want_audio=want_audio))
    await pump_replies(websocket, play=want_audio)


async def push_to_talk(websocket) -> None:
    """Record while the user holds the terminal open, then send."""
    import sounddevice as sd

    print("\nrecording - press Enter to stop")
    frames: list[bytes] = []

    def callback(indata, count, timing, status):  # noqa: ARG001
        frames.append(bytes(indata))

    await websocket.send(encode(MessageType.AUDIO_START, want_audio=True))
    with sd.RawInputStream(samplerate=CLIENT_RATE, blocksize=CHUNK_SAMPLES,
                           channels=1, dtype="int16", callback=callback):
        await asyncio.to_thread(input)

    for chunk in frames:
        await websocket.send(chunk)
    await websocket.send(encode(MessageType.AUDIO_END))
    print(f"sent {sum(len(c) for c in frames) / (CLIENT_RATE * 2):.1f}s of audio")
    await pump_replies(websocket)


def resolve_server(args) -> tuple[str, int]:
    if args.host:
        return args.host, args.port

    from adrien.server.discovery import discover

    print("looking for Adrien on the network...")
    found = discover(timeout=args.timeout)
    if not found:
        raise SystemExit(
            "Adrien unavailable - nothing advertising on this network.\n"
            "Check the Mac is awake, on the same WiFi, and running the service."
        )
    first = found[0]
    print(f"found {first['name']} at {first['host']}:{first['port']}")
    return first["host"], first["port"]


async def amain(args) -> None:
    if args.discover:
        from adrien.server.discovery import discover

        services = discover(timeout=args.timeout)
        for service in services:
            print(f"{service['host']}:{service['port']}  {service['properties']}")
        if not services:
            print("nothing found - Adrien unavailable on this network")
        return

    host, port = resolve_server(args)
    token = args.token or env_str("ADRIEN_WS_TOKEN")
    websocket = await connect(host, port, token)

    try:
        if args.status:
            await websocket.send(encode(MessageType.STATUS))
            await pump_replies(websocket, play=False)
        elif args.text:
            await send_text(websocket, args.text, want_audio=not args.no_audio)
        elif args.listen:
            while True:
                await asyncio.to_thread(input, "\npress Enter to talk (ctrl-c to quit) ")
                await push_to_talk(websocket)
        else:
            while True:
                text = await asyncio.to_thread(input, "\nyou > ")
                if text.strip() in ("quit", "exit"):
                    break
                await send_text(websocket, text, want_audio=not args.no_audio)
    finally:
        await websocket.close()


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="Adrien reference client")
    parser.add_argument("--host", help="skip mDNS and connect straight to this host")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default="", help="defaults to ADRIEN_WS_TOKEN")
    parser.add_argument("--text", help="send one message and exit")
    parser.add_argument("--listen", action="store_true", help="push-to-talk mode")
    parser.add_argument("--status", action="store_true", help="print server status and exit")
    parser.add_argument("--discover", action="store_true", help="list Adriens on the LAN")
    parser.add_argument("--no-audio", action="store_true", help="text replies only")
    parser.add_argument("--timeout", type=float, default=3.0, help="mDNS browse seconds")
    args = parser.parse_args()

    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
