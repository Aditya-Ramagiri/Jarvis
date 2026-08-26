"""LAN-only WebSocket server for the phone and iPad clients (spec section 8).

Scope, restated because it is a deliberate limit and not an oversight: this
binds to the local network and nothing else. There is no tunnel, no port
forwarding, no cloud relay, and no attempt to reach the Mac from outside the
house. If the Mac is asleep or the client is on mobile data, the correct
behaviour is for the client to say "Adrien unavailable" - which the protocol
supports and the clients implement.

Two things still guard the socket, because "local network" is not the same as
"trusted network":

* a shared token every client presents in its `hello` frame, compared in
  constant time;
* a bind-address check that refuses to start if it would be reachable beyond
  the LAN.

Clients stay thin: they capture audio and play audio. Every piece of thinking -
STT, LLM, tools, memory, TTS - happens here, so a new client platform is a
microphone and a speaker, not a second implementation of Adrien.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import ipaddress
import socket
import time
from dataclasses import dataclass, field
from typing import Any

from adrien.config import Settings, env_int, env_str
from adrien.config import settings as global_settings
from adrien.core.orchestrator import Orchestrator
from adrien.logging_setup import get_logger
from adrien.server.protocol import (
    CLIENT_SAMPLE_RATE,
    MAX_UTTERANCE_BYTES,
    SERVER_SAMPLE_RATE,
    MessageType,
    decode,
    encode,
    error,
    welcome,
)

log = get_logger(__name__)


@dataclass
class ClientSession:
    """Per-connection state."""

    device: str = "unknown"
    platform: str = ""
    authenticated: bool = False
    receiving_audio: bool = False
    buffer: bytearray = field(default_factory=bytearray)
    want_audio: bool = True
    connected_at: float = field(default_factory=time.time)
    # Set while a confirmation is outstanding, so the client's next text frame
    # is read as the answer rather than as a new request.
    pending_confirmation: asyncio.Future | None = None

    @property
    def label(self) -> str:
        return f"{self.device}{f'/{self.platform}' if self.platform else ''}"


class AdrienServer:
    """WebSocket front end onto a running `Orchestrator`."""

    def __init__(self, orchestrator: Orchestrator, settings: Settings | None = None) -> None:
        self.orchestrator = orchestrator
        self.settings = settings or global_settings()
        self.host = env_str("ADRIEN_WS_HOST", "0.0.0.0")
        self.port = env_int("ADRIEN_WS_PORT", 8765)
        self.token = env_str("ADRIEN_WS_TOKEN")
        self.max_clients = int(self.settings.get("server.max_clients", 8))
        self.sessions: dict[Any, ClientSession] = {}
        self._server = None
        self._advertiser = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        import websockets

        self._assert_local_only()
        if not self.token:
            log.warning(
                "ADRIEN_WS_TOKEN is empty - any device on the WiFi could talk to "
                "Adrien. Generate one with: openssl rand -hex 24"
            )

        self._server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            max_size=2 ** 20,        # 1 MiB frames: plenty for PCM chunks
            ping_interval=20,        # notice a phone that walked out of range
            ping_timeout=20,
        )
        log.info("websocket server listening on %s:%d", self.host, self.port)

        if self.settings.get("server.advertise_mdns", True):
            from adrien.server.discovery import ServiceAdvertiser

            self._advertiser = ServiceAdvertiser(self.port)
            await asyncio.to_thread(self._advertiser.start)

    def _assert_local_only(self) -> None:
        """Refuse to bind somewhere that is not the local network.

        Spec 8 makes local-only a design decision rather than a default, so
        this fails loudly instead of quietly listening on a public interface.
        """
        if self.host in ("0.0.0.0", "::", "localhost", "127.0.0.1"):
            return  # 0.0.0.0 is every *local* interface; the router is the edge
        try:
            address = ipaddress.ip_address(self.host)
        except ValueError as exc:
            raise RuntimeError(
                f"ADRIEN_WS_HOST={self.host!r} is not an IP address"
            ) from exc
        if not (address.is_private or address.is_loopback or address.is_link_local):
            raise RuntimeError(
                f"refusing to bind to the public address {self.host} - Adrien is "
                "local-network only by design (spec section 8)"
            )

    async def stop(self) -> None:
        if self._advertiser is not None:
            await asyncio.to_thread(self._advertiser.stop)
            self._advertiser = None
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        log.info("websocket server stopped")

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------
    async def _handle_client(self, websocket) -> None:
        import websockets

        if len(self.sessions) >= self.max_clients:
            await websocket.send(error("too many clients connected", fatal=True))
            await websocket.close()
            return

        session = ClientSession()
        self.sessions[websocket] = session
        peer = getattr(websocket, "remote_address", ("?",))[0]
        log.info("client connected from %s", peer)

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    await self._on_binary(websocket, session, message)
                else:
                    await self._on_text(websocket, session, message)
        except websockets.exceptions.ConnectionClosed:
            log.info("client %s disconnected", session.label)
        except Exception:
            log.exception("client handler failed")
        finally:
            if session.pending_confirmation and not session.pending_confirmation.done():
                session.pending_confirmation.set_result(False)
            self.sessions.pop(websocket, None)

    async def _on_text(self, websocket, session: ClientSession, raw: str) -> None:
        frame = decode(raw)
        if frame is None:
            await websocket.send(error("could not parse that frame"))
            return

        if frame.type == MessageType.HELLO:
            await self._on_hello(websocket, session, frame)
            return

        if not session.authenticated:
            await websocket.send(error("send a hello frame first", fatal=True))
            await websocket.close()
            return

        if frame.type == MessageType.PING:
            await websocket.send(encode(MessageType.PONG, at=time.time()))

        elif frame.type == MessageType.STATUS:
            await websocket.send(
                encode(MessageType.STATUS_REPORT, **self.orchestrator.status())
            )

        elif frame.type == MessageType.AUDIO_START:
            session.receiving_audio = True
            session.buffer.clear()
            session.want_audio = bool(frame.get("want_audio", True))

        elif frame.type == MessageType.AUDIO_END:
            session.receiving_audio = False
            await self._process_utterance(websocket, session)

        elif frame.type == MessageType.TEXT:
            text = str(frame.get("text") or "")
            # A pending confirmation claims the next text frame: the user is
            # answering a question, not starting a new request.
            if session.pending_confirmation and not session.pending_confirmation.done():
                from adrien.tools.permissions import interpret_confirmation

                session.pending_confirmation.set_result(
                    interpret_confirmation(text) is True
                )
                return
            session.want_audio = bool(frame.get("want_audio", True))
            await self._process_text(websocket, session, text)

        elif frame.type == MessageType.CANCEL:
            self.orchestrator.speaker.stop()
            session.buffer.clear()
            session.receiving_audio = False

        else:
            await websocket.send(error(f"unknown frame type {frame.type!r}"))

    async def _on_hello(self, websocket, session: ClientSession, frame) -> None:
        supplied = str(frame.get("token") or "")
        # Constant-time compare: a token check that leaks timing on a LAN is
        # a small hole, but it is a free one to close.
        if self.token and not hmac.compare_digest(supplied, self.token):
            log.warning("rejected a client with a bad token")
            await websocket.send(error("bad token", fatal=True))
            await websocket.close()
            return

        session.authenticated = True
        session.device = str(frame.get("device") or "unknown")[:64]
        session.platform = str(frame.get("platform") or "")[:32]
        log.info("client %s authenticated", session.label)
        await websocket.send(welcome(
            assistant=str(self.settings.get("assistant.name", "Adrien")),
            wake_word_needed=False,
        ))

    async def _on_binary(self, websocket, session: ClientSession, chunk: bytes) -> None:
        if not session.authenticated:
            return
        if not session.receiving_audio:
            log.debug("dropping audio that arrived outside an utterance")
            return
        if len(session.buffer) + len(chunk) > MAX_UTTERANCE_BYTES:
            session.receiving_audio = False
            session.buffer.clear()
            await websocket.send(error("that utterance was too long"))
            return
        session.buffer.extend(chunk)

    # ------------------------------------------------------------------
    # Turns
    # ------------------------------------------------------------------
    async def _process_utterance(self, websocket, session: ClientSession) -> None:
        pcm = bytes(session.buffer)
        session.buffer.clear()
        if not pcm:
            await websocket.send(error("no audio arrived"))
            return

        await websocket.send(encode(MessageType.STATE, state="thinking"))
        with self._confirmation_channel(websocket, session):
            result, audio = await self.orchestrator.handle_remote_audio(
                pcm, sample_rate=CLIENT_SAMPLE_RATE, source=session.label
            )

        if result.error and not result.reply:
            await websocket.send(error(result.error))
            await websocket.send(encode(MessageType.STATE, state="idle"))
            return

        await websocket.send(encode(MessageType.REPLY, text=result.reply,
                                    tools=result.tool_calls))
        if session.want_audio and audio:
            await self._send_audio(websocket, audio)
        await websocket.send(encode(MessageType.STATE, state="idle"))

    async def _process_text(self, websocket, session: ClientSession, text: str) -> None:
        if not text.strip():
            return
        await websocket.send(encode(MessageType.STATE, state="thinking"))
        with self._confirmation_channel(websocket, session):
            result = await self.orchestrator.handle_text(
                text, speak=False, source=session.label
            )

        await websocket.send(encode(MessageType.REPLY, text=result.reply,
                                    tools=result.tool_calls))
        if session.want_audio and result.reply:
            audio = await self.orchestrator.tts.synthesize(result.reply)
            if audio:
                await self._send_audio(websocket, audio)
        await websocket.send(encode(MessageType.STATE, state="idle"))

    async def _send_audio(self, websocket, pcm: bytes, chunk: int = 16_000) -> None:
        """Stream PCM back in chunks so the client can start playing early."""
        await websocket.send(encode(MessageType.AUDIO_START,
                                    sample_rate=SERVER_SAMPLE_RATE, bytes=len(pcm)))
        for start in range(0, len(pcm), chunk):
            await websocket.send(pcm[start:start + chunk])
        await websocket.send(encode(MessageType.AUDIO_END))

    @contextlib.contextmanager
    def _confirmation_channel(self, websocket, session: ClientSession):
        """Route confirmation questions to *this* client for the turn.

        Without this a destructive tool asked for from the phone would try to
        confirm through the Mac's speaker, where nobody is standing.
        """
        original = self.orchestrator.permissions.confirm_fn

        async def confirm_via_client(prompt: str) -> bool:
            loop = asyncio.get_running_loop()
            future: asyncio.Future = loop.create_future()
            session.pending_confirmation = future
            await websocket.send(encode(MessageType.CONFIRM, prompt=prompt))
            try:
                return await asyncio.wait_for(future, timeout=45)
            except TimeoutError:
                log.info("client %s never answered the confirmation", session.label)
                return False
            finally:
                session.pending_confirmation = None

        self.orchestrator.permissions.confirm_fn = confirm_via_client
        try:
            yield
        finally:
            self.orchestrator.permissions.confirm_fn = original

    # ------------------------------------------------------------------
    async def broadcast(self, message: str) -> None:
        """Push a frame to every authenticated client (reminders, state)."""
        for websocket, session in list(self.sessions.items()):
            if not session.authenticated:
                continue
            try:
                await websocket.send(message)
            except Exception:
                self.sessions.pop(websocket, None)


def local_ip() -> str:
    """Best guess at this machine's LAN address, for logs and QR pairing."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            # No packet is actually sent; this just asks the routing table
            # which interface would be used to reach the outside world.
            probe.connect(("192.168.1.1", 80))
            return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
