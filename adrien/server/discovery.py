"""mDNS/Bonjour advertisement, so clients never ask for an IP address.

Spec section 8 is explicit that a client should find the Mac by itself. The
Mac's LAN address changes with DHCP leases and network moves, so an IP typed
into a phone once is an IP that is wrong a fortnight later.

The service advertises as `_adrien._tcp.local.` with a TXT record carrying the
protocol version and assistant name, so a client can tell a compatible brain
from an incompatible one before it connects.
"""

from __future__ import annotations

import socket
from typing import Any

from adrien.logging_setup import get_logger
from adrien.server.protocol import MDNS_SERVICE_TYPE, PROTOCOL_VERSION

log = get_logger(__name__)


class ServiceAdvertiser:
    """Publishes Adrien on the local network over mDNS."""

    def __init__(self, port: int, name: str = "Adrien") -> None:
        self.port = port
        self.name = name
        self._zeroconf: Any = None
        self._info: Any = None

    def start(self) -> bool:
        """Register the service. Returns False if mDNS is unavailable.

        A failure here is not fatal - clients can still be pointed at an
        address by hand - so it logs and carries on.
        """
        try:
            from zeroconf import ServiceInfo, Zeroconf
        except ImportError:
            log.warning("zeroconf is not installed; clients will need a manual address")
            return False

        try:
            hostname = socket.gethostname().split(".")[0]
            addresses = [socket.inet_aton(ip) for ip in _local_addresses()]
            if not addresses:
                log.warning("no local IPv4 address found; skipping mDNS")
                return False

            self._info = ServiceInfo(
                MDNS_SERVICE_TYPE,
                f"{self.name} on {hostname}.{MDNS_SERVICE_TYPE}",
                addresses=addresses,
                port=self.port,
                properties={
                    "assistant": self.name,
                    "protocol": str(PROTOCOL_VERSION),
                    "path": "/",
                },
                server=f"{hostname}.local.",
            )
            self._zeroconf = Zeroconf()
            self._zeroconf.register_service(self._info)
            log.info("advertising %s on %s:%d via mDNS", self.name, hostname, self.port)
            return True
        except Exception as exc:
            log.warning("could not advertise over mDNS: %s", exc)
            self.stop()
            return False

    def stop(self) -> None:
        try:
            if self._zeroconf is not None and self._info is not None:
                self._zeroconf.unregister_service(self._info)
            if self._zeroconf is not None:
                self._zeroconf.close()
        except Exception as exc:  # pragma: no cover
            log.debug("mDNS shutdown was untidy: %s", exc)
        finally:
            self._zeroconf = None
            self._info = None


def _local_addresses() -> list[str]:
    """Every private IPv4 address on this machine."""
    import ipaddress

    found: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if address not in found:
                found.append(address)
    except socket.gaierror:
        pass

    if not found:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("192.168.1.1", 80))
                found.append(probe.getsockname()[0])
        except OSError:
            pass

    return [
        address for address in found
        if not ipaddress.ip_address(address).is_loopback
    ]


def discover(timeout: float = 3.0) -> list[dict[str, Any]]:
    """Find Adrien on the LAN. Used by `adrien discover` and the test clients."""
    try:
        from zeroconf import ServiceBrowser, Zeroconf
    except ImportError:
        return []

    found: list[dict[str, Any]] = []

    class Listener:
        def add_service(self, zeroconf, service_type, name):  # noqa: ANN001
            info = zeroconf.get_service_info(service_type, name, timeout=int(timeout * 1000))
            if not info:
                return
            for raw in info.addresses:
                found.append({
                    "name": name,
                    "host": socket.inet_ntoa(raw),
                    "port": info.port,
                    "properties": {
                        key.decode(): value.decode()
                        for key, value in (info.properties or {}).items()
                        if key and value
                    },
                })

        def update_service(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            """Required by the ServiceBrowser interface; nothing to do."""

        def remove_service(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            """Required by the ServiceBrowser interface; nothing to do."""

    zeroconf = Zeroconf()
    browser = ServiceBrowser(zeroconf, MDNS_SERVICE_TYPE, Listener())
    try:
        import time

        time.sleep(timeout)
    finally:
        browser.cancel()
        zeroconf.close()
    return found
