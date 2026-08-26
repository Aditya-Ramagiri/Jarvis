"""Gaming tools: the home Minecraft server, Modrinth, game launchers (spec 7.3).

Minecraft server control shells out to commands the user configures in `.env`
(`MINECRAFT_START_CMD` and friends). Adrien deliberately does not guess at
systemd units, docker containers or screen sessions - the user's server is set
up however it is set up, and a wrong guess here means a server that will not
come back up.

Modrinth uses its public REST API properly (spec section 3), with no key
required for search.
"""

from __future__ import annotations

from typing import Any

from adrien.config import env_int, env_str
from adrien.logging_setup import get_logger
from adrien.tools._shell import run
from adrien.tools.registry import ToolResult, tool

log = get_logger(__name__)

MODRINTH_API = "https://api.modrinth.com/v2"


def _server_command(action: str) -> tuple[str, str]:
    """Configured shell command for start/stop/restart, or an explanation."""
    command = env_str(f"MINECRAFT_{action.upper()}_CMD")
    if not command:
        return "", (
            f"there is no MINECRAFT_{action.upper()}_CMD in .env, so Adrien does not "
            f"know how to {action} the server"
        )
    return command, ""


def _control_server(action: str, timeout: float = 90.0) -> ToolResult:
    command, error = _server_command(action)
    if not command:
        return ToolResult.failure(error)
    # shell=True is intentional here and only here: the value is a command line
    # the user wrote in their own .env, not anything the model supplied.
    result = run(command, shell=True, timeout=timeout)
    if not result.ok:
        return ToolResult.failure(f"the {action} command failed: {result.output[:300]}")
    return ToolResult.success({"action": action, "output": result.output[:400]},
                              speak=f"{action}ing the Minecraft server")


@tool(category="gaming", destructive=True, confirm="Start the Minecraft server?", timeout=120.0)
def start_minecraft_server() -> ToolResult:
    """Start the home-hosted Minecraft server."""
    return _control_server("start")


@tool(category="gaming", irreversible=True,
      confirm="Stop the Minecraft server? Anyone playing will be disconnected.", timeout=120.0)
def stop_minecraft_server() -> ToolResult:
    """Stop the home-hosted Minecraft server, disconnecting anyone online."""
    return _control_server("stop")


@tool(category="gaming", irreversible=True,
      confirm="Restart the Minecraft server? Anyone playing will be disconnected.", timeout=180.0)
def restart_minecraft_server() -> ToolResult:
    """Restart the home-hosted Minecraft server."""
    return _control_server("restart")


@tool(category="gaming", timeout=15.0)
def check_minecraft_players_online(host: str = "", port: int = 0) -> ToolResult:
    """Check whether the Minecraft server is up and who is playing on it.

    Args:
        host: Server address. Defaults to the one configured in .env.
        port: Server port. Defaults to the configured one, or 25565.
    """
    address = host or env_str("MINECRAFT_HOST", "rhs.raidnxt.com")
    query_port = port or env_int("MINECRAFT_QUERY_PORT", 25565)

    try:
        from mcstatus import JavaServer
    except ImportError:
        return ToolResult.failure("mcstatus is not installed")

    try:
        server = JavaServer.lookup(f"{address}:{query_port}", timeout=8)
        status = server.status()
    except Exception as exc:
        return ToolResult.failure(
            f"the server at {address} did not answer ({type(exc).__name__}); it is probably down"
        )

    names: list[str] = []
    if status.players.sample:
        names = [player.name for player in status.players.sample]

    online = status.players.online
    if online == 0:
        speak = "the server is up, nobody is on"
    elif names:
        speak = f"{online} online: {', '.join(names[:6])}"
    else:
        speak = f"{online} player{'s' if online != 1 else ''} online"

    return ToolResult.success(
        {
            "host": address,
            "online": online,
            "max": status.players.max,
            "players": names,
            "version": status.version.name,
            "latency_ms": round(status.latency, 1),
        },
        speak=speak,
    )


@tool(category="gaming", timeout=20.0)
async def check_modrinth(query: str, project_type: str = "", limit: int = 5,
                         game_version: str = "") -> ToolResult:
    """Search Modrinth for a mod, modpack, resource pack or plugin, or look one
    up by name to see its latest version.

    Args:
        query: What to search for, e.g. a mod name or a topic.
        project_type: Narrow to mod, modpack, resourcepack, shader or plugin.
        limit: How many results to return.
        game_version: Only show results for this Minecraft version.
    """
    import json as _json

    from adrien.core.http import get_client

    facets: list[list[str]] = []
    if project_type:
        facets.append([f"project_type:{project_type}"])
    if game_version:
        facets.append([f"versions:{game_version}"])

    params: dict[str, Any] = {"query": query, "limit": max(1, min(limit, 10))}
    if facets:
        params["facets"] = _json.dumps(facets)

    try:
        response = await get_client().get(f"{MODRINTH_API}/search", params=params, timeout=15)
    except Exception as exc:
        return ToolResult.failure(f"could not reach Modrinth ({type(exc).__name__})")

    if response.status_code != 200:
        return ToolResult.failure(f"Modrinth returned {response.status_code}")

    hits = (response.json() or {}).get("hits") or []
    if not hits:
        return ToolResult.success({"results": []}, speak=f"nothing on Modrinth for {query}")

    results = [
        {
            "name": hit.get("title"),
            "slug": hit.get("slug"),
            "type": hit.get("project_type"),
            "downloads": hit.get("downloads"),
            "latest_version": (hit.get("versions") or [None])[-1],
            "description": (hit.get("description") or "")[:180],
        }
        for hit in hits
    ]
    top = results[0]
    return ToolResult.success(
        {"query": query, "results": results},
        speak=f"top hit is {top['name']}, {top['downloads']:,} downloads" if top.get("downloads")
        else f"top hit is {top['name']}",
    )


@tool(category="gaming", timeout=25.0)
async def check_modrinth_project_updates(slug: str, game_version: str = "") -> ToolResult:
    """Check the most recent release of a specific Modrinth project.

    Args:
        slug: The project's Modrinth slug or id.
        game_version: Only consider versions for this Minecraft version.
    """
    import json as _json

    from adrien.core.http import get_client

    params: dict[str, Any] = {}
    if game_version:
        params["game_versions"] = _json.dumps([game_version])

    try:
        response = await get_client().get(
            f"{MODRINTH_API}/project/{slug}/version", params=params, timeout=15
        )
    except Exception as exc:
        return ToolResult.failure(f"could not reach Modrinth ({type(exc).__name__})")

    if response.status_code == 404:
        return ToolResult.failure(f"Modrinth has no project called {slug}")
    if response.status_code != 200:
        return ToolResult.failure(f"Modrinth returned {response.status_code}")

    versions = response.json() or []
    if not versions:
        return ToolResult.success({"versions": []}, speak=f"{slug} has no matching releases")

    latest = versions[0]
    return ToolResult.success(
        {
            "project": slug,
            "latest": latest.get("version_number"),
            "name": latest.get("name"),
            "published": latest.get("date_published"),
            "game_versions": latest.get("game_versions"),
            "loaders": latest.get("loaders"),
        },
        speak=f"the latest {slug} release is {latest.get('version_number')}",
    )


# Common launchers, so "open Steam" does not depend on the LLM guessing the
# exact bundle name.
_LAUNCHER_ALIASES = {
    "steam": "Steam",
    "epic": "Epic Games Launcher",
    "epic games": "Epic Games Launcher",
    "minecraft": "Minecraft",
    "prism": "Prism Launcher",
    "prism launcher": "Prism Launcher",
    "modrinth": "Modrinth App",
    "battle.net": "Battle.net",
    "battlenet": "Battle.net",
    "gog": "GOG Galaxy",
}


@tool(category="gaming")
def open_game_launcher(name: str) -> ToolResult:
    """Open a game or game launcher, such as Steam or Prism Launcher.

    Args:
        name: The launcher or game to open.
    """
    from adrien.tools.system_tools import open_app

    return open_app(_LAUNCHER_ALIASES.get(name.strip().lower(), name))
