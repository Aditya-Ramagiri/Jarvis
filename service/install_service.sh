#!/usr/bin/env bash
#
# Install Adrien as a background service that starts at login (spec 3, 10).
#
# Installs a LaunchAgent rather than a LaunchDaemon on purpose: Adrien needs the
# user's audio session, keychain and GUI session, and a system daemon has none
# of those. It runs as you, at login, with no dock icon.
#
#   ./service/install_service.sh            # install and start
#   ./service/install_service.sh --restart  # reload after a code change
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.raidnxt.adrien"
AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$AGENTS_DIR/$LABEL.plist"
TEMPLATE="$PROJECT_ROOT/service/$LABEL.plist"

say()  { printf '  %s\n' "$*"; }
fail() { printf '\n  ✗ %s\n\n' "$*" >&2; exit 1; }

[[ "$(uname)" == "Darwin" ]] || fail "this installs a macOS LaunchAgent; you are on $(uname)"

printf '\nInstalling Adrien\n\n'

# --- Python -----------------------------------------------------------------
# Prefer the project's venv: launchd runs with a minimal PATH and would
# otherwise pick a system Python that has none of the dependencies.
if [[ -x "$PROJECT_ROOT/.venv/bin/python3" ]]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python3"
    say "using the project venv"
elif [[ -x "$PROJECT_ROOT/venv/bin/python3" ]]; then
    PYTHON="$PROJECT_ROOT/venv/bin/python3"
    say "using the project venv"
else
    PYTHON="$(command -v python3)" || fail "python3 not found"
    say "⚠ no venv found - using $PYTHON"
    say "  a venv is safer: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
fi

"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
    || fail "Python 3.11+ is required; $PYTHON is $("$PYTHON" -V)"
say "python: $PYTHON ($("$PYTHON" -V 2>&1))"

# --- Preflight --------------------------------------------------------------
[[ -f "$PROJECT_ROOT/.env" ]] \
    || fail ".env is missing - copy .env.example to .env and fill in your keys"

if ! "$PYTHON" -c "
import sys; sys.path.insert(0, '$PROJECT_ROOT')
from adrien.config import env_key_pool, load_env
load_env()
sys.exit(0 if env_key_pool('GROQ_API_KEY') else 1)
" 2>/dev/null; then
    fail "no GROQ_API_KEY_1 in .env - Adrien cannot think or hear without it"
fi
say "keys: found"

mkdir -p "$PROJECT_ROOT/logs" "$AGENTS_DIR"

# --- Unload any previous copy ----------------------------------------------
if launchctl list | grep -q "$LABEL"; then
    say "unloading the running service"
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null \
        || launchctl unload "$PLIST_PATH" 2>/dev/null || true
fi

# --- Write the plist --------------------------------------------------------
sed -e "s|__PROJECT_ROOT__|$PROJECT_ROOT|g" \
    -e "s|__PYTHON__|$PYTHON|g" \
    "$TEMPLATE" > "$PLIST_PATH"
chmod 644 "$PLIST_PATH"
say "installed $PLIST_PATH"

# --- Load -------------------------------------------------------------------
# bootstrap is the modern form; load is kept for older macOS.
launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH" 2>/dev/null \
    || launchctl load -w "$PLIST_PATH"
launchctl enable "gui/$(id -u)/$LABEL" 2>/dev/null || true

sleep 2
if launchctl list | grep -q "$LABEL"; then
    say "✓ Adrien is running, and will start at login"
else
    fail "the service did not start - check logs/adrien.err.log"
fi

cat <<EOF

  Permissions macOS will ask for on first use
  -------------------------------------------
  Grant these to the *Python binary above*, not to Terminal, or the service
  will be denied when launchd starts it:

    System Settings > Privacy & Security > Microphone
      -> required, or Adrien cannot hear anything

    System Settings > Privacy & Security > Accessibility
      -> required for Discord messaging and app control

    System Settings > Privacy & Security > Automation
      -> allow control of System Events and Discord

  Useful commands
  ---------------
    tail -f "$PROJECT_ROOT/logs/adrien.err.log"
    launchctl kickstart -k gui/$(id -u)/$LABEL     # restart
    ./service/uninstall_service.sh                 # remove

EOF
