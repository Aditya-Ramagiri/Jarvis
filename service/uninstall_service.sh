#!/usr/bin/env bash
#
# Remove the Adrien LaunchAgent. Leaves the code, the .env and everything
# Adrien remembers alone - this stops the service, it does not uninstall the
# project. Data lives in ~/Library/Application Support/Adrien.
#
set -euo pipefail

LABEL="com.raidnxt.adrien"
PLIST_PATH="$HOME/Library/LaunchAgents/$LABEL.plist"

printf '\nRemoving the Adrien service\n\n'

if launchctl list | grep -q "$LABEL"; then
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null \
        || launchctl unload "$PLIST_PATH" 2>/dev/null || true
    printf '  stopped\n'
fi

if [[ -f "$PLIST_PATH" ]]; then
    rm -f "$PLIST_PATH"
    printf '  removed %s\n' "$PLIST_PATH"
fi

cat <<EOF

  ✓ Adrien will no longer start at login.

  Its memory and settings are untouched:
    ~/Library/Application Support/Adrien

  To delete those too:
    rm -rf ~/Library/Application\\ Support/Adrien

EOF
