#!/usr/bin/env bash
#
# Start Adrien.
#
#   ./start.sh                 set everything up if needed, then run
#   ./start.sh --fast          skip the dependency check (faster restarts)
#   ./start.sh --chat          type instead of talking (no microphone needed)
#   ./start.sh --doctor        just run the diagnostics and exit
#   ./start.sh --service       install as a background service that starts at login
#   ./start.sh --no-server     run without the phone/iPad server
#
# Safe to run repeatedly: it only does the work that still needs doing.
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(pwd)"
VENV="$ROOT/.venv"
STAMP="$VENV/.deps-installed"

# --- output helpers ---------------------------------------------------------
if [[ -t 1 ]]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'
    YELLOW=$'\033[33m'; RESET=$'\033[0m'
else
    BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; RESET=""
fi

step() { printf '%s==>%s %s\n' "$BOLD" "$RESET" "$*"; }
ok()   { printf '  %s✓%s %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '  %s!%s %s\n' "$YELLOW" "$RESET" "$*"; }
die()  { printf '\n  %s✗ %s%s\n\n' "$RED" "$*" "$RESET" >&2; exit 1; }

# --- arguments --------------------------------------------------------------
MODE="run"
FAST=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast)       FAST=1 ;;
        --chat)       MODE="chat" ;;
        --doctor)     MODE="doctor" ;;
        --service)    MODE="service" ;;
        --no-server)  EXTRA_ARGS+=("--no-server") ;;
        --debug)      EXTRA_ARGS+=("--log-level" "DEBUG") ;;
        -h|--help)    sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^#//'; exit 0 ;;
        *)            die "unknown option: $1  (try --help)" ;;
    esac
    shift
done

printf '\n%sAdrien%s\n\n' "$BOLD" "$RESET"

# --- 1. Python --------------------------------------------------------------
step "Checking Python"

PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
        PYTHON="$(command -v "$candidate")"
        break
    fi
done

if [[ -z "$PYTHON" ]]; then
    printf '\n'
    warn "Adrien needs Python 3.11 or newer."
    if [[ "$(uname)" == "Darwin" ]]; then
        printf '  Install it with:  %sbrew install python@3.12%s\n' "$BOLD" "$RESET"
    fi
    die "no suitable Python found"
fi
ok "$("$PYTHON" -V) at $PYTHON"

# --- 2. PortAudio (macOS) ---------------------------------------------------
# sounddevice binds to PortAudio. Without it the wheel installs happily and
# then fails at runtime with an opaque OSError, so check it up front.
if [[ "$(uname)" == "Darwin" && "$MODE" != "chat" && "$MODE" != "doctor" ]]; then
    step "Checking audio libraries"
    if [[ -f /opt/homebrew/lib/libportaudio.dylib || -f /usr/local/lib/libportaudio.dylib ]]; then
        ok "PortAudio present"
    elif command -v brew >/dev/null 2>&1; then
        warn "PortAudio missing - installing it (needed for the microphone)"
        brew install portaudio || die "could not install PortAudio. Try: brew install portaudio"
        ok "PortAudio installed"
    else
        warn "PortAudio not found and Homebrew is not installed."
        warn "The microphone will not work until you install it:"
        printf '    %s/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"%s\n' "$DIM" "$RESET"
        printf '    %sbrew install portaudio%s\n' "$DIM" "$RESET"
    fi
fi

# --- 3. Virtual environment -------------------------------------------------
step "Preparing the environment"

if [[ ! -d "$VENV" ]]; then
    "$PYTHON" -m venv "$VENV" || die "could not create a virtualenv at $VENV"
    ok "created .venv"
else
    ok "using existing .venv"
fi

VPY="$VENV/bin/python"
[[ -x "$VPY" ]] || die ".venv looks broken - delete it and run ./start.sh again"

# --- 4. Dependencies --------------------------------------------------------
# Skipped when requirements.txt has not changed since the last install, so a
# normal restart does not pay for a full dependency resolve.
NEED_INSTALL=1
if [[ "$FAST" == "1" ]]; then
    NEED_INSTALL=0
elif [[ -f "$STAMP" ]] && [[ "$STAMP" -nt "$ROOT/requirements.txt" ]]; then
    NEED_INSTALL=0
fi

if [[ "$NEED_INSTALL" == "1" ]]; then
    step "Installing dependencies (a few minutes the first time)"
    "$VPY" -m pip install --quiet --upgrade pip setuptools wheel \
        || warn "could not upgrade pip - carrying on"

    # Pass 1: the must-haves, as one transaction. A failure here is fatal.
    if ! "$VPY" -m pip install --quiet -r "$ROOT/requirements.txt"; then
        printf '\n'
        die "could not install the core dependencies (see the error above)"
    fi
    ok "core dependencies installed"

    # Pass 2: the optional ones, ONE AT A TIME so a single unresolvable wheel
    # costs that one feature instead of the entire install. This is not
    # hypothetical - openwakeword's tflite-runtime dependency has no wheel for
    # Python 3.11+, and installing everything as one transaction meant a single
    # failure left nothing installed at all.
    step "Installing optional features"
    FAILED=()
    if [[ ! -f "$ROOT/requirements-optional.txt" ]]; then
        # An older checkout, or a partial pull. The core set is already in, so
        # warn and carry on rather than dying on a redirect error.
        warn "requirements-optional.txt is missing - skipping optional features"
        warn "run 'git pull' to get it"
    else
    while IFS= read -r requirement; do
        [[ -z "$requirement" || "$requirement" == \#* ]] && continue
        name="${requirement%%[<>=;]*}"
        name="$(printf '%s' "$name" | tr -d '[:space:]')"

        # openwakeword declares tflite-runtime, which Adrien never uses - it
        # runs the ONNX path. Its real needs are already installed.
        if [[ "$name" == "openwakeword" ]]; then
            if "$VPY" -m pip install --quiet --no-deps "$requirement" 2>/dev/null; then
                ok "$name"
            else
                FAILED+=("$name")
            fi
            continue
        fi

        if "$VPY" -m pip install --quiet "$requirement" 2>/dev/null; then
            ok "$name"
        else
            FAILED+=("$name")
        fi
    done < "$ROOT/requirements-optional.txt"
    fi

    if [[ ${#FAILED[@]} -gt 0 ]]; then
        printf '\n'
        warn "could not install: ${FAILED[*]}"
        warn "Adrien still runs; those features report themselves unavailable."
        printf '    %s./start.sh again will retry them%s\n' "$DIM" "$RESET"
    fi
    touch "$STAMP"
else
    ok "dependencies up to date (--fast, or requirements.txt unchanged)"
fi

# --- 5. Configuration -------------------------------------------------------
step "Checking configuration"

if [[ ! -f "$ROOT/config/settings.json" ]]; then
    cp "$ROOT/config/settings.example.json" "$ROOT/config/settings.json"
    ok "created config/settings.json"
else
    ok "config/settings.json present"
fi

if [[ ! -f "$ROOT/.env" ]]; then
    cp "$ROOT/.env.example" "$ROOT/.env"
    printf '\n'
    warn "Created .env from the template - it has no keys in it yet."
    printf '\n  Open it and fill in at least these three:\n\n'
    printf '    %sGROQ_API_KEY_1%s        console.groq.com\n' "$BOLD" "$RESET"
    printf '    %sFISH_AUDIO_API_KEY_1%s  fish.audio\n' "$BOLD" "$RESET"
    printf '    %sFISH_AUDIO_VOICE_ID%s   the voice you want Adrien to use\n\n' "$BOLD" "$RESET"
    printf '    %s$EDITOR %s/.env%s\n\n' "$DIM" "$ROOT" "$RESET"
    printf '  Then run ./start.sh again.\n\n'
    exit 1
fi

# A .env with no Groq key produces a confusing runtime failure, so catch it here.
if ! "$VPY" - <<'PY' 2>/dev/null
import sys
sys.path.insert(0, ".")
from adrien.config import env_key_pool, load_env
load_env()
sys.exit(0 if env_key_pool("GROQ_API_KEY") else 1)
PY
then
    printf '\n'
    warn "No GROQ_API_KEY_1 in .env - Adrien cannot think or hear without it."
    printf '    %s$EDITOR %s/.env%s\n\n' "$DIM" "$ROOT" "$RESET"
    exit 1
fi
ok ".env has keys"

# Generate the client pairing token if it is missing, so the phone and iPad
# have something to pair against without a manual step.
if ! grep -qE '^ADRIEN_WS_TOKEN=.+' "$ROOT/.env" 2>/dev/null; then
    TOKEN="$("$VPY" -c 'import secrets; print(secrets.token_hex(24))')"
    if grep -q '^ADRIEN_WS_TOKEN=' "$ROOT/.env"; then
        "$VPY" - "$ROOT/.env" "$TOKEN" <<'PY'
import pathlib, sys
path, token = pathlib.Path(sys.argv[1]), sys.argv[2]
lines = path.read_text().splitlines()
path.write_text("\n".join(
    f"ADRIEN_WS_TOKEN={token}" if line.startswith("ADRIEN_WS_TOKEN=") else line
    for line in lines
) + "\n")
PY
    else
        printf '\nADRIEN_WS_TOKEN=%s\n' "$TOKEN" >> "$ROOT/.env"
    fi
    ok "generated a client pairing token"
fi

mkdir -p "$ROOT/logs"

# --- 6. Go ------------------------------------------------------------------
case "$MODE" in
    doctor)
        printf '\n'
        exec "$VPY" -m adrien doctor
        ;;

    chat)
        step "Starting Adrien (typed - no microphone needed)"
        printf '\n'
        exec "$VPY" -m adrien chat
        ;;

    service)
        step "Installing Adrien as a login service"
        printf '\n'
        exec "$ROOT/service/install_service.sh"
        ;;

    run)
        step "Running diagnostics"
        # doctor exits non-zero when something is missing; show it but let the
        # user decide, since most failures are partial and Adrien degrades.
        set +e
        "$VPY" -m adrien doctor
        DOCTOR_STATUS=$?
        set -e

        if [[ $DOCTOR_STATUS -ne 0 ]]; then
            printf '\n'
            warn "Some checks failed (see above)."
            warn "Adrien will start anyway and skip whatever is unavailable."
            printf '\n'
        fi

        step "Starting Adrien"
        printf '\n'
        printf '  Say the wake word, wait for the tone, then talk.\n'
        printf '  %sCtrl-C to stop.%s\n\n' "$DIM" "$RESET"
        exec "$VPY" -m adrien run "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
        ;;
esac
