# Architecture

How Adrien is put together, and why each piece is shaped the way it is. Where a
decision could reasonably have gone another way, the reasoning is here rather
than in a commit message nobody will find.

## The shape of a turn

```
  microphone (always open)
        │
        ▼
  wake_word.py ── openWakeWord, local, ~1% of a core, never hits an API
        │  detected
        ▼
  audio.py ────── record until the VAD says the sentence ended
        │
        ▼
  stt.py ──────── Groq whisper-large-v3, own key pool
        │  text
        ▼
  memory/manager.py ── semantic recall, injected as a system message
        │
        ▼
  llm_router.py ─ tier the model, rotate keys, fall back across providers
        │
        ├── tool calls ──▶ permissions.py ──▶ registry.py ──┐
        │                  (confirm?)          (execute)     │
        │◀────────────────── result fed back ────────────────┘
        │  final text
        ▼
  tts.py ──────── Fish Audio, streamed PCM
        │
        ▼
  audio.py ────── play, watching the mic for barge-in
        │
        ▼
  conversation.py ─ open the follow-up window; remember any interruption
```

## Layering

The package is deliberately split so the parts with no hardware and no network
in them import cleanly anywhere:

| Layer | Modules | Needs |
|---|---|---|
| **Pure** | `config`, `logging_setup`, `core/keypool`, `core/llm_types`, `core/conversation`, `tools/registry`, `tools/permissions`, `server/protocol` | nothing |
| **Network** | `core/llm_router`, `core/providers/*`, `core/stt`, `core/tts`, `memory/*` | httpx, keys |
| **Hardware** | `core/audio`, `core/wake_word`, `menubar` | sounddevice, onnx, AppKit |

Every heavyweight import happens *inside* the function that needs it. That is
not stylistic: it is what lets the 177-test suite run on a machine with no
audio stack, no ChromaDB and no API keys, and it is what lets a missing
optional dependency cost one feature instead of the whole assistant.

## Key rotation (spec section 4)

The one subsystem everything else depends on. `KeyPool` holds interchangeable
keys for one provider and hands them out **least-recently-used**.

**Circuit breaker, not retry-in-place.** A 429 puts that key into cooldown and
the next attempt takes a different key. Retrying the same key costs a full
round trip and buys nothing, and latency is a stated priority. The provider's
`Retry-After` wins over the configured default when it sends one.

**Acquisition never sleeps.** When every key is cooling, `acquire()` returns
`None` immediately so the router can fall through to the next provider rather
than blocking. Rotation costs a lock and a scan over a handful of entries.

**Separate pools over the same keys.** STT, chat and vision each get their own
pool. Groq's limits are per model per account, so a Whisper 429 says nothing
about whether that account can still serve llama — sharing a pool would
sideline a good chat key every time a long dictation hit the audio quota.

```
for provider in (groq, gemini):
    for lease in provider.pool.leases():
        try:    return await provider.chat(key=lease.key, ...)
        except rate_limited: lease.rate_limited(); continue   # no sleep
        except transient:    lease.failed();       continue
        except bad_request:  raise         # another key fails identically
raise AllProvidersFailed                   # the only user-visible failure
```

That last line matters: `AllProvidersFailed` is the *only* plumbing problem
Adrien mentions out loud (spec 4.5). Everything else is invisible.

## Model tiering

Two models, one router. The default is the **smart** model, because
conversation and multi-tool chains are explicitly its job. The **fast** model
handles what most utterances actually are — short literal commands, where 70B
buys nothing but latency.

A turn goes smart if: it already carries tool results (mid-chain reasoning), it
contains a reasoning cue from `llm.force_smart_keywords`, or the request runs
past a dozen words. Otherwise fast.

## Provider abstraction

Groq is OpenAI-shaped; Gemini is not. Rather than let either dialect leak
upwards, everything above the provider layer speaks the neutral types in
`llm_types.py`, and each adapter translates at its own boundary.

Gemini's adapter carries real work: system messages move to
`system_instruction`, tool calls become `functionCall` parts, tool results ride
on a *user* turn as `functionResponse`, and JSON Schema is reduced to the
subset Gemini accepts. It is tested independently, because it is the code most
likely to be wrong and least likely to be exercised — it only runs when every
Groq key is down.

Both providers are spoken to over `httpx` rather than their SDKs. The SDKs want
to own retries and client lifecycle; both belong to `KeyPool` and `LLMRouter`
here.

## Tool calling (spec section 7)

A tool is a decorated function. Its JSON schema is derived from the signature,
type hints and docstring, so a tool cannot drift out of sync with the schema
the model sees. There is no `if/elif` dispatch anywhere: `registry.schemas()`
feeds the provider's native function-calling API and `registry.execute()` runs
whatever comes back.

Three things the registry does that are easy to skip and painful to omit:

- **Argument coercion.** Models send `"5"` where an integer was declared and
  invent argument names. Unknown keys are dropped, declared types are coerced,
  and a failure comes back as one clean sentence the model can act on rather
  than a `TypeError`.
- **Timeouts.** Every call is capped. A hung tool would hang the conversation.
- **Truncation and redaction.** Results go straight into the next prompt, so a
  giant `git log` is capped and anything credential-shaped is stripped.

Sync tools run in a thread, so the event loop stays responsive and barge-in
keeps working while a tool is running.

## Permissions (spec section 9)

Resolution order, most specific first: per-tool setting → category setting →
global default. With two rules layered on top:

- A tool not marked `destructive` is **always** auto. Confirming "what's the
  weather" would make Adrien exhausting, and there is nothing to undo.
- A tool marked `irreversible` — powering the machine down, sending a message
  someone else will read, running arbitrary code — **cannot** be made automatic
  by a category-wide or global "auto". Setting `system: auto` is a reasonable
  thing to want for volume and app control; it is not consent to skip the
  question on a shutdown. Opting out of that takes naming the tool.

The confirmation itself is an injected callable, so the same policy serves a
spoken yes/no on the Mac, a tap on the phone, and an auto-yes in tests. An
ambiguous answer is never consent, and a negative anywhere in the sentence
beats an affirmative — the costs are asymmetric.

## Memory (spec section 6)

Two stores, both written on every fact, because they answer different questions:

- **ChromaDB** answers *what is this about*. One collection holds facts,
  session summaries and chunked transcripts, distinguished by metadata. They
  share a collection so "what was I frustrated about last week regarding the
  build system" can be answered from whichever kind actually knows.
- **SQLite** answers *when, from which session, is it still true*. Exact
  lookups with no embedding round trip, and it owns supersession.

A changed fact **supersedes** rather than deletes. "The server moved" should
not erase where it used to be, because that is a thing people ask.

Facts are `(subject, predicate, value, category)` with a free-form category, so
facts about a new project need no migration — just a new string.

Transcripts are **chunked** before embedding. A 40-minute conversation embedded
as one vector retrieves for everything and means nothing.

Retrieval keeps facts and history in separate lists rather than one ranked
merge: they play different roles in the prompt, and a chatty old transcript
should not crowd out the one standing fact that matters.

The post-session digest prompt is strict about what counts as durable. A memory
full of "the user asked about the weather" is worse than no memory — it crowds
out real facts and makes every future prompt longer and slower. An empty fact
list is the correct answer for most conversations.

## Conversation flow (spec 5.2, 5.3)

**The follow-up window is not a chat mode.** After a reply, the mic stays open
for a few seconds so the user can refine the task in flight. Adrien does not
prompt, does not fill silence, and returns to passive listening if nothing is
said.

**Interruption keeps the thread.** Playback is written so it can stop mid-word:
TTS is requested as raw PCM and never more than 20 ms is buffered. When the
user cuts in, `Conversation` records the full intended reply and the fraction
actually spoken. "Keep going" resumes from short-term state with **no second
model call** — the reply already exists, and re-deriving it would cost latency
and risk saying something different.

Barge-in on a laptop has no echo cancellation, so two guards keep Adrien from
interrupting itself: a run of consecutive speech frames (a syllable, not a
click) and an energy threshold calibrated against the first moments of
playback, when only Adrien is audible.

## Speech

**Whisper large-v3, not a faster variant.** A hard requirement in the spec, and
correct: a smaller model that mangles a disfluent request costs far more time
than it saves. Recent conversation is fed to Whisper as a decoding hint, which
markedly improves proper nouns — "Modrinth", "raidnxt", friends' names.

**TTS as raw PCM, not mp3.** The bytes go straight to the output device: no
decoder dependency, no buffering a file before the first word, and playback can
be cut mid-word. An mp3 path would need frame-by-frame decoding or `afplay`,
which cannot be interrupted cleanly.

Long replies are synthesised **sentence by sentence**, with sentence two being
generated while sentence one is still being spoken.

## Discord: why UI automation

The one integration that drives a GUI instead of an API, and the reason is
account safety:

- A **bot token** can only speak as a bot. Messages arrive from "Adrien BOT",
  and a bot cannot be in the user's DMs at all. It does not do the thing asked.
- A **self-bot** does do the thing, and is explicitly against Discord's ToS.
  Discord detects it and bans accounts. Trading someone's account for a tidier
  implementation is not a trade worth making.

Driving the desktop app is the only route that sends a real message from the
real user. Every keystroke is one a person could have typed. Because UI
automation is genuinely more fragile, the code verifies rather than assumes:
it checks Discord is frontmost, waits for the switcher to settle, and reads the
message field back through the Accessibility API **before** pressing Enter.
Sending "I'm running lat" to the wrong person is far worse than failing
cleanly, so every uncertain path fails cleanly.

Everything macOS exposes properly — volume, apps, Spotlight, clipboard — uses
the real API instead. Reaching for simulated keystrokes when an API exists is
how automation rots.

## Clients (spec section 8)

One WebSocket carrying JSON control frames and raw PCM binary frames. Audio has
no JSON envelope because base64 would inflate every utterance by a third for no
benefit.

Clients are thin: capture audio, play audio. All thinking happens on the Mac,
so a new platform is a microphone and a speaker, not a second Adrien.

**Local network only, and checked rather than assumed.** The server refuses at
startup to bind to a publicly routable address. There is no tunnel, no port
forwarding, no relay. Off the WiFi the clients say "Adrien unavailable".

mDNS advertisement means no client ever asks for an IP — a DHCP renewal would
silently break a typed address, and the failure would look like "Adrien is
broken".

A confirmation raised during a phone's turn is routed **back to that phone**,
not asked through the Mac's speaker where nobody is standing.

## Security

- Secrets live only in `.env`, which is gitignored.
- A `logging.Filter` on the root logger redacts credentials from every record,
  including ones arriving through third-party exception text. Redaction is
  enforced centrally rather than trusted to each call site.
- The same redaction runs over everything persisted: memory records, tool
  results, transcripts, clipboard history.
- The summariser drops any extracted "fact" whose value looks like a
  credential.
- Subprocesses run without a shell, since arguments arrive from an LLM. The one
  exception is the user's own Minecraft start/stop commands, which are shell
  strings they wrote themselves.
- AppleScript strings are escaped before interpolation — AppleScript has no
  parameter binding, so a quote in a message would otherwise end the literal.
- The client socket takes a shared token, compared in constant time.

## Where things live

```
adrien/
├── config.py           secrets from .env, behaviour from settings.json
├── logging_setup.py    central redaction
├── __main__.py         the CLI (run, chat, doctor, status, tools, memory)
├── menubar.py          macOS status item
├── core/
│   ├── keypool.py      LRU rotation + circuit breaker
│   ├── llm_router.py   tiering, rotation, provider fallback
│   ├── llm_types.py    provider-neutral messages and tool calls
│   ├── providers/      groq.py, gemini.py
│   ├── audio.py        capture, VAD, playback, barge-in
│   ├── wake_word.py    openWakeWord
│   ├── stt.py          Groq Whisper
│   ├── tts.py          Fish Audio
│   ├── conversation.py follow-up window, interruption memory
│   └── orchestrator.py the main loop
├── memory/             vector_store, structured_store, summarizer, manager
├── tools/              registry, permissions, and the seven tool modules
└── server/             protocol, ws_server, discovery
```

Data lives outside the repo, in `~/Library/Application Support/Adrien`:
`adrien.sqlite3`, `chroma/`, `logs/`, `clipboard_history.json`.
