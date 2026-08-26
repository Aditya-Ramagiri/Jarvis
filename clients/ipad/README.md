# Adrien for iPad

A thin client, same as Android: microphone in, speaker out, all thinking on the
Mac.

## The platform limitation, stated plainly

**True always-on background listening is not achievable for a third-party
iPadOS app, and this client does not pretend otherwise** (spec section 8).

iOS/iPadOS suspends apps shortly after they leave the foreground. The
background modes that keep an app alive — `audio`, `voip`, location — are
policed by App Review against their stated purpose, and "listen for a wake
word" is not one of them. `voip` in particular was narrowed years ago
specifically to stop this pattern. The workarounds that circulate (a silent
audio loop to hold the `audio` mode, a fake location subscription) are
rejections waiting to happen and burn the battery besides.

So: **Adrien on iPad works when the app is open.** That is the real ceiling,
and building against it is more useful than building something that
half-works until iPadOS kills it mid-sentence.

What closes most of the gap:

| Route | What it gives you |
|---|---|
| **App open** | Full experience: push to talk, streamed replies, confirmations. |
| **Home Screen widget** | One tap straight into a listening state. `AdrienWidget.swift`. |
| **Shortcuts / App Intents** | "Hey Siri, ask Adrien…" — Siri handles the wake word, and hands the text to Adrien. This is as close to hands-free as the platform allows. |
| **Stage Manager** | Keep the app in a corner window; it stays foregrounded and fully live. |

The Siri route is worth the effort: `AskAdrienIntent` makes Adrien reachable
without touching the device, using the only wake word iPadOS will run for a
third party.

## What is here

| File | What it does |
|---|---|
| `AdrienClient.swift` | The protocol — `URLSessionWebSocketTask`, handshake, audio streaming, playback. |
| `AudioEngine.swift` | `AVAudioEngine` capture at 16 kHz and streamed playback at 24 kHz. |
| `Discovery.swift` | Bonjour browse for `_adrien._tcp` via `NWBrowser`. |
| `ContentView.swift` | Push-to-talk, state, confirmations. |
| `AskAdrienIntent.swift` | App Intent, so Siri and Shortcuts can reach Adrien. |

## Setup

1. Open the folder in Xcode 15+, create an iOS App target named `Adrien`, and
   add these files to it.
2. `Info.plist` needs:
   - `NSMicrophoneUsageDescription` — why the mic is used.
   - `NSLocalNetworkUsageDescription` — required since iOS 14 for *any* local
     network traffic, including Bonjour. Without it, discovery silently fails.
   - `NSBonjourServices` — an array containing `_adrien._tcp`.
3. Paste the Mac's `ADRIEN_WS_TOKEN` on first launch.
4. Run on a device on the same WiFi as the Mac. The simulator cannot browse
   Bonjour to the host network reliably.

## When the Mac is unreachable

Show **Adrien unavailable** and keep browsing. No tunnel, no relay, no
fallback — that is the design (spec section 8), not a missing feature.
