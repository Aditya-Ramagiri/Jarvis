# Adrien for Android

A thin client: microphone in, speaker out. Every piece of thinking happens on
the Mac (spec section 8), so this app is a transport and a UI, nothing more.

## What is here

| File | What it does |
|---|---|
| `AdrienClient.kt` | The whole protocol — handshake, audio streaming, playback. Mirrors `clients/PROTOCOL.md`. |
| `AdrienService.kt` | Foreground service with a persistent notification, so the connection and mic survive the app being backgrounded. |
| `Discovery.kt` | NSD (Android's mDNS) browse for `_adrien._tcp`. No IP typing. |
| `MainActivity.kt` | Push-to-talk button, connection state, the last exchange. |
| `AndroidManifest.xml` | Permissions and the foreground-service declaration. |

## Why a foreground service

Android kills background microphone access aggressively, and rightly so. A
foreground service with a persistent notification is the sanctioned way to hold
a live mic and socket — the same pattern Google Assistant uses. The notification
is not an annoyance to design around; it is the honest signal that something is
listening.

Since Android 14 the service must declare `foregroundServiceType="microphone"`
and hold `FOREGROUND_SERVICE_MICROPHONE`, both of which are in the manifest.

## Setup

1. Open `clients/android` in Android Studio (Giraffe or newer).
2. Put the Mac's `ADRIEN_WS_TOKEN` into the app: **Settings → Pair**, or drop it
   into `local.properties` as `adrien.token=...` for development builds.
3. Build and run on a device **on the same WiFi as the Mac**. An emulator will
   not find the service — emulator networking does not carry mDNS.

## Wake word on Android

Not implemented here, deliberately. Running openWakeWord on the phone would
mean a second always-listening model, a second battery drain, and a second
place where "Adrien" has to be trained. Push-to-talk is the honest v1; if
always-on is wanted later, the model to port is the same ONNX file the Mac uses
and the hook is `AdrienService.startListening()`.

## When the Mac is unreachable

The client shows **Adrien unavailable** and keeps browsing for the service. It
does not fall back to anything, because there is nothing to fall back to — that
is the design (spec section 8), not a gap.
