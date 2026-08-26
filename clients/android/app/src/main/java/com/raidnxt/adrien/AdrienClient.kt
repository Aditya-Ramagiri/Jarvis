package com.raidnxt.adrien

import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * The Adrien protocol, end to end. See clients/PROTOCOL.md.
 *
 * Deliberately holds no UI and no Android service plumbing: AdrienService owns
 * the lifecycle, MainActivity owns the screen, and this owns the wire. That
 * split is what lets the same class back a future Wear or TV client.
 */
class AdrienClient(
    private val token: String,
    private val deviceName: String,
    private val scope: CoroutineScope,
) {
    enum class State { DISCONNECTED, CONNECTING, IDLE, LISTENING, THINKING, SPEAKING, CONFIRMING }

    private val http = OkHttpClient.Builder()
        // The server pings every 20s; matching it here means a dead socket is
        // noticed in seconds rather than whenever the user next speaks.
        .pingInterval(20, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    private var socket: WebSocket? = null
    private val player = AudioPlayer()

    private val _state = MutableStateFlow(State.DISCONNECTED)
    val state: StateFlow<State> = _state

    private val _lastReply = MutableStateFlow("")
    val lastReply: StateFlow<String> = _lastReply

    /** Set while the server is waiting on a yes/no. The UI must ask a human. */
    private val _pendingConfirmation = MutableStateFlow<String?>(null)
    val pendingConfirmation: StateFlow<String?> = _pendingConfirmation

    fun connect(host: String, port: Int) {
        _state.value = State.CONNECTING
        val request = Request.Builder().url("ws://$host:$port/").build()
        socket = http.newWebSocket(request, object : WebSocketListener() {

            override fun onOpen(webSocket: WebSocket, response: Response) {
                webSocket.send(
                    JSONObject()
                        .put("type", "hello")
                        .put("token", token)
                        .put("device", deviceName)
                        .put("platform", "android")
                        .toString()
                )
            }

            override fun onMessage(webSocket: WebSocket, text: String) = handleControl(text)

            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                // Play as it arrives; waiting for audio_end would add the whole
                // synthesis time to perceived latency.
                player.write(bytes.toByteArray())
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.w(TAG, "socket failed: ${t.message}")
                _state.value = State.DISCONNECTED
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                _state.value = State.DISCONNECTED
            }
        })
    }

    private fun handleControl(raw: String) = scope.launch(Dispatchers.Default) {
        val frame = runCatching { JSONObject(raw) }.getOrNull() ?: return@launch
        when (frame.optString("type")) {
            "welcome" -> {
                Log.i(TAG, "connected to ${frame.optString("assistant")}")
                _state.value = State.IDLE
            }
            "reply" -> _lastReply.value = frame.optString("text")
            "state" -> _state.value = when (frame.optString("state")) {
                "thinking" -> State.THINKING
                "speaking" -> State.SPEAKING
                "listening" -> State.LISTENING
                "confirming" -> State.CONFIRMING
                else -> State.IDLE
            }
            "confirm" -> {
                // Surfaced to the user. Never answered automatically: the whole
                // point of the confirmation layer is that a person said yes.
                _pendingConfirmation.value = frame.optString("prompt")
                _state.value = State.CONFIRMING
            }
            "audio_start" -> player.start(frame.optInt("sample_rate", SERVER_SAMPLE_RATE))
            "audio_end" -> player.stop()
            "error" -> {
                Log.e(TAG, "server error: ${frame.optString("reason")}")
                if (frame.optBoolean("fatal")) disconnect()
            }
        }
    }

    // -- sending ----------------------------------------------------------
    fun beginUtterance(wantAudio: Boolean = true) {
        socket?.send(
            JSONObject().put("type", "audio_start").put("want_audio", wantAudio).toString()
        )
        _state.value = State.LISTENING
    }

    /** One chunk of 16 kHz mono PCM, straight from AudioRecord. */
    fun sendAudio(pcm: ByteArray, length: Int) {
        socket?.send(pcm.copyOf(length).toByteString())
    }

    fun endUtterance() {
        socket?.send(JSONObject().put("type", "audio_end").toString())
        _state.value = State.THINKING
    }

    fun sendText(text: String, wantAudio: Boolean = true) {
        socket?.send(
            JSONObject().put("type", "text").put("text", text)
                .put("want_audio", wantAudio).toString()
        )
        _state.value = State.THINKING
    }

    /** Answer an outstanding confirmation. Only ever call this from a tap. */
    fun answerConfirmation(yes: Boolean) {
        _pendingConfirmation.value = null
        socket?.send(
            JSONObject().put("type", "text").put("text", if (yes) "yes" else "no").toString()
        )
    }

    /** Barge-in from the client side: stop playback and drop the reply. */
    fun cancel() {
        player.stop()
        socket?.send(JSONObject().put("type", "cancel").toString())
        _state.value = State.IDLE
    }

    fun disconnect() {
        player.release()
        socket?.close(1000, "bye")
        socket = null
        _state.value = State.DISCONNECTED
    }

    companion object {
        const val TAG = "AdrienClient"
        const val CLIENT_SAMPLE_RATE = 16_000
        const val SERVER_SAMPLE_RATE = 24_000
    }
}
