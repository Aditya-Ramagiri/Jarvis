package com.raidnxt.adrien

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.IBinder
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * Foreground service holding the mic and the socket.
 *
 * Android will not let a backgrounded app keep a microphone open, and should
 * not: a persistent notification is the honest signal that something is
 * listening. This is the same pattern Google Assistant uses, and since
 * Android 14 the `microphone` foreground-service type makes it explicit.
 *
 * The service owns the connection lifecycle so a dropped WiFi link reconnects
 * without the user reopening the app.
 */
class AdrienService : Service() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private lateinit var client: AdrienClient
    private var recorder: AudioRecord? = null
    private var captureJob: Job? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        startForegroundCompat()

        client = AdrienClient(
            token = Prefs.token(this),
            deviceName = Build.MODEL ?: "Android",
            scope = scope,
        )
        connectWhenFound()
    }

    /**
     * Browse mDNS and connect. Retries with a widening delay rather than
     * hammering: the Mac being asleep is a normal state, not an error.
     */
    private fun connectWhenFound() = scope.launch {
        var backoffMs = 2_000L
        while (true) {
            val service = Discovery.findAdrien(this@AdrienService, timeoutMs = 4_000)
            if (service != null) {
                client.connect(service.host, service.port)
                updateNotification("Connected to Adrien")
                return@launch
            }
            updateNotification("Adrien unavailable")
            kotlinx.coroutines.delay(backoffMs)
            backoffMs = (backoffMs * 2).coerceAtMost(60_000L)
        }
    }

    // -- push to talk ------------------------------------------------------
    @Suppress("MissingPermission") // RECORD_AUDIO is checked by MainActivity
    fun startListening() {
        if (captureJob != null) return

        val bufferSize = AudioRecord.getMinBufferSize(
            AdrienClient.CLIENT_SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        ).coerceAtLeast(CHUNK_BYTES)

        recorder = AudioRecord(
            // VOICE_RECOGNITION gets the platform's own noise suppression and
            // AGC, which is exactly what Whisper wants to be handed.
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            AdrienClient.CLIENT_SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize,
        ).also { it.startRecording() }

        client.beginUtterance()
        updateNotification("Listening")

        captureJob = scope.launch {
            val buffer = ByteArray(CHUNK_BYTES)
            while (recorder?.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                val read = recorder?.read(buffer, 0, buffer.size) ?: -1
                if (read > 0) client.sendAudio(buffer, read)
            }
        }
    }

    fun stopListening() {
        captureJob?.cancel()
        captureJob = null
        recorder?.run {
            if (recordingState == AudioRecord.RECORDSTATE_RECORDING) stop()
            release()
        }
        recorder = null
        client.endUtterance()
        updateNotification("Thinking")
    }

    fun clientHandle(): AdrienClient = client

    override fun onBind(intent: Intent?): IBinder = LocalBinder()

    inner class LocalBinder : android.os.Binder() {
        fun service(): AdrienService = this@AdrienService
    }

    override fun onDestroy() {
        stopListening()
        client.disconnect()
        scope.cancel()
        super.onDestroy()
    }

    // -- notification ------------------------------------------------------
    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Adrien",
            // LOW: the notification must exist, but it should never buzz.
            NotificationManager.IMPORTANCE_LOW,
        ).apply { description = "Keeps Adrien connected and able to listen" }
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    private fun buildNotification(text: String): Notification {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE,
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("Adrien")
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_adrien)
            .setContentIntent(open)
            .setOngoing(true)
            .build()
    }

    private fun startForegroundCompat() {
        val notification = buildNotification("Looking for Adrien")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID, notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE,
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun updateNotification(text: String) {
        runCatching {
            getSystemService(NotificationManager::class.java)
                .notify(NOTIFICATION_ID, buildNotification(text))
        }.onFailure { Log.w(TAG, "could not update the notification", it) }
    }

    companion object {
        const val TAG = "AdrienService"
        private const val CHANNEL_ID = "adrien"
        private const val NOTIFICATION_ID = 1
        // 100 ms of 16 kHz mono 16-bit audio.
        private const val CHUNK_BYTES = 3_200

        fun start(context: Context) {
            context.startForegroundService(Intent(context, AdrienService::class.java))
        }
    }
}
