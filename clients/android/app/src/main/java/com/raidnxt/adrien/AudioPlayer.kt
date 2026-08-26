package com.raidnxt.adrien

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log

/**
 * Streaming playback of the 24 kHz PCM the server sends back.
 *
 * Written against AudioTrack in streaming mode rather than MediaPlayer,
 * because the reply arrives in chunks and should start playing on the first
 * one - MediaPlayer wants a complete file, which would add the whole synthesis
 * time to the pause before Adrien speaks.
 */
class AudioPlayer {

    private var track: AudioTrack? = null

    fun start(sampleRate: Int) {
        stop()
        val minBuffer = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    // ASSISTANT ducks other audio politely and follows the
                    // right volume stream on the phone.
                    .setUsage(AudioAttributes.USAGE_ASSISTANT)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(minBuffer * 2)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
            .also { it.play() }
    }

    fun write(pcm: ByteArray) {
        runCatching { track?.write(pcm, 0, pcm.size) }
            .onFailure { Log.w(TAG, "playback write failed", it) }
    }

    /** Stops immediately, dropping anything still queued - this is barge-in. */
    fun stop() {
        track?.run {
            runCatching {
                pause()
                flush()
                stop()
            }
            release()
        }
        track = null
    }

    fun release() = stop()

    companion object { private const val TAG = "AdrienAudioPlayer" }
}
