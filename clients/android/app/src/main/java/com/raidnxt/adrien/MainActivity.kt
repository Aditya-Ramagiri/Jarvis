package com.raidnxt.adrien

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.IBinder
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import kotlinx.coroutines.flow.MutableStateFlow

/**
 * Push-to-talk, connection state, and the last exchange.
 *
 * Intentionally small. Anything clever belongs on the Mac (spec section 8);
 * this screen exists to hold a button and show what Adrien said.
 */
class MainActivity : ComponentActivity() {

    private var service: AdrienService? = null
    private val bound = MutableStateFlow(false)

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            service = (binder as AdrienService.LocalBinder).service()
            bound.value = true
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            service = null
            bound.value = false
        }
    }

    private val requestMic = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startAndBind() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { AdrienScreen() }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startAndBind()
        } else {
            requestMic.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    private fun startAndBind() {
        AdrienService.start(this)
        bindService(Intent(this, AdrienService::class.java), connection, Context.BIND_AUTO_CREATE)
    }

    override fun onDestroy() {
        if (bound.value) unbindService(connection)
        super.onDestroy()
    }

    @Composable
    private fun AdrienScreen() {
        val isBound by bound.collectAsState()
        val client = service?.clientHandle()
        val state by (client?.state ?: MutableStateFlow(AdrienClient.State.DISCONNECTED))
            .collectAsState()
        val reply by (client?.lastReply ?: MutableStateFlow("")).collectAsState()
        val confirmation by (client?.pendingConfirmation ?: MutableStateFlow<String?>(null))
            .collectAsState()

        MaterialTheme {
            Surface {
                Column(
                    modifier = Modifier.fillMaxSize().padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(20.dp),
                ) {
                    Text(
                        text = when (state) {
                            // The one honest thing to say when the Mac is
                            // unreachable. No fallback, by design.
                            AdrienClient.State.DISCONNECTED -> "Adrien unavailable"
                            AdrienClient.State.CONNECTING -> "Looking for Adrien…"
                            AdrienClient.State.LISTENING -> "Listening"
                            AdrienClient.State.THINKING -> "Thinking"
                            AdrienClient.State.SPEAKING -> "Speaking"
                            AdrienClient.State.CONFIRMING -> "Waiting on you"
                            AdrienClient.State.IDLE -> "Ready"
                        },
                        style = MaterialTheme.typography.headlineSmall,
                    )

                    if (reply.isNotBlank()) {
                        Card(modifier = Modifier.fillMaxWidth()) {
                            Text(reply, modifier = Modifier.padding(16.dp))
                        }
                    }

                    Spacer(Modifier.weight(1f))

                    confirmation?.let { prompt ->
                        Card(modifier = Modifier.fillMaxWidth()) {
                            Column(Modifier.padding(16.dp)) {
                                Text(prompt)
                                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                                    Button(onClick = { client?.answerConfirmation(true) }) {
                                        Text("Yes")
                                    }
                                    OutlinedButton(onClick = { client?.answerConfirmation(false) }) {
                                        Text("No")
                                    }
                                }
                            }
                        }
                    }

                    Button(
                        onClick = {
                            if (state == AdrienClient.State.LISTENING) {
                                service?.stopListening()
                            } else {
                                service?.startListening()
                            }
                        },
                        enabled = isBound && state != AdrienClient.State.DISCONNECTED,
                        modifier = Modifier.fillMaxWidth().height(72.dp),
                    ) {
                        Text(if (state == AdrienClient.State.LISTENING) "Stop" else "Talk to Adrien")
                    }
                }
            }
        }
    }
}
