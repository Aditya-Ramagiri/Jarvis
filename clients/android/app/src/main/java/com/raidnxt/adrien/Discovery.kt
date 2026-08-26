package com.raidnxt.adrien

import android.content.Context
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.util.Log
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.coroutines.resume

/**
 * mDNS discovery via Android's NSD.
 *
 * The spec is explicit that the user should never type an IP address, and it
 * is right: a DHCP lease renewal would silently break a hardcoded address, and
 * the failure would look like "Adrien is broken" rather than "the IP moved".
 *
 * Returns the first Adrien found. There is only ever one Mac on the WiFi in
 * this setup, so first-wins is correct and simpler than a picker.
 */
object Discovery {

    data class Found(val host: String, val port: Int, val name: String)

    private const val SERVICE_TYPE = "_adrien._tcp."
    private const val TAG = "AdrienDiscovery"

    suspend fun findAdrien(context: Context, timeoutMs: Long = 5_000): Found? =
        withTimeoutOrNull(timeoutMs) {
            val nsd = context.getSystemService(Context.NSD_SERVICE) as NsdManager

            suspendCancellableCoroutine { continuation ->
                lateinit var discoveryListener: NsdManager.DiscoveryListener

                fun finish(found: Found?) {
                    if (!continuation.isActive) return
                    runCatching { nsd.stopServiceDiscovery(discoveryListener) }
                    continuation.resume(found)
                }

                val resolveListener = object : NsdManager.ResolveListener {
                    override fun onServiceResolved(resolved: NsdServiceInfo) {
                        val host = resolved.host?.hostAddress
                        if (host == null) {
                            Log.w(TAG, "resolved ${resolved.serviceName} with no address")
                            return
                        }
                        finish(Found(host, resolved.port, resolved.serviceName))
                    }

                    override fun onResolveFailed(info: NsdServiceInfo, code: Int) {
                        // Keep browsing: another advertisement may resolve.
                        Log.w(TAG, "resolve failed for ${info.serviceName}: $code")
                    }
                }

                discoveryListener = object : NsdManager.DiscoveryListener {
                    override fun onServiceFound(info: NsdServiceInfo) {
                        // A found service is only a name; resolving turns it
                        // into the host and port we can actually connect to.
                        nsd.resolveService(info, resolveListener)
                    }

                    override fun onStartDiscoveryFailed(type: String, code: Int) {
                        Log.e(TAG, "discovery failed to start: $code")
                        if (continuation.isActive) continuation.resume(null)
                    }

                    override fun onServiceLost(info: NsdServiceInfo) = Unit
                    override fun onDiscoveryStarted(type: String) = Unit
                    override fun onDiscoveryStopped(type: String) = Unit
                    override fun onStopDiscoveryFailed(type: String, code: Int) = Unit
                }

                nsd.discoverServices(SERVICE_TYPE, NsdManager.PROTOCOL_DNS_SD, discoveryListener)
                continuation.invokeOnCancellation {
                    runCatching { nsd.stopServiceDiscovery(discoveryListener) }
                }
            }
        }
}
