package com.raidnxt.adrien

import android.content.Context

/**
 * The pairing token, stored on device.
 *
 * Kept out of the APK deliberately: BuildConfig fields end up in a decompiled
 * build, and this token is what stops anything else on the WiFi talking to
 * Adrien. It is entered once on the pairing screen.
 */
object Prefs {
    private const val FILE = "adrien"
    private const val KEY_TOKEN = "ws_token"

    fun token(context: Context): String =
        context.getSharedPreferences(FILE, Context.MODE_PRIVATE).getString(KEY_TOKEN, "") ?: ""

    fun setToken(context: Context, token: String) {
        context.getSharedPreferences(FILE, Context.MODE_PRIVATE)
            .edit().putString(KEY_TOKEN, token.trim()).apply()
    }

    fun isPaired(context: Context): Boolean = token(context).isNotBlank()
}
