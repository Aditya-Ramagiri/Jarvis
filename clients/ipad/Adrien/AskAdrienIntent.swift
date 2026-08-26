import AppIntents

/// "Hey Siri, ask Adrien what's on my calendar."
///
/// This is the closest iPadOS gets to hands-free for a third-party app: Siri
/// owns the wake word, and hands the transcribed request straight to Adrien.
/// It works from the lock screen and with the app closed, which is exactly
/// what a background listener cannot do (see README).
struct AskAdrienIntent: AppIntent {

    static var title: LocalizedStringResource = "Ask Adrien"
    static var description = IntentDescription("Send a request to Adrien on the Mac.")
    // Answers come back spoken by Siri, so there is no need to surface the app.
    static var openAppWhenRun: Bool = false

    @Parameter(title: "Request")
    var request: String

    static var parameterSummary: some ParameterSummary {
        Summary("Ask Adrien \(\.$request)")
    }

    func perform() async throws -> some IntentResult & ProvidesDialog {
        let token = TokenStore.load()
        guard !token.isEmpty else {
            return .result(dialog: "Adrien isn't paired yet. Open the app to enter the token.")
        }

        let reply = try await AdrienOneShot(token: token).ask(request)
        return .result(dialog: IntentDialog(stringLiteral: reply))
    }
}

/// One request, one reply, no audio - the shape Siri and Shortcuts want.
///
/// Deliberately separate from `AdrienClient`: that one holds a live connection
/// and a playback engine, neither of which makes sense inside an intent that
/// must finish in a few seconds.
struct AdrienOneShot {

    let token: String
    private let discovery = Discovery()

    func ask(_ text: String) async throws -> String {
        guard let found = await find() else {
            return "Adrien is unavailable - the Mac isn't reachable on this network."
        }

        let client = await AdrienClient(token: token)
        await client.connect(host: found.host, port: found.port)
        // want_audio: false - Siri speaks the reply, so synthesising it on the
        // Mac would just add latency to a response nobody hears.
        await client.send(text: text, wantAudio: false)

        for _ in 0..<120 {                       // up to ~30s
            try await Task.sleep(nanoseconds: 250_000_000)
            let reply = await client.lastReply
            if !reply.isEmpty {
                await client.disconnect()
                return reply
            }
        }
        await client.disconnect()
        return "Adrien didn't answer in time."
    }

    private func find() async -> Discovery.Found? {
        await withCheckedContinuation { continuation in
            discovery.findAdrien(timeout: 4) { continuation.resume(returning: $0) }
        }
    }
}
