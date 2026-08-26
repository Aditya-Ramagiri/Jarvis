import Foundation
import Combine

/// The Adrien protocol for iPadOS. See clients/PROTOCOL.md.
///
/// Built on `URLSessionWebSocketTask` rather than a third-party socket
/// library: it is part of the SDK, it handles the TLS-free ws:// case on the
/// local network correctly, and one fewer dependency in an app that does very
/// little is worth having.
@MainActor
final class AdrienClient: ObservableObject {

    enum State: String {
        case disconnected, connecting, idle, listening, thinking, speaking, confirming
    }

    @Published private(set) var state: State = .disconnected
    @Published private(set) var lastReply: String = ""
    /// Set while the Mac is waiting on a yes/no. The UI must ask a person.
    @Published private(set) var pendingConfirmation: String?

    private var task: URLSessionWebSocketTask?
    private let session = URLSession(configuration: .default)
    private let audio = AudioEngine()
    private let token: String
    private let deviceName: String

    init(token: String, deviceName: String = UIDevice.current.name) {
        self.token = token
        self.deviceName = deviceName
    }

    // MARK: - Connection

    func connect(host: String, port: Int) {
        guard let url = URL(string: "ws://\(host):\(port)/") else { return }
        state = .connecting

        let task = session.webSocketTask(with: url)
        self.task = task
        task.resume()
        receiveLoop()

        send(json: [
            "type": "hello",
            "token": token,
            "device": deviceName,
            "platform": "ipados",
        ])
    }

    func disconnect() {
        audio.stopPlayback()
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        state = .disconnected
    }

    private func receiveLoop() {
        task?.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure(let error):
                Task { @MainActor in
                    print("Adrien socket failed: \(error.localizedDescription)")
                    self.state = .disconnected
                }
                return  // the socket is gone; stop looping

            case .success(let message):
                Task { @MainActor in
                    switch message {
                    case .string(let text):
                        self.handleControl(text)
                    case .data(let data):
                        // Play as it arrives rather than waiting for
                        // audio_end: that pause is the whole latency budget.
                        self.audio.enqueue(pcm: data)
                    @unknown default:
                        break
                    }
                }
                self.receiveLoop()
            }
        }
    }

    // MARK: - Incoming frames

    private func handleControl(_ raw: String) {
        guard
            let data = raw.data(using: .utf8),
            let frame = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let type = frame["type"] as? String
        else { return }

        switch type {
        case "welcome":
            state = .idle

        case "reply":
            lastReply = frame["text"] as? String ?? ""

        case "state":
            state = State(rawValue: frame["state"] as? String ?? "idle") ?? .idle

        case "confirm":
            // Never answered automatically - a person has to say yes.
            pendingConfirmation = frame["prompt"] as? String
            state = .confirming

        case "audio_start":
            audio.startPlayback(sampleRate: frame["sample_rate"] as? Double ?? 24_000)

        case "audio_end":
            audio.finishPlayback()

        case "error":
            print("Adrien error: \(frame["reason"] as? String ?? "unknown")")
            if frame["fatal"] as? Bool == true { disconnect() }

        default:
            break
        }
    }

    // MARK: - Outgoing

    private func send(json: [String: Any]) {
        guard
            let data = try? JSONSerialization.data(withJSONObject: json),
            let text = String(data: data, encoding: .utf8)
        else { return }
        task?.send(.string(text)) { error in
            if let error { print("Adrien send failed: \(error.localizedDescription)") }
        }
    }

    func startListening() {
        send(json: ["type": "audio_start", "want_audio": true])
        state = .listening
        audio.startCapture { [weak self] pcm in
            self?.task?.send(.data(pcm)) { _ in }
        }
    }

    func stopListening() {
        audio.stopCapture()
        send(json: ["type": "audio_end"])
        state = .thinking
    }

    func send(text: String, wantAudio: Bool = true) {
        send(json: ["type": "text", "text": text, "want_audio": wantAudio])
        state = .thinking
    }

    /// Answer an outstanding confirmation. Only ever call this from a tap.
    func answerConfirmation(_ yes: Bool) {
        pendingConfirmation = nil
        send(json: ["type": "text", "text": yes ? "yes" : "no"])
    }

    /// Barge-in: stop playback here and tell the Mac to stop sending.
    func cancel() {
        audio.stopPlayback()
        send(json: ["type": "cancel"])
        state = .idle
    }
}
