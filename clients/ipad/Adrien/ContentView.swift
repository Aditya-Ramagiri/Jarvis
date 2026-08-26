import SwiftUI

/// Push-to-talk, connection state, confirmations.
///
/// Small on purpose: everything clever runs on the Mac (spec section 8).
struct ContentView: View {

    @StateObject private var model = AdrienViewModel()

    var body: some View {
        VStack(spacing: 24) {
            Text(model.statusText)
                .font(.title2)
                .foregroundStyle(model.isAvailable ? .primary : .secondary)

            if !model.lastReply.isEmpty {
                ScrollView {
                    Text(model.lastReply)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                }
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
            }

            Spacer()

            if let prompt = model.pendingConfirmation {
                VStack(spacing: 12) {
                    Text(prompt).multilineTextAlignment(.center)
                    HStack(spacing: 16) {
                        Button("Yes") { model.answerConfirmation(true) }
                            .buttonStyle(.borderedProminent)
                        Button("No") { model.answerConfirmation(false) }
                            .buttonStyle(.bordered)
                    }
                }
                .padding()
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
            }

            Button(action: model.toggleListening) {
                Text(model.isListening ? "Stop" : "Talk to Adrien")
                    .frame(maxWidth: .infinity, minHeight: 64)
            }
            .buttonStyle(.borderedProminent)
            .disabled(!model.isAvailable)
        }
        .padding(24)
        .task { await model.connect() }
    }
}

@MainActor
final class AdrienViewModel: ObservableObject {

    @Published var statusText = "Looking for Adrien…"
    @Published var lastReply = ""
    @Published var pendingConfirmation: String?
    @Published var isListening = false
    @Published var isAvailable = false

    private var client: AdrienClient?
    private let discovery = Discovery()
    private var observers: [Any] = []

    func connect() async {
        let token = TokenStore.load()
        guard !token.isEmpty else {
            statusText = "Not paired - enter the token from the Mac"
            return
        }

        let client = AdrienClient(token: token)
        self.client = client
        observe(client)

        discovery.findAdrien { [weak self] found in
            guard let self else { return }
            guard let found else {
                // The one honest thing to say. No relay, no fallback.
                self.statusText = "Adrien unavailable"
                self.isAvailable = false
                return
            }
            client.connect(host: found.host, port: found.port)
        }
    }

    private func observe(_ client: AdrienClient) {
        observers.append(client.$state.sink { [weak self] state in
            guard let self else { return }
            self.isAvailable = state != .disconnected && state != .connecting
            self.isListening = state == .listening
            self.statusText = switch state {
            case .disconnected: "Adrien unavailable"
            case .connecting: "Looking for Adrien…"
            case .idle: "Ready"
            case .listening: "Listening"
            case .thinking: "Thinking"
            case .speaking: "Speaking"
            case .confirming: "Waiting on you"
            }
        })
        observers.append(client.$lastReply.sink { [weak self] in self?.lastReply = $0 })
        observers.append(client.$pendingConfirmation.sink { [weak self] in
            self?.pendingConfirmation = $0
        })
    }

    func toggleListening() {
        guard let client else { return }
        if isListening { client.stopListening() } else { client.startListening() }
    }

    func answerConfirmation(_ yes: Bool) {
        client?.answerConfirmation(yes)
    }
}

/// The pairing token, in the keychain rather than UserDefaults - it is the
/// only thing stopping anything else on the WiFi talking to Adrien.
enum TokenStore {
    private static let account = "adrien.ws.token"

    static func load() -> String {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data else { return "" }
        return String(data: data, encoding: .utf8) ?? ""
    }

    static func save(_ token: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
        var attributes = query
        attributes[kSecValueData as String] = Data(token.utf8)
        SecItemAdd(attributes as CFDictionary, nil)
    }
}
