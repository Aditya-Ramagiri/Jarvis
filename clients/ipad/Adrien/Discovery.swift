import Foundation
import Network

/// Bonjour discovery for `_adrien._tcp`.
///
/// Requires `NSLocalNetworkUsageDescription` and an `NSBonjourServices` entry
/// in Info.plist. Without them iOS 14+ fails the browse silently, which looks
/// exactly like "the Mac is off" - so check those first when nothing is found.
final class Discovery {

    struct Found { let host: String; let port: Int; let name: String }

    private var browser: NWBrowser?

    /// Browse until something is found or `timeout` elapses. First match wins:
    /// there is only ever one Mac on this network.
    func findAdrien(timeout: TimeInterval = 5, completion: @escaping (Found?) -> Void) {
        var finished = false
        func finish(_ result: Found?) {
            guard !finished else { return }
            finished = true
            self.browser?.cancel()
            self.browser = nil
            completion(result)
        }

        let parameters = NWParameters()
        parameters.includePeerToPeer = false
        let browser = NWBrowser(
            for: .bonjour(type: "_adrien._tcp", domain: nil),
            using: parameters
        )
        self.browser = browser

        browser.browseResultsChangedHandler = { results, _ in
            guard let result = results.first else { return }
            if case let .service(name, type, domain, _) = result.endpoint {
                Self.resolve(name: name, type: type, domain: domain) { found in
                    finish(found)
                }
            }
        }

        browser.stateUpdateHandler = { state in
            if case .failed(let error) = state {
                print("Adrien discovery failed: \(error.localizedDescription)")
                finish(nil)
            }
        }

        browser.start(queue: .main)
        DispatchQueue.main.asyncAfter(deadline: .now() + timeout) { finish(nil) }
    }

    /// A browse result is only a name; a connection resolves it to an address.
    private static func resolve(
        name: String, type: String, domain: String,
        completion: @escaping (Found?) -> Void
    ) {
        let connection = NWConnection(
            to: .service(name: name, type: type, domain: domain, interface: nil),
            using: .tcp
        )
        connection.stateUpdateHandler = { state in
            guard case .ready = state else { return }
            defer { connection.cancel() }
            guard case let .hostPort(host, port) = connection.currentPath?.remoteEndpoint else {
                completion(nil)
                return
            }
            // Strip the "%en0" scope suffix that link-local addresses carry.
            let address = "\(host)".components(separatedBy: "%").first ?? "\(host)"
            completion(Found(host: address, port: Int(port.rawValue), name: name))
        }
        connection.start(queue: .main)
    }
}
