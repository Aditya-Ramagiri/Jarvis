import AVFoundation

/// Microphone capture at 16 kHz and streamed playback at 24 kHz.
///
/// `AVAudioEngine` rather than `AVAudioRecorder`/`AVAudioPlayer`: both of those
/// want files, and Adrien streams. The converter exists because the hardware
/// picks its own input rate (48 kHz on most iPads) and Whisper wants 16 kHz.
final class AudioEngine {

    private let engine = AVAudioEngine()
    private var playerNode: AVAudioPlayerNode?
    private var playbackFormat: AVAudioFormat?

    private let targetSampleRate: Double = 16_000

    // MARK: - Capture

    func startCapture(onChunk: @escaping (Data) -> Void) {
        configureSession(.record)

        let input = engine.inputNode
        let hardwareFormat = input.outputFormat(forBus: 0)
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: true
        ) else { return }

        let converter = AVAudioConverter(from: hardwareFormat, to: targetFormat)

        // ~100 ms buffers: small enough that the Mac starts transcribing
        // promptly, large enough not to spam the socket.
        input.installTap(onBus: 0, bufferSize: 4_800, format: hardwareFormat) { buffer, _ in
            guard let converter else { return }
            let capacity = AVAudioFrameCount(
                Double(buffer.frameLength) * self.targetSampleRate / hardwareFormat.sampleRate
            )
            guard let converted = AVAudioPCMBuffer(
                pcmFormat: targetFormat, frameCapacity: max(capacity, 1)
            ) else { return }

            var error: NSError?
            converter.convert(to: converted, error: &error) { _, status in
                status.pointee = .haveData
                return buffer
            }
            if let error {
                print("Adrien capture conversion failed: \(error.localizedDescription)")
                return
            }
            if let data = converted.int16Data { onChunk(data) }
        }

        do {
            try engine.start()
        } catch {
            print("Adrien could not start capture: \(error.localizedDescription)")
        }
    }

    func stopCapture() {
        engine.inputNode.removeTap(onBus: 0)
        if engine.isRunning && playerNode == nil { engine.stop() }
    }

    // MARK: - Playback

    func startPlayback(sampleRate: Double) {
        configureSession(.playback)

        let node = AVAudioPlayerNode()
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: true
        ) else { return }

        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        playerNode = node
        playbackFormat = format

        do {
            if !engine.isRunning { try engine.start() }
            node.play()
        } catch {
            print("Adrien could not start playback: \(error.localizedDescription)")
        }
    }

    /// Schedule one chunk. Called for every binary frame off the socket.
    func enqueue(pcm: Data) {
        guard
            let node = playerNode,
            let format = playbackFormat,
            let buffer = AVAudioPCMBuffer(data: pcm, format: format)
        else { return }
        node.scheduleBuffer(buffer, completionHandler: nil)
    }

    /// Let whatever is queued finish, then tear down.
    func finishPlayback() {
        playerNode?.stop()
        detachPlayer()
    }

    /// Stop immediately, dropping anything queued. This is barge-in.
    func stopPlayback() {
        playerNode?.stop()
        detachPlayer()
    }

    private func detachPlayer() {
        if let node = playerNode { engine.detach(node) }
        playerNode = nil
        playbackFormat = nil
    }

    private func configureSession(_ mode: AVAudioSession.Category) {
        do {
            try AVAudioSession.sharedInstance().setCategory(
                .playAndRecord,
                mode: .voiceChat,          // gives us the platform's echo cancellation
                options: [.defaultToSpeaker, .allowBluetooth]
            )
            try AVAudioSession.sharedInstance().setActive(true)
        } catch {
            print("Adrien audio session failed: \(error.localizedDescription)")
        }
    }
}

// MARK: - Buffer helpers

private extension AVAudioPCMBuffer {
    /// Interleaved int16 samples as `Data`, ready for the socket.
    var int16Data: Data? {
        guard let channel = int16ChannelData else { return nil }
        return Data(bytes: channel[0], count: Int(frameLength) * MemoryLayout<Int16>.size)
    }

    /// Build a buffer from raw int16 PCM arriving off the socket.
    convenience init?(data: Data, format: AVAudioFormat) {
        let frames = AVAudioFrameCount(data.count / MemoryLayout<Int16>.size)
        guard frames > 0 else { return nil }
        self.init(pcmFormat: format, frameCapacity: frames)
        frameLength = frames
        guard let channel = int16ChannelData else { return nil }
        data.withUnsafeBytes { raw in
            guard let base = raw.bindMemory(to: Int16.self).baseAddress else { return }
            channel[0].update(from: base, count: Int(frames))
        }
    }
}
