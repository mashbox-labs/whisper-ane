// A SwiftUI view that records from the microphone and transcribes when you stop.
// Add this file to an iOS 18 app target that depends on WhisperANE, and add NSMicrophoneUsageDescription to Info.plist.

import AVFoundation
import SwiftUI
import WhisperANE

@MainActor
final class Dictation: ObservableObject {
    @Published var status = "Loading model..."
    @Published var text = ""
    @Published var recording = false
    private var whisper: WhisperANE?
    private var recorder: AVAudioRecorder?
    private let file = FileManager.default.temporaryDirectory.appendingPathComponent("dictation.wav")

    func load() async {
        do {
            whisper = try await WhisperANE.load { p in
                Task { @MainActor in self.status = "Downloading \(Int(p * 100))%" }
            }
            status = "Ready"
        } catch {
            status = error.localizedDescription
        }
    }

    func toggle() async {
        if recording { await stop() } else { start() }
    }

    private func start() {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement)
            try session.setActive(true)
            let settings: [String: Any] = [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 16_000,
                                           AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16]
            recorder = try AVAudioRecorder(url: file, settings: settings)
            recorder?.record()
            recording = true
            status = "Listening"
        } catch {
            status = error.localizedDescription
        }
    }

    private func stop() async {
        recorder?.stop()
        recording = false
        guard let whisper else { return }
        status = "Transcribing"
        do {
            text = try await whisper.transcribe(file)
            status = "Ready"
        } catch {
            status = error.localizedDescription
        }
    }
}

struct RecordAndTranscribeView: View {
    @StateObject private var dictation = Dictation()

    var body: some View {
        VStack(spacing: 24) {
            Text(dictation.status).font(.footnote).foregroundStyle(.secondary)
            Text(dictation.text).frame(maxWidth: .infinity, alignment: .leading)
            Button(dictation.recording ? "Stop" : "Record") {
                Task { await dictation.toggle() }
            }
            .buttonStyle(.borderedProminent)
            .disabled(dictation.status.hasPrefix("Loading") || dictation.status.hasPrefix("Downloading"))
        }
        .padding()
        .task { await dictation.load() }
    }
}
