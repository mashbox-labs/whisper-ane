import CoreML
import Foundation

/// Whisper large-v3-turbo with its encoder and decoder on the Apple Neural Engine.
///
/// ```swift
/// let whisper = try await WhisperANE.load()
/// let text = try await whisper.transcribe(URL(fileURLWithPath: "clip.m4a"))
/// ```
public actor WhisperANE {
    /// A transcription with the details behind it.
    public struct Transcript: Sendable {
        /// The text, pieces joined with a space.
        public let text: String
        /// The Whisper language code used (detected when none was given).
        public let language: String
        /// Token ids per 30 s piece, special tokens left out.
        public let tokens: [[Int]]
        /// True when a piece looked degenerate in its short window and was decoded again on the full window.
        public let usedFullWindow: Bool
        /// Seconds spent per stage: `encoder`, `prompt`, `decoder`, `total`.
        public let seconds: [String: Double]
    }

    private let runner: Runner
    /// The folder the model was loaded from.
    public nonisolated let directory: URL
    /// Whisper language codes the model takes (`en`, `th`, `ja`, `yue`, ...).
    public nonisolated let languages: Set<String>
    /// Length of one piece of audio, in seconds. Longer audio is cut into pieces of this length.
    public nonisolated let windowSeconds: Int

    private init(runner: Runner, directory: URL) {
        self.runner = runner
        self.directory = directory
        languages = runner.languageCodes
        windowSeconds = runner.buckets.last ?? 30
    }

    /// Downloads the pinned model revision once (every file checked against its SHA-256), keeps it in
    /// Application Support, and loads it. The first load on a device prepares the model for its Neural Engine and
    /// takes longer; later loads come from the system cache.
    ///
    /// - Parameters:
    ///   - pin: the model revision; the default is the one this version of the package was released with.
    ///   - cacheDirectory: where downloads are kept; the default is `Application Support/WhisperANE`.
    ///   - progress: download progress from 0 to 1, called from a background thread.
    public static func load(pin: ModelPin = .current,
                            cacheDirectory: URL? = nil,
                            computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
                            progress: (@Sendable (Double) -> Void)? = nil) async throws -> WhisperANE {
        let root = try cacheDirectory ?? ModelStore.defaultRoot()
        let folder = try await ModelStore.ensure(pin, root: root, progress: progress)
        return try await open(folder, computeUnits: computeUnits)
    }

    /// Loads a model folder you ship or host yourself (for example inside your app bundle).
    ///
    /// - Parameter pin: when given, the pinned files are copied into `Application Support/WhisperANE` (a clone on
    ///   APFS), checked there and loaded from that copy, so a file changed after the check is never loaded. Symbolic
    ///   links are refused and files the pin does not list are left out. Later loads of the same pin reuse the copy.
    public static func load(from folder: URL,
                            verifying pin: ModelPin? = nil,
                            computeUnits: MLComputeUnits = .cpuAndNeuralEngine) async throws -> WhisperANE {
        var folder = folder
        if let pin {
            let root = try ModelStore.defaultRoot()
            folder = try await Task.detached(priority: .userInitiated) { [folder] in
                try ModelStore.stage(pin, from: folder, root: root)
            }.value
        }
        return try await open(folder, computeUnits: computeUnits)
    }

    /// True when the pinned revision is already downloaded and checked, so `load()` will not touch the network.
    public static func isDownloaded(pin: ModelPin = .current, cacheDirectory: URL? = nil) -> Bool {
        guard !pin.isPlaceholder, let root = try? cacheDirectory ?? ModelStore.defaultRoot() else { return false }
        return ModelStore.isComplete(pin, root: root)
    }

    private static func open(_ folder: URL, computeUnits: MLComputeUnits) async throws -> WhisperANE {
        let runner = try await Task.detached(priority: .userInitiated) {
            try Runner(directory: folder, computeUnits: computeUnits)
        }.value
        return WhisperANE(runner: runner, directory: folder)
    }

    /// Runs every input size once on silence, so the first real transcription does not pay for it.
    public func prewarm() throws {
        try runner.prewarm()
    }

    /// Transcribes 16 kHz mono samples. `language` is a Whisper code; nil detects it from the first 30 s.
    public func transcribe(_ samples: [Float], language: String? = nil) throws -> String {
        try runner.transcribe(samples, language: language).text
    }

    /// Transcribes any audio file AVFoundation can read. `language` is a Whisper code; nil detects it.
    public func transcribe(_ file: URL, language: String? = nil) throws -> String {
        try runner.transcribe(Audio.load(file), language: language).text
    }

    public func transcription(_ samples: [Float], language: String? = nil) throws -> Transcript {
        Transcript(try runner.transcribe(samples, language: language))
    }

    public func transcription(_ file: URL, language: String? = nil) throws -> Transcript {
        Transcript(try runner.transcribe(Audio.load(file), language: language))
    }

    /// Reads an audio file as 16 kHz mono samples, the input `transcribe(_:language:)` takes.
    public static func samples(from file: URL) throws -> [Float] {
        try Audio.load(file)
    }
}

extension WhisperANE.Transcript {
    init(_ r: RunResult) {
        self.init(text: r.text, language: r.language, tokens: r.tokens, usedFullWindow: r.usedFullWindow, seconds: r.seconds)
    }
}
