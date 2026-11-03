import Foundation

/// `manifest.json` in the model folder: the model's sizes, special tokens and input lengths.
struct Manifest: Decodable, Sendable {
    let model: String
    /// Input windows in seconds, smallest first after sorting.
    let buckets: [Int]
    let d_model: Int
    let ctx: Int
    let vocab: Int
    /// Tokens per call of the wide decoder (the prompt is read in one call).
    let verify_s: Int
    let sot: Int
    let eot: Int
    let transcribe: Int
    let notimestamps: Int
    let lang_ids: [String: Int]
    let suppress: [Int]
    let begin_suppress: [Int]
    /// true: AudioEncoder and TextDecoder are each one program that takes every input size.
    let one_program: Bool?

    static func read(_ folder: URL) throws -> Manifest {
        let url = folder.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: url.path) else { throw WhisperANEError.missingFile("manifest.json") }
        let m = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: url))
        guard !m.buckets.isEmpty, m.buckets.allSatisfy({ $0 > 0 }) else { throw WhisperANEError.badModel("manifest has no buckets") }
        return m
    }

    var onePrograms: Bool { one_program == true }

    /// The windows to load. A one-program build holds one Neural Engine copy of the weights for every window, so it
    /// loads them all. A build with one function per window holds one copy per window, so only the 10 s window (when
    /// present) and the full window are loaded.
    var loadedBuckets: [Int] {
        let all = buckets.sorted()
        if onePrograms { return all }
        var use = all.filter { $0 == 10 }
        use.append(all.last!)
        return Array(Set(use)).sorted()
    }
}
