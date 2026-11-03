import CoreML
import Foundation

/// What one call to the runner produced.
struct RunResult: Sendable {
    var text: String
    var language: String
    var tokens: [[Int]]
    var usedFullWindow: Bool
    var seconds: [String: Double]
}

#if arch(arm64)

/// Whisper large-v3-turbo on the Neural Engine:
/// MelSpectrogram (CPU) -> AudioEncoder (Neural Engine, emits the cross-attention keys and values of the decoder
/// layers) -> TextDecoder (Neural Engine, stateful self-attention cache, one call for the prompt, then one per token).
/// Greedy decoding with Whisper's suppress lists. Audio that fits a short window runs there first; when the text looks
/// degenerate (OpenAI's compression ratio above 2.4, or the token cap) the piece is decoded again on the full window.
///
/// Not thread safe: `WhisperANE` (an actor) owns the only reference.
final class Runner: @unchecked Sendable {
    let manifest: Manifest
    let buckets: [Int]
    private let tokenizer: Tokenizer
    private var mels: [Int: MLModel] = [:]
    private var encoders: [Int: MLModel] = [:]
    private var dec1: [Int: MLModel] = [:]
    private var decK: [Int: MLModel] = [:]
    /// Token embeddings, memory mapped (fp16, vocab x d_model).
    private let embed: Data
    private let positions: [Float16]
    /// Token cap per window.
    let maxTokens = 440

    init(directory: URL, computeUnits: MLComputeUnits) throws {
        manifest = try Manifest.read(directory)
        buckets = manifest.loadedBuckets
        let C = manifest.d_model
        func file(_ name: String) throws -> URL {
            let u = directory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: u.path) else { throw WhisperANEError.missingFile(name) }
            return u
        }
        embed = try Data(contentsOf: file("embed_tokens.f16"), options: .alwaysMapped)
        guard embed.count >= manifest.vocab * C * 2 else { throw WhisperANEError.badModel("embed_tokens.f16 is too small") }
        let pd = try Data(contentsOf: file("embed_positions.f16"))
        guard pd.count >= manifest.ctx * C * 2 else { throw WhisperANEError.badModel("embed_positions.f16 is too small") }
        positions = pd.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
        tokenizer = try Tokenizer(vocabURL: file("vocab.json"), firstSpecial: manifest.eot)

        func load(_ package: String, _ function: String?, _ units: MLComputeUnits) throws -> MLModel {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = units
            cfg.functionName = function
            return try MLModel(contentsOf: file(package), configuration: cfg)
        }
        for b in buckets {
            mels[b] = try load("MelSpectrogram.mlmodelc", "mel_\(b)s", .cpuOnly)
            if manifest.onePrograms {
                // One instance per input size. Instances of one program share its Neural Engine copy of the weights,
                // and an instance that always sees the same size never reconfigures between calls.
                encoders[b] = try load("AudioEncoder.mlmodelc", nil, computeUnits)
                dec1[b] = try load("TextDecoder.mlmodelc", nil, computeUnits)
                decK[b] = try load("TextDecoder.mlmodelc", nil, computeUnits)
            } else {
                encoders[b] = try load("AudioEncoder.mlmodelc", "enc_\(b)s", computeUnits)
                dec1[b] = try load("TextDecoder.mlmodelc", "dec_\(b)s_s1", computeUnits)
                decK[b] = try load("TextDecoder.mlmodelc", "dec_\(b)s_s\(manifest.verify_s)", computeUnits)
            }
        }
    }

    var languageCodes: Set<String> { Set(manifest.lang_ids.keys) }

    /// Runs every loaded input size once on silence, so the first real call does not pay for it and a model that
    /// cannot take one of its sizes fails here.
    func prewarm() throws {
        for b in buckets {
            let cross = try encode([Float](repeating: 0, count: b * 16_000), bucket: b)
            let state = dec1[b]!.makeState()
            _ = try dec1[b]!.prediction(from: inputs([manifest.sot], position: 0, width: 1, cross: cross), using: state)
            let prompt = [manifest.lang_ids["en"] ?? manifest.sot, manifest.transcribe, manifest.notimestamps]
            _ = try decK[b]!.prediction(from: inputs(prompt, position: 1, width: manifest.verify_s, cross: cross), using: state)
        }
    }

    /// 16 kHz mono samples. Audio longer than the full window is cut into pieces of that length (no overlap).
    /// `language` nil: Whisper's language detection on the first piece, then that language for every piece.
    func transcribe(_ samples: [Float], language: String?) throws -> RunResult {
        if let language, manifest.lang_ids[language] == nil { throw WhisperANEError.unknownLanguage(language) }
        var language = language
        let full = buckets.last!
        let window = full * 16_000
        var pieces: [String] = []
        var tokens: [[Int]] = []
        var timings: [String: Double] = [:]
        var start = 0
        var fellBack = false
        while start < samples.count || pieces.isEmpty {
            let piece = Array(samples[start..<min(start + window, samples.count)])
            let bucket = buckets.first { piece.count <= $0 * 16_000 } ?? full
            var (ids, t, used) = try run(piece, bucket: bucket, language: language)
            language = used
            var text = tokenizer.decode(ids)
            if bucket != full, compressionRatio(text) > 2.4 || ids.count >= maxTokens {
                let again = try run(piece, bucket: full, language: used)
                for (k, v) in again.t { t[k, default: 0] += v }
                ids = again.ids
                text = tokenizer.decode(ids)
                fellBack = true
            }
            for (k, v) in t { timings[k, default: 0] += v }
            tokens.append(ids)
            pieces.append(text.trimmingCharacters(in: .whitespacesAndNewlines))
            start += window
        }
        return RunResult(text: pieces.filter { !$0.isEmpty }.joined(separator: " "), language: language ?? "",
                         tokens: tokens, usedFullWindow: fellBack, seconds: timings)
    }

    private func encode(_ audio: [Float], bucket b: Int) throws -> MLMultiArray {
        let N = b * 16_000
        let ain = try array([1, N], .float32)
        let ap = ain.dataPointer.bindMemory(to: Float.self, capacity: N)
        for i in 0..<N { ap[i] = i < audio.count ? audio[i] : 0 }
        guard let mel = try mels[b]!.prediction(from: MLDictionaryFeatureProvider(dictionary: ["audio": ain]))
            .featureValue(for: "mel")?.multiArrayValue else { throw WhisperANEError.badModel("no mel output") }
        let mel16 = try contiguous(mel)
        guard let encoded = try encoders[b]!.prediction(from: MLDictionaryFeatureProvider(dictionary: ["mel": mel16]))
            .featureValue(for: "cross_kv")?.multiArrayValue else { throw WhisperANEError.badModel("no encoder output") }
        // The encoder output can carry padded rows. A one-program decoder would repack them on every token, so they
        // are packed once here.
        return manifest.onePrograms ? try packRows(encoded) : encoded
    }

    private func run(_ audio: [Float], bucket b: Int, language: String?) throws -> (ids: [Int], t: [String: Double], lang: String) {
        let tA = Date()
        let cross = try encode(audio, bucket: b)
        let tC = Date()
        let state = dec1[b]!.makeState()
        var code: String
        var p0 = 0
        var prompt: [Int]
        if let language {
            guard let lang = manifest.lang_ids[language] else { throw WhisperANEError.unknownLanguage(language) }
            code = language
            prompt = [manifest.sot, lang, manifest.transcribe, manifest.notimestamps]
        } else {
            // Language detection as OpenAI's Whisper does it: the start token alone, then the most likely language.
            guard let o0 = try dec1[b]!.prediction(from: inputs([manifest.sot], position: 0, width: 1, cross: cross), using: state)
                .featureValue(for: "logits")?.multiArrayValue else { throw WhisperANEError.badModel("no logits") }
            let st = o0.strides.map(\.intValue)
            let lp = o0.dataPointer.bindMemory(to: Float16.self, capacity: 1)
            var best = ("en", -Float.infinity)
            for (c, id) in manifest.lang_ids where id < manifest.vocab {
                let v = Float(lp[id * st[1]])
                if v > best.1 { best = (c, v) }
            }
            code = best.0
            p0 = 1
            prompt = [manifest.lang_ids[code]!, manifest.transcribe, manifest.notimestamps]
        }
        var ids: [Int] = []
        guard let o = try decK[b]!.prediction(from: inputs(prompt, position: p0, width: manifest.verify_s, cross: cross), using: state)
            .featureValue(for: "logits")?.multiArrayValue else { throw WhisperANEError.badModel("no logits") }
        var t = argmax(o, column: prompt.count - 1, first: true)
        let tD = Date()
        var p = p0 + prompt.count
        while t != manifest.eot, ids.count < maxTokens, p < manifest.ctx - 1 {
            ids.append(t)
            try autoreleasepool {
                guard let o1 = try dec1[b]!.prediction(from: inputs([t], position: p, width: 1, cross: cross), using: state)
                    .featureValue(for: "logits")?.multiArrayValue else { throw WhisperANEError.badModel("no logits") }
                t = argmax(o1, column: 0, first: false)
            }
            p += 1
        }
        let tE = Date()
        return (ids, ["encoder": tC.timeIntervalSince(tA), "prompt": tD.timeIntervalSince(tC),
                      "decoder": tE.timeIntervalSince(tD), "total": tE.timeIntervalSince(tA)], code)
    }

    private func array(_ shape: [Int], _ type: MLMultiArrayDataType) throws -> MLMultiArray {
        try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: type)
    }

    /// fp16 [1, R, 1, T] with padded rows -> contiguous, one memcpy per row.
    private func packRows(_ m: MLMultiArray) throws -> MLMultiArray {
        let shape = m.shape.map(\.intValue), strides = m.strides.map(\.intValue)
        guard m.dataType == .float16, shape.count == 4, shape[0] == 1, shape[2] == 1, strides[3] == 1 else { return try contiguous(m) }
        let R = shape[1], T = shape[3]
        let out = try array(shape, .float16)
        let src = m.dataPointer.bindMemory(to: Float16.self, capacity: R * strides[1])
        let dst = out.dataPointer.bindMemory(to: Float16.self, capacity: R * T)
        if strides[1] == T {
            memcpy(dst, src, R * T * 2)
        } else {
            for r in 0..<R { memcpy(dst + r * T, src + r * strides[1], T * 2) }
        }
        return out
    }

    /// A contiguous fp16 copy (Neural Engine outputs can carry padded strides).
    private func contiguous(_ m: MLMultiArray) throws -> MLMultiArray {
        let shape = m.shape.map(\.intValue), strides = m.strides.map(\.intValue)
        let out = try array(shape, .float16)
        let dst = out.dataPointer.bindMemory(to: Float16.self, capacity: m.count)
        var idx = [Int](repeating: 0, count: shape.count)
        let src16 = m.dataType == .float16 ? m.dataPointer.bindMemory(to: Float16.self, capacity: 1) : nil
        let src32 = m.dataType == .float32 ? m.dataPointer.bindMemory(to: Float.self, capacity: 1) : nil
        for i in 0..<m.count {
            var off = 0
            for k in 0..<shape.count { off += idx[k] * strides[k] }
            dst[i] = src16?[off] ?? Float16(src32?[off] ?? 0)
            var k = shape.count - 1
            while k >= 0 {
                idx[k] += 1
                if idx[k] < shape[k] { break }
                idx[k] = 0
                k -= 1
            }
        }
        return out
    }

    private func inputs(_ tokens: [Int], position p0: Int, width S: Int, cross: MLMultiArray) throws -> MLDictionaryFeatureProvider {
        let C = manifest.d_model, CTX = manifest.ctx
        let x = try array([1, C, 1, S], .float16)
        let mask = try array([1, 1, S, CTX], .float16)
        let wr = try array([S, CTX], .float16)
        let xp = x.dataPointer.bindMemory(to: Float16.self, capacity: C * S)
        let mp = mask.dataPointer.bindMemory(to: Float16.self, capacity: S * CTX)
        let wp = wr.dataPointer.bindMemory(to: Float16.self, capacity: S * CTX)
        embed.withUnsafeBytes { raw in
            let e = raw.bindMemory(to: Float16.self)
            for j in 0..<S {
                let t = j < tokens.count ? tokens[j] : manifest.eot
                let p = min(p0 + j, CTX - 1)
                for c in 0..<C { xp[c * S + j] = Float16(Float(e[t * C + c]) + Float(positions[p * C + c])) }
                for k in 0..<CTX {
                    mp[j * CTX + k] = k <= p ? 0 : -10000
                    wp[j * CTX + k] = k == p ? 1 : 0
                }
            }
        }
        return try MLDictionaryFeatureProvider(dictionary: ["x": x, "cross_kv": cross, "mask": mask, "wr": wr])
    }

    private func argmax(_ logits: MLMultiArray, column j: Int, first: Bool) -> Int {
        let V = manifest.vocab, strides = logits.strides.map(\.intValue)
        let p = logits.dataPointer.bindMemory(to: Float16.self, capacity: 1)
        var l = [Float](repeating: 0, count: V)
        for v in 0..<V { l[v] = Float(p[v * strides[1] + j * strides[3]]) }
        for s in manifest.suppress where s < V { l[s] = -.infinity }
        if first { for s in manifest.begin_suppress where s < V { l[s] = -.infinity } }
        var best = 0, bestV = -Float.infinity
        for (i, v) in l.enumerated() where v > bestV {
            bestV = v
            best = i
        }
        return best
    }
}

#else

/// Intel Macs have no Neural Engine and no Swift `Float16`: the API is there so a universal build links, and loading
/// throws.
final class Runner: @unchecked Sendable {
    let buckets: [Int] = []
    var languageCodes: Set<String> { [] }
    init(directory: URL, computeUnits: MLComputeUnits) throws { throw WhisperANEError.unsupportedDevice }
    func prewarm() throws { throw WhisperANEError.unsupportedDevice }
    func transcribe(_ samples: [Float], language: String?) throws -> RunResult { throw WhisperANEError.unsupportedDevice }
}

#endif
