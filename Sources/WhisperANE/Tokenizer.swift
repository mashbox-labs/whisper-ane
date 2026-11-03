import Compression
import Foundation

/// Whisper's byte-level BPE, decode side only.
struct Tokenizer: Sendable {
    private let idToToken: [Int: String]
    private let unicodeToByte: [Character: UInt8]
    /// Ids at or above this one are special tokens and are never printed.
    let firstSpecial: Int

    init(vocab: [String: Int], firstSpecial: Int) {
        var inv: [Int: String] = [:]
        for (k, v) in vocab { inv[v] = k }
        idToToken = inv
        unicodeToByte = Self.unicodeToByteTable()
        self.firstSpecial = firstSpecial
    }

    init(vocabURL: URL, firstSpecial: Int) throws {
        guard let vocab = try JSONSerialization.jsonObject(with: Data(contentsOf: vocabURL)) as? [String: Int] else {
            throw WhisperANEError.badModel("vocab.json is not a token map")
        }
        self.init(vocab: vocab, firstSpecial: firstSpecial)
    }

    /// GPT-2 `bytes_to_unicode`, inverted.
    static func unicodeToByteTable() -> [Character: UInt8] {
        var bs: [Int] = Array(33...126) + Array(161...172) + Array(174...255)
        var cs = bs
        var n = 0
        for b in 0..<256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var table: [Character: UInt8] = [:]
        for (b, c) in zip(bs, cs) { table[Character(Unicode.Scalar(c)!)] = UInt8(b) }
        return table
    }

    func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        for i in ids where i < firstSpecial {
            guard let tok = idToToken[i] else { continue }
            for ch in tok { if let b = unicodeToByte[ch] { bytes.append(b) } }
        }
        return String(decoding: bytes, as: UTF8.self)
    }
}

/// OpenAI's compression-ratio check (zlib on the UTF-8 text). The Compression framework writes the raw deflate
/// stream; zlib's 6 framing bytes are added so the ratio is on the same scale as zlib's.
func compressionRatio(_ s: String) -> Double {
    let src = Array(s.utf8)
    guard !src.isEmpty else { return 0 }
    let cap = src.count + 64
    var dst = [UInt8](repeating: 0, count: cap)
    let n = src.withUnsafeBufferPointer { sp in
        dst.withUnsafeMutableBufferPointer { dp in
            compression_encode_buffer(dp.baseAddress!, cap, sp.baseAddress!, src.count, nil, COMPRESSION_ZLIB)
        }
    }
    guard n > 0 else { return 0 }
    return Double(src.count) / Double(n + 6)
}
