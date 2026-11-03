import Foundation
import XCTest
@testable import WhisperANE

final class TokenizerTests: XCTestCase {
    func testByteTableCoversEveryByte() {
        let table = Tokenizer.unicodeToByteTable()
        XCTAssertEqual(table.count, 256)
        XCTAssertEqual(Set(table.values).count, 256)
        XCTAssertEqual(table["\u{0120}"], 0x20)   // GPT-2 writes a space as U+0120
        XCTAssertEqual(table["A"], 0x41)
    }

    func testDecodeJoinsWordsAndDropsSpecialTokens() {
        let t = Tokenizer(vocab: ["Hello": 0, "\u{0120}world": 1, "<|endoftext|>": 5], firstSpecial: 5)
        XCTAssertEqual(t.decode([0, 1, 5]), "Hello world")
        XCTAssertEqual(t.decode([5, 6, 0]), "Hello")
    }

    func testDecodeRebuildsMultiByteCharactersAcrossTokens() {
        // "é" is C3 A9 in UTF-8; GPT-2 maps those bytes to "Ã" and "©", here split over two tokens.
        let t = Tokenizer(vocab: ["\u{00C3}": 0, "\u{00A9}": 1], firstSpecial: 10)
        XCTAssertEqual(t.decode([0, 1]), "é")
    }
}

final class CompressionRatioTests: XCTestCase {
    func testEmptyIsZero() {
        XCTAssertEqual(compressionRatio(""), 0)
    }

    // Reference ratios from Python's zlib.compress on the same UTF-8 text.
    func testMatchesZlibOnNormalText() {
        let s = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
        XCTAssertEqual(compressionRatio(s), 107.0 / 79.0, accuracy: 0.05)
        XCTAssertLessThan(compressionRatio(s), 2.4)
    }

    func testMatchesZlibOnThai() {
        XCTAssertEqual(compressionRatio("สวัสดีครับ วันนี้อากาศดีมาก"), 79.0 / 62.0, accuracy: 0.05)
    }

    func testRepeatedTextIsAboveTheFallbackThreshold() {
        let s = Array(repeating: "the", count: 24).joined(separator: " ")
        XCTAssertGreaterThan(compressionRatio(s), 2.4)
    }
}

final class PinTests: XCTestCase {
    func testShippedPinIsEitherPlaceholderOrComplete() {
        let pin = ModelPin.current
        XCTAssertEqual(pin.repo, "mashbox-labs/whisper-large-v3-turbo-coreml-ane")
        if pin.isPlaceholder {
            XCTAssertEqual(pin.revision, "FLEX_REVISION")
        } else {
            XCTAssertEqual(pin.revision.count, 40)
            XCTAssertTrue(pin.files.contains { $0.path == "manifest.json" })
            XCTAssertNoThrow(try pin.checkPaths())
        }
    }

    func testPlaceholderPinRefusesToLoad() async {
        let pin = ModelPin(repo: "mashbox-labs/whisper-large-v3-turbo-coreml-ane", revision: "FLEX_REVISION", files: [])
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        do {
            _ = try await WhisperANE.load(pin: pin, cacheDirectory: root)
            XCTFail("loaded a placeholder pin")
        } catch {
            XCTAssertEqual(error as? WhisperANEError, .notPinned)
        }
        XCTAssertFalse(WhisperANE.isDownloaded(pin: pin, cacheDirectory: root))
    }

    func testUnsafePathsAreRejected() {
        let sha = String(repeating: "0", count: 64)
        for path in ["../x", "/etc/passwd", "a/../../b", "a//b", "", "./a"] {
            let pin = ModelPin(repo: "r/r", revision: "x", files: [.init(path: path, size: 1, sha256: sha)])
            XCTAssertThrowsError(try pin.checkPaths(), path)
        }
        let bad = ModelPin(repo: "r/r", revision: "x", files: [.init(path: "a", size: 1, sha256: "zz")])
        XCTAssertThrowsError(try bad.checkPaths())
    }

    func testPinRoundTripsAsJSON() throws {
        let pin = ModelPin(repo: "r/r", revision: String(repeating: "a", count: 40),
                           files: [.init(path: "manifest.json", size: 3, sha256: String(repeating: "b", count: 64))])
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".json")
        try JSONEncoder().encode(pin).write(to: url)
        XCTAssertEqual(try ModelPin.read(from: url), pin)
    }

    func testDownloadURLIsPinnedToTheRevision() throws {
        let pin = ModelPin(repo: "mashbox-labs/m", revision: "abc", files: [])
        let u = try ModelStore.url(pin, .init(path: "AudioEncoder.mlmodelc/weights/weight.bin", size: 1, sha256: ""))
        XCTAssertEqual(u.absoluteString, "https://huggingface.co/mashbox-labs/m/resolve/abc/AudioEncoder.mlmodelc/weights/weight.bin")
    }
}

final class ChecksumTests: XCTestCase {
    private func folder(with files: [String: String]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        for (name, body) in files {
            let u = dir.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: u.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(body.utf8).write(to: u)
        }
        return dir
    }

    func testSHA256() throws {
        let dir = try folder(with: ["abc": "abc"])
        XCTAssertEqual(try ModelStore.sha256(of: dir.appendingPathComponent("abc")),
                       "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }

    func testVerifyAcceptsMatchingFiles() throws {
        let dir = try folder(with: ["sub/abc": "abc"])
        let pin = ModelPin(repo: "r/r", revision: "x", files: [
            .init(path: "sub/abc", size: 3, sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        ])
        XCTAssertNoThrow(try ModelStore.verify(pin, folder: dir))
    }

    func testVerifyRejectsAChangedFile() throws {
        let dir = try folder(with: ["abc": "abd"])
        let pin = ModelPin(repo: "r/r", revision: "x", files: [
            .init(path: "abc", size: 3, sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        ])
        XCTAssertThrowsError(try ModelStore.verify(pin, folder: dir)) {
            XCTAssertEqual($0 as? WhisperANEError, .checksumMismatch(path: "abc"))
        }
    }

    func testVerifyRejectsAWrongSizeAndAMissingFile() throws {
        let dir = try folder(with: ["abc": "abc"])
        let sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        XCTAssertThrowsError(try ModelStore.verify(ModelPin(repo: "r/r", revision: "x", files: [.init(path: "abc", size: 4, sha256: sha)]), folder: dir))
        XCTAssertThrowsError(try ModelStore.verify(ModelPin(repo: "r/r", revision: "x", files: [.init(path: "nope", size: 3, sha256: sha)]), folder: dir)) {
            XCTAssertEqual($0 as? WhisperANEError, .missingFile("nope"))
        }
    }

    func testCompleteFolderNeedsTheMarkerAndEverySize() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        let pin = ModelPin(repo: "o/m", revision: "rev1", files: [.init(path: "abc", size: 3, sha256: sha)])
        let dir = ModelStore.folder(for: pin, root: root)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("abc".utf8).write(to: dir.appendingPathComponent("abc"))
        XCTAssertFalse(ModelStore.isComplete(pin, root: root))
        try "rev1".write(to: dir.appendingPathComponent(ModelStore.completeMarker), atomically: true, encoding: .utf8)
        XCTAssertTrue(ModelStore.isComplete(pin, root: root))
        try Data("ab".utf8).write(to: dir.appendingPathComponent("abc"))
        XCTAssertFalse(ModelStore.isComplete(pin, root: root))
    }
}

final class ManifestTests: XCTestCase {
    private func manifest(buckets: [Int], onePrograms: Bool?) throws -> Manifest {
        var obj: [String: Any] = [
            "model": "m", "buckets": buckets, "d_model": 4, "ctx": 8, "vocab": 10, "verify_s": 8, "sot": 1, "eot": 0,
            "transcribe": 2, "notimestamps": 3, "lang_ids": ["en": 4], "suppress": [5], "begin_suppress": [0],
        ]
        if let onePrograms { obj["one_program"] = onePrograms }
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: obj).write(to: dir.appendingPathComponent("manifest.json"))
        return try Manifest.read(dir)
    }

    func testOneProgramLoadsEveryWindow() throws {
        XCTAssertEqual(try manifest(buckets: [30, 10], onePrograms: true).loadedBuckets, [10, 30])
    }

    func testOneFunctionPerWindowLoadsTenAndThirty() throws {
        XCTAssertEqual(try manifest(buckets: [5, 10, 20, 30], onePrograms: nil).loadedBuckets, [10, 30])
        XCTAssertEqual(try manifest(buckets: [5, 30], onePrograms: false).loadedBuckets, [30])
    }

    func testMissingManifestIsReported() {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        XCTAssertThrowsError(try Manifest.read(dir)) { XCTAssertEqual($0 as? WhisperANEError, .missingFile("manifest.json")) }
    }
}

/// Runs only when WHISPER_ANE_MODEL points at a model folder and WHISPER_ANE_AUDIO at a clip of the JFK quote.
final class EndToEndTests: XCTestCase {
    func testTranscribesAClip() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let model = env["WHISPER_ANE_MODEL"], let audio = env["WHISPER_ANE_AUDIO"] else {
            throw XCTSkip("set WHISPER_ANE_MODEL and WHISPER_ANE_AUDIO to run")
        }
        let whisper = try await WhisperANE.load(from: URL(fileURLWithPath: model))
        let t = try await whisper.transcription(URL(fileURLWithPath: audio), language: nil)
        XCTAssertEqual(t.language, "en")
        XCTAssertTrue(t.text.lowercased().contains("ask not what your country can do for you"), t.text)
    }
}
