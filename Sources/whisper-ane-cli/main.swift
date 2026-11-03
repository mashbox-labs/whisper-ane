import Foundation
import WhisperANE

let usage = """
usage: whisper-ane [options] AUDIO...

Transcribes audio files with Whisper large-v3-turbo on the Neural Engine.
The pinned model is downloaded on first use and kept in Application Support.

options:
  --language CODE   Whisper language code (en, th, ja, ...). Default: detect.
  --model DIR       load a model folder instead of the pinned download
  --pin FILE        use a pin written by scripts/pin.py --json
  --cache DIR       where downloads are kept
  --prewarm         run every input size once before transcribing
  --json            one JSON line per file: text, language, tokens, seconds
"""

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

var language: String?
var model: URL?
var pinFile: URL?
var cache: URL?
var json = false
var prewarm = false
var files: [URL] = []

var args = CommandLine.arguments.dropFirst()
while let a = args.popFirst() {
    func value() -> String {
        guard let v = args.popFirst() else { fail("\(a) needs a value\n\n\(usage)") }
        return v
    }
    switch a {
    case "--language": language = value()
    case "--model": model = URL(fileURLWithPath: value())
    case "--pin": pinFile = URL(fileURLWithPath: value())
    case "--cache": cache = URL(fileURLWithPath: value())
    case "--json": json = true
    case "--prewarm": prewarm = true
    case "-h", "--help": print(usage); exit(0)
    default:
        if a.hasPrefix("--") { fail("unknown option \(a)\n\n\(usage)") }
        files.append(URL(fileURLWithPath: a))
    }
}
if files.isEmpty { fail(usage) }

do {
    let pin = try pinFile.map(ModelPin.read(from:))
    let started = Date()
    let whisper: WhisperANE
    if let model {
        whisper = try await WhisperANE.load(from: model, verifying: pin)
    } else {
        let lastShown = LockedInt()
        whisper = try await WhisperANE.load(pin: pin ?? .current, cacheDirectory: cache) { p in
            let pct = Int(p * 100)
            if lastShown.exchange(pct) != pct, pct % 5 == 0 {
                FileHandle.standardError.write(Data("download \(pct)%\n".utf8))
            }
        }
    }
    if prewarm { try await whisper.prewarm() }
    FileHandle.standardError.write(Data(String(format: "loaded in %.2f s\n", Date().timeIntervalSince(started)).utf8))
    for f in files {
        let t = try await whisper.transcription(f, language: language)
        if json {
            let row: [String: Any] = ["file": f.lastPathComponent, "text": t.text, "language": t.language,
                                      "tokens": t.tokens, "full_window": t.usedFullWindow, "seconds": t.seconds]
            let data = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys, .withoutEscapingSlashes])
            print(String(decoding: data, as: UTF8.self))
        } else {
            print(files.count > 1 ? "\(f.lastPathComponent): \(t.text)" : t.text)
        }
    }
} catch {
    fail("error: \(error.localizedDescription)")
}

final class LockedInt: @unchecked Sendable {
    private var value = -1
    private let lock = NSLock()
    func exchange(_ v: Int) -> Int {
        lock.lock()
        defer { lock.unlock() }
        let old = value
        value = v
        return old
    }
}
