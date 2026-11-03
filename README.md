# WhisperANE

Speech to text in your Swift app with OpenAI's Whisper large-v3-turbo, running on the Apple Neural Engine.

One Swift package, no dependencies. It downloads our Core ML build of the model from
[Hugging Face](https://huggingface.co/mashbox-labs/whisper-large-v3-turbo-coreml-ane) at a pinned revision, checks
every file against its SHA-256, keeps it on the device, and takes any of Whisper's 100 languages.

```swift
import WhisperANE

let whisper = try await WhisperANE.load()
let text = try await whisper.transcribe(URL(fileURLWithPath: "clip.m4a"))
```

## Quick start

1. **Add the package.** In Xcode: File > Add Package Dependencies, then
   `https://github.com/mashbox-labs/whisper-ane`. Or in `Package.swift`:

   ```swift
   .package(url: "https://github.com/mashbox-labs/whisper-ane", from: "0.1.0")
   ```

   and add `"WhisperANE"` to your target's dependencies.

2. **Load the model once**, for example when your app starts:

   ```swift
   let whisper = try await WhisperANE.load { progress in
       print("download", progress)   // 0...1, only on the first run
   }
   ```

3. **Transcribe** a file or samples:

   ```swift
   let text = try await whisper.transcribe(fileURL)                  // any format AVFoundation reads
   let thai = try await whisper.transcribe(fileURL, language: "th")  // skip language detection
   let fromMic = try await whisper.transcribe(samples)               // [Float], 16 kHz mono
   ```

On a Mac you can try it from the command line:

```sh
git clone https://github.com/mashbox-labs/whisper-ane && cd whisper-ane
swift run -c release whisper-ane clip.wav
```

## Speed and accuracy

From the [model card](https://huggingface.co/mashbox-labs/whisper-large-v3-turbo-coreml-ane): a MacBook Air (M5),
the same 500 FLEURS test clips (100 per language), greedy decoding, no timestamps, the model already loaded. Time is
end to end, from audio to text, median per clip.

| Language | Median per clip | Error rate |
|---|---|---|
| English | 180 ms | 5.69% WER |
| Japanese | 363 ms | 4.97% CER |
| Korean | 339 ms | 3.63% CER |
| Chinese (Mandarin) | 340 ms | 8.23% CER |
| Thai | 537 ms | 13.76% CER |

On the same clips and the same Mac, this build (8-bit weights) is 2.0x to 3.2x faster than
[Argmax WhisperKit](https://github.com/argmaxinc/WhisperKit) v1.1.1 with its 632 MB compressed turbo model, and 1.4x
to 2.9x faster than [whisper.cpp](https://github.com/ggml-org/whisper.cpp) v1.9.5 with its Core ML encoder and
OpenAI's weights at full precision (fp16). The encoder takes 64.4 ms for
a 10 s window and 375.0 ms for a 30 s window. On iPhone Air, a 5 s Thai clip transcribes in 1.05 s, and the model
loads from the system cache in 1.84 s. The card has the full speed and accuracy tables for all three, and names the five languages scored.

## Requirements

- iOS 18, iPadOS 18, macOS 15 or Mac Catalyst 18, or later.
- Apple silicon. On an Intel Mac the package builds, and `load()` throws `WhisperANEError.unsupportedDevice`.
- Swift 6 toolchain (Xcode 16 or later).
- Not meant for app extensions such as keyboards: their memory limit is too small for this model.

## How it works

1. **Mel spectrogram** on the CPU (a Core ML model).
2. **Encoder** on the Neural Engine. Audio up to 10 s runs in a 10 s window, longer audio in a 30 s window.
3. **Decoder** on the Neural Engine, with its attention cache kept in Core ML state. The prompt is read in one call,
   then one call per token, greedy, with Whisper's suppress lists.
4. If the text from the 10 s window looks degenerate (OpenAI's compression-ratio check, or the token cap), that piece
   is decoded again in the 30 s window.

With no language given, the language is detected from the first 30 s, the way OpenAI's Whisper does it, and kept for
the rest of the audio.

The first load on a device prepares the model for its Neural Engine and takes noticeably longer. Later loads come from
the system cache. Call `load()` early, keep the instance, and reuse it.

## API

```swift
public actor WhisperANE {
    // Download (once) and load the pinned revision.
    static func load(pin: ModelPin = .current, cacheDirectory: URL? = nil,
                     computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
                     progress: (@Sendable (Double) -> Void)? = nil) async throws -> WhisperANE
    // Load a folder you ship or host yourself; with a pin, every file is checked first.
    static func load(from folder: URL, verifying pin: ModelPin? = nil,
                     computeUnits: MLComputeUnits = .cpuAndNeuralEngine) async throws -> WhisperANE
    static func isDownloaded(pin: ModelPin = .current, cacheDirectory: URL? = nil) -> Bool
    static func samples(from file: URL) throws -> [Float]          // 16 kHz mono

    func transcribe(_ file: URL, language: String? = nil) throws -> String
    func transcribe(_ samples: [Float], language: String? = nil) throws -> String
    func transcription(_ file: URL, language: String? = nil) throws -> Transcript
    func transcription(_ samples: [Float], language: String? = nil) throws -> Transcript
    func prewarm() throws                                           // optional: run each input size once

    nonisolated let languages: Set<String>                          // Whisper codes: en, th, ja, yue, ...
    nonisolated let windowSeconds: Int
    nonisolated let directory: URL
}

extension WhisperANE {
    public struct Transcript { let text: String; let language: String; let tokens: [[Int]]
                               let usedFullWindow: Bool; let seconds: [String: Double] }
}
```

`WhisperANE` is an actor, so calls from several tasks run one at a time.

### Shipping the model inside your app

Put the files of the pinned revision in a folder reference in your app (keeping their names), then:

```swift
let folder = Bundle.main.url(forResource: "whisper-large-v3-turbo-coreml-ane", withExtension: nil)!
let whisper = try await WhisperANE.load(from: folder, verifying: .current)
```

`verifying:` hashes every file on each load, which takes a moment on a phone. Leave it out once your build is checked.

### Model revision

`ModelPin.current` in `Sources/WhisperANE/Pins.swift` names the Hugging Face commit this version of the package
loads, with the size and SHA-256 of every file. `scripts/pin.py <commit>` rewrites it from the Hugging Face API. A
file that does not match is never used.

## Limits

- Audio longer than 30 s is cut into 30 s pieces with no overlap, so a word on a cut can be split.
- Text only: no timestamps, no translation, greedy decoding only.
- Runs on the device. The only network access is the model download from huggingface.co.

## Example app

[`Examples/RecordAndTranscribe.swift`](Examples/RecordAndTranscribe.swift) is a SwiftUI view that records from the
microphone and transcribes when you stop. Add `NSMicrophoneUsageDescription` to your Info.plist.

## Tests

```sh
swift test
```

The unit tests need no model and no Neural Engine. To run the end-to-end test too, point it at a model folder and a
recording of JFK's "ask not what your country can do for you":

```sh
WHISPER_ANE_MODEL=/path/to/model WHISPER_ANE_AUDIO=/path/to/jfk.wav swift test
```

## Credits and licence

- Model: [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) by OpenAI, MIT licence. The
  model repo on Hugging Face carries OpenAI's `LICENSE` text unchanged.
- Decoding steps follow [OpenAI's Whisper](https://github.com/openai/whisper) (MIT).
- Conversion: [Core ML Tools](https://github.com/apple/coremltools) by Apple.
- Compared against: [Argmax WhisperKit](https://github.com/argmaxinc/WhisperKit) (MIT) and
  [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (MIT).
- Evaluation data: [FLEURS](https://huggingface.co/datasets/google/fleurs) by Google, CC BY 4.0.

The code in this repository is Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).

The same model runs in [Free Flow](https://mashbox.gg/freeflow), our on-device dictation app, for languages such as
Thai, Korean, Japanese and Chinese. Built by [MASHBOX Labs](https://mashbox.gg).
