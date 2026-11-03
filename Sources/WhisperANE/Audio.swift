@preconcurrency import AVFoundation
import Foundation

enum Audio {
    static let rate = 16_000.0

    /// Any file AVFoundation reads (WAV, M4A, MP3, CAF, ...) as 16 kHz mono Float32.
    static func load(_ url: URL) throws -> [Float] {
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: url) } catch { throw WhisperANEError.audio("cannot open \(url.lastPathComponent): \(error.localizedDescription)") }
        let format = file.processingFormat
        guard file.length > 0 else { return [] }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw WhisperANEError.audio("cannot allocate a buffer")
        }
        try file.read(into: buffer)
        if format.sampleRate == rate, format.channelCount == 1, let ch = buffer.floatChannelData?[0] {
            return Array(UnsafeBufferPointer(start: ch, count: Int(buffer.frameLength)))
        }
        let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false)!
        guard let converter = AVAudioConverter(from: format, to: target),
              let out = AVAudioPCMBuffer(pcmFormat: target,
                                         frameCapacity: AVAudioFrameCount(Double(buffer.frameLength) * rate / format.sampleRate + 64))
        else { throw WhisperANEError.audio("cannot convert \(format) to 16 kHz mono") }
        var fed = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if fed {
                status.pointee = .endOfStream
                return nil
            }
            fed = true
            status.pointee = .haveData
            return buffer
        }
        if let error { throw WhisperANEError.audio(error.localizedDescription) }
        guard let ch = out.floatChannelData?[0] else { return [] }
        return Array(UnsafeBufferPointer(start: ch, count: Int(out.frameLength)))
    }
}
