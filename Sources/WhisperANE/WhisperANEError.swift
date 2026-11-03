import Foundation

public enum WhisperANEError: Error, LocalizedError, Equatable {
    /// The package was built without a pinned model revision.
    case notPinned
    case badPin(String)
    case download(String)
    case checksumMismatch(path: String)
    case missingFile(String)
    case badModel(String)
    case unknownLanguage(String)
    case audio(String)
    /// The Neural Engine build needs an Apple silicon device.
    case unsupportedDevice

    public var errorDescription: String? {
        switch self {
        case .notPinned: return "No model revision is pinned in this build of WhisperANE."
        case .badPin(let s): return "Bad model pin: \(s)"
        case .download(let s): return "Download failed: \(s)"
        case .checksumMismatch(let p): return "\(p) does not match its pinned SHA-256."
        case .missingFile(let p): return "Missing model file: \(p)"
        case .badModel(let s): return "Model error: \(s)"
        case .unknownLanguage(let c): return "Unknown language code: \(c)"
        case .audio(let s): return "Audio error: \(s)"
        case .unsupportedDevice: return "WhisperANE needs an Apple silicon device."
        }
    }
}
