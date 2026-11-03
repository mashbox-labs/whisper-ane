import Foundation

/// One exact revision of the model on Hugging Face, with the size and SHA-256 of every file in it.
/// The package only loads files that match their pin.
public struct ModelPin: Sendable, Codable, Equatable {
    public struct File: Sendable, Codable, Equatable {
        public let path: String
        public let size: Int64
        public let sha256: String

        public init(path: String, size: Int64, sha256: String) {
            self.path = path
            self.size = size
            self.sha256 = sha256
        }
    }

    /// Hugging Face repo, `owner/name`.
    public let repo: String
    /// Full commit hash of the revision.
    public let revision: String
    public let files: [File]

    public init(repo: String, revision: String, files: [File]) {
        self.repo = repo
        self.revision = revision
        self.files = files
    }

    static let placeholderRevision = "FLEX_REVISION"

    /// True until `scripts/pin.py` has written a real revision into `Pins.swift`.
    public var isPlaceholder: Bool {
        revision == Self.placeholderRevision || files.isEmpty
    }

    /// Reads a pin written by `scripts/pin.py --json`.
    public static func read(from url: URL) throws -> ModelPin {
        try JSONDecoder().decode(ModelPin.self, from: Data(contentsOf: url))
    }

    /// Rejects a repo, revision or path that could leave the model folder.
    func checkPaths() throws {
        if revision.count != 40 || !revision.allSatisfy({ ("0"..."9").contains($0) || ("a"..."f").contains($0) }) {
            throw WhisperANEError.badPin("revision must be a 40-character lowercase commit hash")
        }
        let names = repo.split(separator: "/", omittingEmptySubsequences: false)
        if names.count != 2 || names.contains(where: { name in
            guard let first = name.first, first.isASCII, first.isLetter || first.isNumber else { return true }
            return name.contains("..") || !name.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber || "._-".contains($0)) }
        }) {
            throw WhisperANEError.badPin("repo must be owner/name")
        }
        for f in files {
            let parts = f.path.split(separator: "/", omittingEmptySubsequences: false)
            if f.path.isEmpty || f.path.hasPrefix("/") || parts.contains("..") || parts.contains("") || parts.contains(".") {
                throw WhisperANEError.badPin("unsafe path \(f.path)")
            }
            if f.sha256.count != 64 || !f.sha256.allSatisfy({ $0.isHexDigit }) {
                throw WhisperANEError.badPin("bad sha256 for \(f.path)")
            }
        }
    }
}
