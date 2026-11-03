// Written by scripts/pin.py. Do not edit by hand.

extension ModelPin {
    /// The model revision this version of the package loads.
    ///
    /// PLACEHOLDER: `FLEX_REVISION` is replaced by the real commit hash, and `files` by the size and SHA-256 of each
    /// file, when `python3 scripts/pin.py <commit>` is run for the release. Until then `WhisperANE.load()` throws
    /// `WhisperANEError.notPinned`.
    public static let current = ModelPin(
        repo: "mashbox-labs/whisper-large-v3-turbo-coreml-ane",
        revision: "FLEX_REVISION",
        files: []
    )
}
