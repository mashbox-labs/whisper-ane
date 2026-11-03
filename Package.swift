// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "WhisperANE",
    platforms: [.iOS(.v18), .macOS(.v15), .macCatalyst(.v18)],
    products: [
        .library(name: "WhisperANE", targets: ["WhisperANE"]),
        .executable(name: "whisper-ane", targets: ["whisper-ane-cli"]),
    ],
    targets: [
        .target(name: "WhisperANE"),
        .executableTarget(name: "whisper-ane-cli", dependencies: ["WhisperANE"]),
        .testTarget(name: "WhisperANETests", dependencies: ["WhisperANE"]),
    ]
)
