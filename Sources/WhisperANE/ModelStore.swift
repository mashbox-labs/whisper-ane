import CryptoKit
import Darwin
import Foundation

/// Downloads a pinned revision from Hugging Face once, checks every file's size and SHA-256, and keeps it in
/// `<root>/<repo>/<revision>/`. A folder is only used after every file in it has matched.
enum ModelStore {
    static let completeMarker = ".complete"

    static func defaultRoot() throws -> URL {
        try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent("WhisperANE", isDirectory: true)
    }

    static func folder(for pin: ModelPin, root: URL) -> URL {
        root.appendingPathComponent(pin.repo.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
            .appendingPathComponent(pin.revision, isDirectory: true)
    }

    /// The pin's folder and its download folder, both checked to stay under `root`.
    static func paths(for pin: ModelPin, root: URL) throws -> (final: URL, partial: URL) {
        try pin.checkPaths()
        let final = folder(for: pin, root: root).standardizedFileURL
        let partial = final.deletingLastPathComponent().appendingPathComponent(pin.revision + ".partial", isDirectory: true).standardizedFileURL
        let base = root.standardizedFileURL.path.hasSuffix("/") ? root.standardizedFileURL.path : root.standardizedFileURL.path + "/"
        guard final.path.hasPrefix(base), partial.path.hasPrefix(base) else { throw WhisperANEError.badPin("model folder leaves the cache") }
        return (final, partial)
    }

    /// True when the pinned revision is already downloaded and checked.
    static func isComplete(_ pin: ModelPin, root: URL) -> Bool {
        guard let dir = try? paths(for: pin, root: root).final else { return false }
        guard let marker = try? String(contentsOf: dir.appendingPathComponent(completeMarker), encoding: .utf8),
              marker.trimmingCharacters(in: .whitespacesAndNewlines) == pin.revision else { return false }
        return pin.files.allSatisfy { f in
            (try? FileManager.default.attributesOfItem(atPath: dir.appendingPathComponent(f.path).path)[.size] as? NSNumber)?.int64Value == f.size
        }
    }

    static func ensure(_ pin: ModelPin, root: URL, progress: (@Sendable (Double) -> Void)?) async throws -> URL {
        if pin.isPlaceholder { throw WhisperANEError.notPinned }
        let fm = FileManager.default
        let (final, partial) = try paths(for: pin, root: root)
        if isComplete(pin, root: root) {
            progress?(1)
            return final
        }
        try fm.createDirectory(at: partial, withIntermediateDirectories: true)
        let total = max(1, pin.files.reduce(Int64(0)) { $0 + $1.size })
        var done: Int64 = 0
        progress?(0)
        for f in pin.files {
            let dest = partial.appendingPathComponent(f.path)
            if fm.fileExists(atPath: dest.path), (try? check(dest, f)) != nil {
                done += f.size
                progress?(Double(done) / Double(total))
                continue
            }
            try fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
            let base = done
            let tmp = try await download(pin, f) { fraction in
                progress?((Double(base) + fraction * Double(f.size)) / Double(total))
            }
            do {
                try check(tmp, f)
            } catch {
                try? fm.removeItem(at: tmp)
                throw error
            }
            if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
            try fm.moveItem(at: tmp, to: dest)
            done += f.size
            progress?(Double(done) / Double(total))
        }
        return try finish(pin, partial, final)
    }

    /// Marks `partial` complete and moves it into place.
    private static func finish(_ pin: ModelPin, _ partial: URL, _ final: URL) throws -> URL {
        let fm = FileManager.default
        try pin.revision.write(to: partial.appendingPathComponent(completeMarker), atomically: true, encoding: .utf8)
        if fm.fileExists(atPath: final.path) { try fm.removeItem(at: final) }
        try fm.moveItem(at: partial, to: final)
        var url = final
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? url.setResourceValues(values)
        return final
    }

    /// Copies every pinned file from a folder the package does not own into the cache and checks the copy, so the
    /// bytes that load are the bytes that were hashed. Files the pin does not list are not copied, and a symbolic
    /// link anywhere under `source` is refused. On APFS the copy is a clone: no extra disk, and later writes to
    /// `source` do not reach it.
    static func stage(_ pin: ModelPin, from source: URL, root: URL) throws -> URL {
        if pin.isPlaceholder { throw WhisperANEError.notPinned }
        let fm = FileManager.default
        let (final, partial) = try paths(for: pin, root: root)
        if isComplete(pin, root: root) { return final }
        for f in pin.files {
            let dest = partial.appendingPathComponent(f.path)
            try fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? fm.removeItem(at: dest)
            try copyChecked(source, f, to: dest)
        }
        return try finish(pin, partial, final)
    }

    private static func copyChecked(_ source: URL, _ f: ModelPin.File, to dest: URL) throws {
        var u = source
        var st = stat()
        for part in f.path.split(separator: "/") {
            u.appendPathComponent(String(part))
            guard lstat(u.path, &st) == 0 else { throw WhisperANEError.missingFile(f.path) }
            if st.st_mode & S_IFMT == S_IFLNK { throw WhisperANEError.badModel("\(f.path) is a symbolic link") }
        }
        let src = open(u.path, O_RDONLY | O_NOFOLLOW)
        guard src >= 0 else { throw WhisperANEError.missingFile(f.path) }
        defer { close(src) }
        guard fstat(src, &st) == 0, st.st_mode & S_IFMT == S_IFREG else { throw WhisperANEError.badModel("\(f.path) is not a regular file") }
        if fclonefileat(src, AT_FDCWD, dest.path, 0) == 0 {
            do { try check(dest, f) } catch { try? FileManager.default.removeItem(at: dest); throw error }
            return
        }
        // Not clonable (another volume): copy and hash the same bytes in one pass.
        let out = open(dest.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0o644)
        guard out >= 0 else { throw WhisperANEError.badModel("cannot write \(f.path): \(String(cString: strerror(errno)))") }
        var ok = false
        defer {
            close(out)
            if !ok { try? FileManager.default.removeItem(at: dest) }
        }
        let buf = UnsafeMutableRawBufferPointer.allocate(byteCount: 4 << 20, alignment: 16)
        defer { buf.deallocate() }
        var hasher = SHA256()
        var size: Int64 = 0
        while true {
            let n = read(src, buf.baseAddress, buf.count)
            if n < 0, errno == EINTR { continue }
            guard n >= 0 else { throw WhisperANEError.badModel("cannot read \(f.path): \(String(cString: strerror(errno)))") }
            if n == 0 { break }
            let chunk = UnsafeRawBufferPointer(rebasing: buf[0..<n])
            hasher.update(bufferPointer: chunk)
            var written = 0
            while written < n {
                let w = write(out, chunk.baseAddress! + written, n - written)
                if w < 0, errno == EINTR { continue }
                guard w > 0 else { throw WhisperANEError.badModel("cannot write \(f.path): \(String(cString: strerror(errno)))") }
                written += w
            }
            size += Int64(n)
        }
        guard size == f.size, hasher.finalize().map({ String(format: "%02x", $0) }).joined() == f.sha256.lowercased() else {
            throw WhisperANEError.checksumMismatch(path: f.path)
        }
        ok = true
    }

    /// Size and SHA-256 of every pinned file in `folder`.
    static func verify(_ pin: ModelPin, folder: URL) throws {
        try pin.checkPaths()
        for f in pin.files {
            let u = folder.appendingPathComponent(f.path)
            guard FileManager.default.fileExists(atPath: u.path) else { throw WhisperANEError.missingFile(f.path) }
            try check(u, f)
        }
    }

    static func check(_ url: URL, _ f: ModelPin.File) throws {
        let size = (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value
        guard size == f.size, try sha256(of: url) == f.sha256.lowercased() else {
            throw WhisperANEError.checksumMismatch(path: f.path)
        }
    }

    static func sha256(of url: URL) throws -> String {
        let h = try FileHandle(forReadingFrom: url)
        defer { try? h.close() }
        var hasher = SHA256()
        while let chunk = try h.read(upToCount: 4 << 20), !chunk.isEmpty { hasher.update(data: chunk) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func url(_ pin: ModelPin, _ f: ModelPin.File) throws -> URL {
        var allowed = CharacterSet.urlPathAllowed
        allowed.remove("/")
        func encoded(_ s: String) -> String {
            s.split(separator: "/").map { String($0).addingPercentEncoding(withAllowedCharacters: allowed) ?? String($0) }.joined(separator: "/")
        }
        guard let u = URL(string: "https://huggingface.co/\(encoded(pin.repo))/resolve/\(encoded(pin.revision))/\(encoded(f.path))") else {
            throw WhisperANEError.badPin("bad URL for \(f.path)")
        }
        return u
    }

    private static func download(_ pin: ModelPin, _ f: ModelPin.File, progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        let watcher = ProgressWatcher(progress)
        let (tmp, response): (URL, URLResponse)
        do {
            (tmp, response) = try await URLSession.shared.download(from: url(pin, f), delegate: watcher)
        } catch {
            throw WhisperANEError.download("\(f.path): \(error.localizedDescription)")
        }
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            try? FileManager.default.removeItem(at: tmp)
            throw WhisperANEError.download("\(f.path): HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0)")
        }
        // Moved out of the system's download location into a file this package owns.
        let kept = FileManager.default.temporaryDirectory.appendingPathComponent("whisper-ane-" + UUID().uuidString)
        try FileManager.default.moveItem(at: tmp, to: kept)
        return kept
    }
}

/// Reports a download's progress from its task.
private final class ProgressWatcher: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    private let report: @Sendable (Double) -> Void
    private var observation: NSKeyValueObservation?

    init(_ report: @escaping @Sendable (Double) -> Void) { self.report = report }

    func urlSession(_ session: URLSession, didCreateTask task: URLSessionTask) {
        let report = self.report
        observation = task.progress.observe(\.fractionCompleted) { p, _ in report(p.fractionCompleted) }
    }
}
