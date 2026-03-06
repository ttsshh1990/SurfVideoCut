import SwiftUI
import WebKit

struct WebView: NSViewRepresentable {
    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.preferences.javaScriptEnabled = true
        config.userContentController.add(context.coordinator, name: "native")
        // Allow file:// access beyond bundle for loading local videos
        let webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = context.coordinator
        webView.uiDelegate = context.coordinator

        if let url = Bundle.main.url(forResource: "ui_mock", withExtension: "html") {
            // Grant wide read access so file:// videos outside the bundle can load
            webView.loadFileURL(url, allowingReadAccessTo: URL(fileURLWithPath: "/"))
        } else {
            let html = "<h3 style='font-family:-apple-system'>ui_mock.html not found in bundle</h3>"
            webView.loadHTMLString(html, baseURL: nil)
        }
        return webView
    }

    func updateNSView(_ nsView: WKWebView, context: Context) {
        // No dynamic update needed
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    class Coordinator: NSObject, WKNavigationDelegate, WKUIDelegate, WKScriptMessageHandler {
        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            checkStartupEnvironment(webView: webView)
        }

        func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
            guard message.name == "native" else { return }
            NSLog("[bridge] message received: %@", "\(message.body)")
            if let body = message.body as? [String: Any], let type = body["type"] as? String {
                if type == "pickOutdir", let webView = message.webView {
                    pickOutdir(for: webView)
                } else if type == "export", let webView = message.webView {
                    handleExport(body: body, webView: webView)
                } else if type == "pickSdRoot", let webView = message.webView {
                    pickSdRoot(for: webView)
                } else if type == "combine", let webView = message.webView {
                    handleCombine(body: body, webView: webView)
                } else if type == "clean", let webView = message.webView {
                    handleClean(body: body, webView: webView)
                } else if type == "detect", let webView = message.webView {
                    handleDetect(body: body, webView: webView)
                } else if type == "autoDetectSdRoot", let webView = message.webView {
                    autoDetectSdRoot(body: body, webView: webView)
                } else if type == "debug" {
                    if let msg = body["message"] as? String {
                        NSLog("[js] %@", msg)
                    }
                }
            }
        }

        private func checkStartupEnvironment(webView: WKWebView) {
            DispatchQueue.global(qos: .utility).async {
                let env = self.processEnvironment()
                let pythonPath = self.resolvePythonExecutable(environment: env)
                let pythonOK = pythonPath != nil
                let pythonMsg: String
                if let pythonPath {
                    let pythonResult = self.runCommand(executablePath: pythonPath, arguments: ["--version"], environment: env)
                    let version = pythonResult.output.isEmpty ? "python3 available" : pythonResult.output
                    pythonMsg = "\(version) (\(pythonPath))"
                } else {
                    pythonMsg = "python3 not found"
                }
                let ffmpegPath = self.bundledFFmpegPath()
                let ffmpegOK = FileManager.default.isExecutableFile(atPath: ffmpegPath)
                let ffmpegMsg = ffmpegOK ? ffmpegPath : "Bundled ffmpeg missing"
                DispatchQueue.main.async {
                    self.notifyStartupEnvironment(webView: webView, pythonOK: pythonOK, pythonMessage: pythonMsg, ffmpegOK: ffmpegOK, ffmpegMessage: ffmpegMsg)
                }
            }
        }

        private func processEnvironment() -> [String: String] {
            var env = ProcessInfo.processInfo.environment
            let bundledBin = bundledBinPath()
            let existing = env["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"
            let extra = ["/opt/homebrew/bin", "/usr/local/bin"]
            env["PATH"] = ([bundledBin] + extra + [existing]).joined(separator: ":")
            return env
        }

        private func resolvePythonExecutable(environment: [String: String]) -> String? {
            var candidates: [String] = []
            let pathEntries = (environment["PATH"] ?? "").split(separator: ":").map(String.init)
            candidates.append(contentsOf: pathEntries.map { "\($0)/python3" })
            candidates.append(contentsOf: [
                "/opt/homebrew/bin/python3",
                "/usr/local/bin/python3",
                "/usr/bin/python3"
            ])

            var seen = Set<String>()
            for candidate in candidates {
                guard !candidate.isEmpty, !seen.contains(candidate) else { continue }
                seen.insert(candidate)
                guard FileManager.default.isExecutableFile(atPath: candidate) else { continue }
                let result = runCommand(executablePath: candidate, arguments: ["--version"], environment: environment)
                if result.status == 0 {
                    return candidate
                }
            }
            return nil
        }

        private func bundledBinPath() -> String {
            Bundle.main.resourceURL?.appendingPathComponent("surf_video_cut/bin").path ?? "/usr/bin"
        }

        private func bundledFFmpegPath() -> String {
            Bundle.main.resourceURL?.appendingPathComponent("surf_video_cut/bin/ffmpeg").path ?? "ffmpeg"
        }

        private func runCommand(executablePath: String, arguments: [String], environment: [String: String]) -> (status: Int32, output: String) {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: executablePath)
            proc.arguments = arguments
            proc.environment = environment
            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError = pipe
            do {
                try proc.run()
            } catch {
                return (1, error.localizedDescription)
            }
            proc.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return (proc.terminationStatus, output)
        }

        private func pickOutdir(for webView: WKWebView) {
            let panel = NSOpenPanel()
            panel.allowsMultipleSelection = false
            panel.canChooseDirectories = true
            panel.canChooseFiles = false
            panel.canCreateDirectories = true
            panel.prompt = "Choose"
            panel.begin { result in
                if result == .OK, let url = panel.url {
                    let path = url.path.replacingOccurrences(of: "\"", with: "\\\"")
                    let js = "document.getElementById('outdirPath').value = \"\(path)\";"
                    webView.evaluateJavaScript(js, completionHandler: nil)
                }
            }
        }

        private func pickSdRoot(for webView: WKWebView) {
            let panel = NSOpenPanel()
            panel.allowsMultipleSelection = false
            panel.canChooseDirectories = true
            panel.canChooseFiles = false
            panel.prompt = "Choose SD Root"
            panel.begin { result in
                if result == .OK, let url = panel.url {
                    let path = url.path.replacingOccurrences(of: "\"", with: "\\\"")
                    let js = "window.nativeSetSdRoot && window.nativeSetSdRoot(\"\(path)\");"
                    webView.evaluateJavaScript(js, completionHandler: nil)
                }
            }
        }

        func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters, initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping ([URL]?) -> Void) {
            let panel = NSOpenPanel()
            panel.allowsMultipleSelection = parameters.allowsMultipleSelection
            panel.canChooseFiles = true
            panel.canChooseDirectories = false
            panel.allowedFileTypes = ["mp4", "mov", "m4v", "avi", "mkv", "mpg"]
            panel.begin { result in
                if result == .OK {
                    let urls = panel.urls
                    if let url = urls.first {
                        let path = url.path.replacingOccurrences(of: "\"", with: "\\\"")
                        let js = "window.nativeSetVideoPath && window.nativeSetVideoPath(\"\(path)\");"
                        webView.evaluateJavaScript(js, completionHandler: nil)
                    }
                    completionHandler(urls)
                } else {
                    completionHandler(nil)
                }
            }
        }

        private func handleExport(body: [String: Any], webView: WKWebView) {
            guard
                let videoPath = body["videoPath"] as? String,
                let outdir = body["outdir"] as? String,
                let segments = body["segments"] as? [[String: Any]]
            else {
                notifyExport(webView: webView, ok: false, message: "Missing video/outdir/segments")
                return
            }
            let keepAudio = true
            let jobs = 8
            let fileManager = FileManager.default
            // derive clips folder under base outdir
            let clipsDir = URL(fileURLWithPath: outdir).appendingPathComponent("clips").path
            do {
                try fileManager.createDirectory(atPath: clipsDir, withIntermediateDirectories: true, attributes: nil)
            } catch {
                notifyExport(webView: webView, ok: false, message: "Failed to create outdir: \(error.localizedDescription)")
                return
            }
            let stem = URL(fileURLWithPath: videoPath).deletingPathExtension().lastPathComponent
            let tempSegments = URL(fileURLWithPath: clipsDir).appendingPathComponent("\(stem)_segments.__cut__.txt")
            var text = ""
            for seg in segments {
                guard let s = seg["start"] as? Double, let e = seg["end"] as? Double else { continue }
                text.append("\(formatTimecode(s))-\(formatTimecode(e))\n")
            }
            do {
                try text.write(to: tempSegments, atomically: true, encoding: .utf8)
            } catch {
                notifyExport(webView: webView, ok: false, message: "Failed to write temp segments: \(error.localizedDescription)")
                return
            }

            let scriptPath = Bundle.main.resourceURL?.appendingPathComponent("surf_video_cut/cut_video.py").path ?? "cut_video.py"
            let env = processEnvironment()
            guard let pythonPath = resolvePythonExecutable(environment: env) else {
                notifyExport(webView: webView, ok: false, message: "python3 not found")
                return
            }
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: pythonPath)
            proc.arguments = [scriptPath, "--input", videoPath, "--outdir", clipsDir, "--segments-file", tempSegments.path, "--jobs", "\(jobs)", "--keep-audio"]
            proc.environment = env
            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError = pipe
            do {
                try proc.run()
            } catch {
                notifyExport(webView: webView, ok: false, message: "Failed to start export: \(error.localizedDescription)")
                return
            }
            proc.waitUntilExit()
            let status = proc.terminationStatus
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let log = String(data: data, encoding: .utf8) ?? ""
            if status == 0 {
                notifyExport(webView: webView, ok: true, message: "Exported to \(clipsDir)\n\(log)")
            } else {
                notifyExport(webView: webView, ok: false, message: "Export failed (code \(status))\n\(log)")
            }
            try? fileManager.removeItem(at: tempSegments)
        }

        private func notifyExport(webView: WKWebView, ok: Bool, message: String) {
            let escaped = message.replacingOccurrences(of: "\"", with: "\\\"").replacingOccurrences(of: "\n", with: "\\n")
            let js = "window.nativeExportDone && window.nativeExportDone(\(ok ? "true" : "false"), \"\(escaped)\");"
            webView.evaluateJavaScript(js, completionHandler: nil)
        }

        private func notifyStartupEnvironment(webView: WKWebView, pythonOK: Bool, pythonMessage: String, ffmpegOK: Bool, ffmpegMessage: String) {
            let py = escapeForJS(pythonMessage)
            let ff = escapeForJS(ffmpegMessage)
            let js = "window.nativeStartupEnvironment && window.nativeStartupEnvironment(\(pythonOK ? "true" : "false"), \"\(py)\", \(ffmpegOK ? "true" : "false"), \"\(ff)\");"
            webView.evaluateJavaScript(js, completionHandler: nil)
        }

        private func handleCombine(body: [String: Any], webView: WKWebView) {
            guard
                let sdRoot = body["sdRoot"] as? String,
                let outdir = body["outdir"] as? String
            else {
                notifyCombine(webView: webView, ok: false, message: "Missing sdRoot/outdir")
                return
            }
            let fileManager = FileManager.default
            DispatchQueue.global(qos: .userInitiated).async {
                // Compute input parts and total size for progress fallback
                let inputDir = URL(fileURLWithPath: sdRoot)
                let parts = (try? fileManager.contentsOfDirectory(at: inputDir, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]))?
                    .filter { $0.pathExtension.lowercased() == "mp4" } ?? []
                let totalBytes: Int64 = parts.reduce(0) { acc, url in
                    let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                    return acc + Int64(size)
                }
                let outPath = URL(fileURLWithPath: outdir).appendingPathComponent(inputDir.lastPathComponent + ".mp4")

                do {
                    try fileManager.createDirectory(atPath: outdir, withIntermediateDirectories: true, attributes: nil)
                } catch {
                    DispatchQueue.main.async {
                        self.notifyCombine(webView: webView, ok: false, message: "Failed to create outdir: \(error.localizedDescription)")
                    }
                    return
                }
                let scriptPath = Bundle.main.resourceURL?.appendingPathComponent("surf_video_cut/combine_simple.py").path ?? "combine_simple.py"
                var env = self.processEnvironment()
                guard let pythonPath = self.resolvePythonExecutable(environment: env) else {
                    DispatchQueue.main.async {
                        self.notifyCombine(webView: webView, ok: false, message: "python3 not found")
                    }
                    return
                }
                let proc = Process()
                proc.executableURL = URL(fileURLWithPath: pythonPath)
                proc.arguments = [scriptPath, "--sd-root", sdRoot, "--outdir", outdir]
                env["PYTHONUNBUFFERED"] = "1"
                proc.environment = env
                let pipe = Pipe()
                proc.standardOutput = pipe
                proc.standardError = pipe
                var combineLogData = Data()
                var combineLogBuffer = ""
                do {
                    try proc.run()
                } catch {
                    DispatchQueue.main.async {
                        self.notifyCombine(webView: webView, ok: false, message: "Failed to start combine: \(error.localizedDescription)")
                    }
                    return
                }
                // Fallback progress by output file size
                let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
                timer.schedule(deadline: .now() + 0.5, repeating: 0.5)
                timer.setEventHandler {
                    guard totalBytes > 0 else { return }
                    let size = (try? fileManager.attributesOfItem(atPath: outPath.path)[.size] as? NSNumber)?.int64Value ?? 0
                    let pct = min(0.99, max(0.0, Double(size) / Double(totalBytes)))
                    self.dispatchCombineProgress(pct, webView: webView)
                }
                timer.resume()

                pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
                    guard let self = self else { return }
                    let data = handle.availableData
                    if data.isEmpty {
                        pipe.fileHandleForReading.readabilityHandler = nil
                        return
                    }
                    combineLogData.append(data)
                    if var chunk = String(data: data, encoding: .utf8) {
                        chunk = chunk.replacingOccurrences(of: "\r", with: "\n")
                        combineLogBuffer.append(chunk)
                        let parts = combineLogBuffer.components(separatedBy: "\n")
                        combineLogBuffer = parts.last ?? ""
                        for line in parts.dropLast() {
                            self.consumeCombineLogLine(line, webView: webView)
                        }
                    }
                }
                proc.waitUntilExit()
                pipe.fileHandleForReading.readabilityHandler = nil
                timer.cancel()
                let status = proc.terminationStatus
                if !combineLogBuffer.isEmpty {
                    self.consumeCombineLogLine(combineLogBuffer, webView: webView)
                }
                let log = String(data: combineLogData, encoding: .utf8) ?? ""
                DispatchQueue.main.async {
                    if status == 0 {
                        self.dispatchCombineProgress(1.0, webView: webView)
                        let path = outPath.path.replacingOccurrences(of: "\"", with: "\\\"")
                        let js = "window.nativeCombineDone && window.nativeCombineDone(true, \"Combined\", \"\(path)\");"
                        webView.evaluateJavaScript(js, completionHandler: nil)
                    } else {
                        self.notifyCombine(webView: webView, ok: false, message: "Combine failed (code \(status))\n\(log)")
                    }
                }
            }
        }

        private func handleDetect(body: [String: Any], webView: WKWebView) {
            guard
                let videoPath = body["videoPath"] as? String,
                let outdir = body["outdir"] as? String
            else {
                notifyDetect(webView: webView, ok: false, message: "Missing video/outdir")
                return
            }
            let fm = FileManager.default
            do {
                try fm.createDirectory(atPath: outdir, withIntermediateDirectories: true, attributes: nil)
            } catch {
                notifyDetect(webView: webView, ok: false, message: "Failed to create outdir: \(error.localizedDescription)")
                return
            }
            DispatchQueue.main.async {
                let js = "window.nativeDetectProgress && window.nativeDetectProgress(0);"
                webView.evaluateJavaScript(js, completionHandler: nil)
            }
            let resourcesURL = Bundle.main.resourceURL?.appendingPathComponent("surf_video_cut")
            let modelPath = resourcesURL?.appendingPathComponent("models/yolov8n.pt").path
            let activityPath = resourcesURL?.appendingPathComponent("models/activity_classifier.pth").path
            DispatchQueue.global(qos: .userInitiated).async {
                let scriptPath = Bundle.main.resourceURL?.appendingPathComponent("surf_video_cut/process_improved.py").path ?? "process_improved.py"
                var env = self.processEnvironment()
                guard let pythonPath = self.resolvePythonExecutable(environment: env) else {
                    DispatchQueue.main.async {
                        self.notifyDetect(webView: webView, ok: false, message: "python3 not found")
                    }
                    return
                }
                let moduleCheck = self.runCommand(
                    executablePath: pythonPath,
                    arguments: ["-c", "import torch, torchvision, cv2, ultralytics, tqdm"],
                    environment: env
                )
                if moduleCheck.status != 0 {
                    let detail = moduleCheck.output.isEmpty ? "Required Python packages are missing for detection." : moduleCheck.output
                    DispatchQueue.main.async {
                        self.notifyDetect(webView: webView, ok: false, message: "Detection dependencies not available in \(pythonPath)\\n\(detail)")
                    }
                    return
                }
                let proc = Process()
                proc.executableURL = URL(fileURLWithPath: pythonPath)
                var args = [scriptPath, "--input", videoPath, "--outdir", outdir]
                if let modelPath = modelPath {
                    args += ["--model", modelPath]
                }
                if let activityPath = activityPath {
                    args += ["--activity-model", activityPath]
                }
                proc.arguments = args
                env["PYTHONUNBUFFERED"] = "1"
                proc.environment = env
                let pipe = Pipe()
                proc.standardOutput = pipe
                proc.standardError = pipe
                var detectLogData = Data()
                var detectLogBuffer = ""
                do {
                    try proc.run()
                } catch {
                    DispatchQueue.main.async {
                        self.notifyDetect(webView: webView, ok: false, message: "Failed to start detection: \(error.localizedDescription)")
                    }
                    return
                }
                pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
                    guard let self = self else { return }
                    let data = handle.availableData
                    if data.isEmpty {
                        pipe.fileHandleForReading.readabilityHandler = nil
                        return
                    }
                    detectLogData.append(data)
                    if var chunk = String(data: data, encoding: .utf8) {
                        chunk = chunk.replacingOccurrences(of: "\r", with: "\n")
                        detectLogBuffer.append(chunk)
                        let parts = detectLogBuffer.components(separatedBy: "\n")
                        detectLogBuffer = parts.last ?? ""
                        for line in parts.dropLast() {
                            self.consumeDetectLogLine(line, webView: webView)
                        }
                    }
                }
                proc.waitUntilExit()
                pipe.fileHandleForReading.readabilityHandler = nil
                if !detectLogBuffer.isEmpty {
                    self.consumeDetectLogLine(detectLogBuffer, webView: webView)
                }
                let log = String(data: detectLogData, encoding: .utf8) ?? ""
                let status = proc.terminationStatus
                DispatchQueue.main.async {
                    let stem = URL(fileURLWithPath: videoPath).deletingPathExtension().lastPathComponent
                    let segmentsPath = URL(fileURLWithPath: outdir).appendingPathComponent("\(stem)_segments.txt").path
                    if status == 0 {
                        self.dispatchDetectProgress(1.0, webView: webView)
                        let segmentsJSON = self.segmentsJSON(at: segmentsPath) ?? "[]"
                        self.notifyDetect(webView: webView, ok: true, message: "Detection completed", segmentsJSON: segmentsJSON, segmentsPath: segmentsPath)
                    } else {
                        self.notifyDetect(webView: webView, ok: false, message: "Detection failed (code \(status))\\n\(log)")
                    }
                }
            }
        }

        private func notifyCombine(webView: WKWebView, ok: Bool, message: String) {
            let escaped = message.replacingOccurrences(of: "\"", with: "\\\"").replacingOccurrences(of: "\n", with: "\\n")
            let js = "window.nativeCombineDone && window.nativeCombineDone(\(ok ? "true" : "false"), \"\(escaped)\", null);"
            webView.evaluateJavaScript(js, completionHandler: nil)
        }

        private func autoDetectSdRoot(body: [String: Any], webView: WKWebView) {
            let fileManager = FileManager.default
            let currentHint = (body["current"] as? String) ?? ""
            NSLog("[autoDetect] start current=%@", currentHint)
            var bases: [URL] = []
            if let current = body["current"] as? String, !current.isEmpty {
                let url = URL(fileURLWithPath: current)
                bases.append(url)
                bases.append(url.deletingLastPathComponent()) // parent
            }
            if let vols = fileManager.mountedVolumeURLs(includingResourceValuesForKeys: nil, options: [.skipHiddenVolumes]) {
                bases.append(contentsOf: vols)
            }
            // always include /Volumes as a base
            bases.append(URL(fileURLWithPath: "/Volumes"))
            let uniq = Array(Set(bases))

            var sessions: [URL] = []
            for base in uniq {
                let dcim = base.appendingPathComponent("DCIM/SOLOSHOT3")
                guard let users = try? fileManager.contentsOfDirectory(at: dcim, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]) else { continue }
                for user in users where user.hasDirectoryPath {
                    if let vids = try? fileManager.contentsOfDirectory(at: user, includingPropertiesForKeys: [.contentModificationDateKey], options: [.skipsHiddenFiles]) {
                        sessions.append(contentsOf: vids.filter { $0.hasDirectoryPath })
                    }
                }
            }
            NSLog("[autoDetect] sessions found=%d", sessions.count)

            // Fallback: if current path looks like a session with mp4 parts, use it
            if sessions.isEmpty, !currentHint.isEmpty {
                let curURL = URL(fileURLWithPath: currentHint)
                if let files = try? fileManager.contentsOfDirectory(at: curURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) {
                    let mp4s = files.filter { $0.pathExtension.lowercased() == "mp4" }
                    if !mp4s.isEmpty {
                        let path = curURL.path.replacingOccurrences(of: "\"", with: "\\\"")
                        NSLog("[autoDetect] using current hint with mp4s %@", path)
                        let js = "window.nativeAutoSdRoot && window.nativeAutoSdRoot(\"\(path)\");"
                        webView.evaluateJavaScript(js, completionHandler: nil)
                        return
                    }
                }
            }

            let newest = sessions.sorted { a, b in
                let ad = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let bd = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return ad > bd
            }.first

            if let path = newest?.path.replacingOccurrences(of: "\"", with: "\\\"") {
                let js = "window.nativeAutoSdRoot && window.nativeAutoSdRoot(\"\(path)\");"
                webView.evaluateJavaScript(js, completionHandler: nil)
            } else {
                // fallback: return /Volumes
                let fallback = "/Volumes"
                let js = "window.nativeAutoSdRoot && window.nativeAutoSdRoot(\"\(fallback)\");"
                webView.evaluateJavaScript(js, completionHandler: nil)
            }
        }

        private func handleClean(body: [String: Any], webView: WKWebView) {
            guard let sdRoot = body["sdRoot"] as? String, !sdRoot.isEmpty else {
                return
            }
            DispatchQueue.main.async {
                let alert = NSAlert()
                alert.messageText = "Delete session folder?"
                alert.informativeText = "This will permanently remove:\n\(sdRoot)"
                alert.alertStyle = .warning
                alert.addButton(withTitle: "Delete")
                alert.addButton(withTitle: "Cancel")
                let response = alert.runModal()
                if response == .alertFirstButtonReturn {
                    self.performClean(sdRoot: sdRoot, webView: webView)
                } else {
                    let js = "window.nativeCleanDone && window.nativeCleanDone(false, \"Canceled\");"
                    webView.evaluateJavaScript(js, completionHandler: nil)
                }
            }
        }

        private func performClean(sdRoot: String, webView: WKWebView) {
            DispatchQueue.global(qos: .userInitiated).async {
                let fm = FileManager.default
                do {
                    NSLog("[clean] removing %@", sdRoot)
                    try fm.removeItem(atPath: sdRoot)
                    DispatchQueue.main.async {
                        let js = "window.nativeCleanDone && window.nativeCleanDone(true, \"Cleaned\");"
                        webView.evaluateJavaScript(js, completionHandler: nil)
                    }
                } catch {
                    DispatchQueue.main.async {
                        let msg = error.localizedDescription.replacingOccurrences(of: "\"", with: "\\\"")
                        NSLog("[clean] failed: %@", msg)
                        let js = "window.nativeCleanDone && window.nativeCleanDone(false, \"\(msg)\");"
                        webView.evaluateJavaScript(js, completionHandler: nil)
                    }
                }
            }
        }

        private func notifyDetect(webView: WKWebView, ok: Bool, message: String, segmentsJSON: String? = nil, segmentsPath: String? = nil) {
            let msgEsc = escapeForJS(message)
            let segArg: String
            if let segmentsJSON = segmentsJSON {
                segArg = "\"\(escapeForJS(segmentsJSON))\""
            } else {
                segArg = "null"
            }
            let pathArg: String
            if let segmentsPath = segmentsPath {
                pathArg = "\"\(escapeForJS(segmentsPath))\""
            } else {
                pathArg = "null"
            }
            let js = "window.nativeDetectDone && window.nativeDetectDone(\(ok ? "true" : "false"), \"\(msgEsc)\", \(segArg), \(pathArg));"
            webView.evaluateJavaScript(js, completionHandler: nil)
        }

        private func segmentsJSON(at path: String) -> String? {
            let fm = FileManager.default
            guard fm.fileExists(atPath: path) else { return nil }
            guard let raw = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
            var items: [[String: Double]] = []
            raw.components(separatedBy: .newlines).forEach { line in
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.isEmpty || trimmed.hasPrefix("#") { return }
                guard let firstToken = trimmed.split(whereSeparator: { $0.isWhitespace }).first else { return }
                let rangeParts = firstToken.split(separator: "-", maxSplits: 1)
                if rangeParts.count != 2 { return }
                let startStr = String(rangeParts[0])
                let endStr = String(rangeParts[1])
                guard let start = timecodeToSeconds(startStr), let end = timecodeToSeconds(endStr) else { return }
                let s = min(start, end)
                let e = max(start, end)
                if e - s <= 0 {
                    return
                }
                items.append(["start": s, "end": e])
            }
            guard let data = try? JSONSerialization.data(withJSONObject: items, options: []) else { return nil }
            return String(data: data, encoding: .utf8)
        }

        private func timecodeToSeconds(_ text: String) -> Double? {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { return nil }
            let parts = trimmed.split(separator: ":")
            if parts.count == 3 {
                let h = Double(parts[0]) ?? 0
                let m = Double(parts[1]) ?? 0
                let s = Double(parts[2]) ?? 0
                return h * 3600 + m * 60 + s
            } else if parts.count == 2 {
                let m = Double(parts[0]) ?? 0
                let s = Double(parts[1]) ?? 0
                return m * 60 + s
            } else if parts.count == 1 {
                return Double(parts[0])
            }
            return nil
        }

        private func escapeForJS(_ text: String) -> String {
            return text
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
                .replacingOccurrences(of: "\n", with: "\\n")
                .replacingOccurrences(of: "\r", with: "")
        }

        private func consumeCombineLogLine(_ rawLine: String, webView: WKWebView) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty else { return }
            if let pct = parseCombineProgress(from: line) {
                dispatchCombineProgress(pct, webView: webView)
                return
            }
            NSLog("[combine-log] %@", line)
        }

        private func consumeDetectLogLine(_ rawLine: String, webView: WKWebView) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty else { return }
            if let pct = parsePercentFromProgressLine(line) {
                dispatchDetectProgress(pct, webView: webView)
                return
            }
            NSLog("[detect-log] %@", line)
        }

        private func parseCombineProgress(from line: String) -> Double? {
            guard let range = line.range(of: "[PROGRESS]") else { return nil }
            let valueString = line[range.upperBound...].trimmingCharacters(in: .whitespaces)
            guard let value = Double(valueString) else { return nil }
            return max(0.0, min(1.0, value))
        }

        private func parsePercentFromProgressLine(_ line: String) -> Double? {
            guard line.contains("%|"), let percentIndex = line.firstIndex(of: "%") else { return nil }
            var idx = percentIndex
            var digits = ""
            while idx > line.startIndex {
                idx = line.index(before: idx)
                let ch = line[idx]
                if ch.isNumber || ch == "." {
                    digits.insert(ch, at: digits.startIndex)
                    if idx == line.startIndex { break }
                } else if digits.isEmpty {
                    continue
                } else {
                    break
                }
            }
            if digits.isEmpty {
                return nil
            }
            guard let value = Double(digits) else { return nil }
            return max(0.0, min(1.0, value / 100.0))
        }

        private func dispatchCombineProgress(_ value: Double, webView: WKWebView) {
            let clamped = max(0.0, min(1.0, value))
            DispatchQueue.main.async {
                let js = "window.nativeCombineProgress && window.nativeCombineProgress(\(clamped));"
                webView.evaluateJavaScript(js, completionHandler: nil)
            }
        }

        private func dispatchDetectProgress(_ value: Double, webView: WKWebView) {
            let clamped = max(0.0, min(1.0, value))
            DispatchQueue.main.async {
                let js = "window.nativeDetectProgress && window.nativeDetectProgress(\(clamped));"
                webView.evaluateJavaScript(js, completionHandler: nil)
            }
        }

        private func formatTimecode(_ sec: Double) -> String {
            let total = max(0.0, sec)
            let h = Int(total / 3600)
            let m = Int((total.truncatingRemainder(dividingBy: 3600)) / 60)
            let s = total.truncatingRemainder(dividingBy: 60)
            if h > 0 {
                return String(format: "%d:%02d:%05.2f", h, m, s)
            }
            return String(format: "%d:%05.2f", m, s)
        }
    }
}
