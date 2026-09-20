import {
  Activity,
  Aperture,
  ArrowUpRight,
  Check,
  ChevronDown,
  CircleAlert,
  CloudUpload,
  Focus,
  Moon,
  RotateCcw,
  Sun,
  Trash2,
} from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";

import type { CalibrationInput, RunView } from "./api";
import { createRun, getRun } from "./api";
import type { SceneLayers } from "./components/Scene";

const LazyScene = lazy(async () => import("./components/Scene").then((module) => ({ default: module.Scene })));
const MAX_VIDEO_BYTES = 50 * 1024 * 1024;
const ACCEPTED_TYPES = new Set(["video/mp4", "video/quicktime", "video/webm"]);

type Phase = "empty" | "selected" | "uploading" | "processing" | "result" | "error";
type Theme = "light" | "dark";

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatType(file: File) {
  const suffix = file.name.split(".").pop()?.toUpperCase();
  return suffix || file.type.replace("video/", "").toUpperCase();
}

function validateFile(file: File): string | null {
  if (!ACCEPTED_TYPES.has(file.type)) return "Choose an MP4, MOV, or WebM video.";
  if (file.size > MAX_VIDEO_BYTES) return "This video is larger than the 50 MB limit.";
  return null;
}

function ThemeToggle({ theme, onToggle }: { theme: Theme; onToggle: () => void }) {
  const dark = theme === "dark";
  return (
    <button className="icon-button theme-toggle" type="button" onClick={onToggle} aria-label={`Switch to ${dark ? "light" : "dark"} theme`}>
      <Sun aria-hidden="true" className={dark ? "theme-icon hidden-icon" : "theme-icon"} strokeWidth={1.5} />
      <Moon aria-hidden="true" className={dark ? "theme-icon" : "theme-icon hidden-icon"} strokeWidth={1.5} />
    </button>
  );
}

function StepRail({ phase }: { phase: Phase }) {
  const current = phase === "empty" || phase === "selected" ? 0 : phase === "uploading" || phase === "processing" ? 1 : 2;
  return (
    <ol className="step-rail" aria-label="Reconstruction stages">
      {["Upload", "Reconstruct", "Explore"].map((label, index) => (
        <li key={label} className={index <= current ? "active-step" : ""} aria-current={index === current ? "step" : undefined}>
          <span>{index < current ? <Check size={13} strokeWidth={2} /> : `0${index + 1}`}</span>{label}
        </li>
      ))}
    </ol>
  );
}

interface AppProps { pollIntervalMs?: number }

export function App({ pollIntervalMs = 650 }: AppProps) {
  const [theme, setTheme] = useState<Theme>(() => localStorage.getItem("slam-theme") === "dark" ? "dark" : "light");
  const [phase, setPhase] = useState<Phase>("empty");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [run, setRun] = useState<RunView | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [advanced, setAdvanced] = useState(false);
  const [calibration, setCalibration] = useState({ fx: "", fy: "", cx: "", cy: "" });
  const [layers, setLayers] = useState<SceneLayers>({ points: true, trajectory: true, keyframes: true, grid: true });
  const [resetSignal, setResetSignal] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);

  const selectFile = (selected: File | undefined) => {
    if (!selected) return;
    const problem = validateFile(selected);
    if (problem) { setError(problem); setPhase("error"); return; }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(selected));
    setFile(selected);
    setError(null);
    setRun(null);
    setPhase("selected");
  };

  const reset = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null); setFile(null); setError(null); setRun(null); setAdvanced(false);
    setUploadProgress(0); setPhase("empty"); setResetSignal((value) => value + 1);
    if (inputRef.current) inputRef.current.value = "";
  };

  const parsedCalibration = useMemo<CalibrationInput | null>(() => {
    if (!advanced) return null;
    const values = Object.values(calibration).map(Number);
    if (values.some((value) => !Number.isFinite(value) || value <= 0)) return null;
    return { fx: values[0], fy: values[1], cx: values[2], cy: values[3] };
  }, [advanced, calibration]);

  const start = async () => {
    if (!file || (advanced && !parsedCalibration)) return;
    setError(null); setPhase("uploading"); setUploadProgress(0);
    try {
      const accepted = await createRun(file, parsedCalibration, setUploadProgress);
      setPhase("processing");
      for (;;) {
        const current = await getRun(accepted.id);
        setRun(current);
        if (["COMPLETED", "PARTIAL", "FAILED"].includes(current.status)) {
          if (current.status === "FAILED") { setError(current.error?.message ?? "Reconstruction failed."); setPhase("error"); }
          else setPhase("result");
          break;
        }
        await new Promise((resolve) => window.setTimeout(resolve, pollIntervalMs));
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Reconstruction failed.");
      setPhase("error");
    }
  };

  const processingLabel = run?.stage === "optimizing" ? "Reducing local drift" : run?.stage === "tracking" ? "Tracking camera motion" : "Preparing geometric reconstruction";
  const result = run?.result;

  return (
    <div className="app-shell">
      <a className="skip-link" href="#workspace">Skip to workspace</a>
      <header className="floating-nav">
        <a className="brand" href="#workspace" aria-label="O-HIVE Sparse SLAM home"><span><Aperture size={17} strokeWidth={1.5} /></span><strong>O—HIVE</strong><small>SLAM / 02</small></a>
        <StepRail phase={phase} />
        <div className="nav-actions"><span className="privacy-dot"><i /> Ephemeral video</span><ThemeToggle theme={theme} onToggle={() => { const next = theme === "light" ? "dark" : "light"; setTheme(next); localStorage.setItem("slam-theme", next); }} /></div>
      </header>

      <main id="workspace" className="workspace">
        <section className="intro">
          <p className="eyebrow">Classical geometry · Browser explorer</p>
          <h1>Reconstruct camera motion.<br /><em>Inspect the geometry.</em></h1>
          <p className="lede">Upload a short monocular RGB video. The system estimates a relative trajectory, triangulates a sparse map, and exposes every useful diagnostic—without pretending scale is metric.</p>
        </section>

        {(phase === "empty" || phase === "selected" || phase === "uploading" || phase === "error") && (
          <section className="upload-layout" aria-label="Video upload workspace">
            <div className="double-bezel upload-bezel">
              <div className={`upload-core ${file ? "has-file" : ""}`} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); selectFile(event.dataTransfer.files[0]); }}>
                <input ref={inputRef} id="video-input" className="visually-hidden" type="file" accept="video/mp4,video/quicktime,video/webm,.mp4,.mov,.webm" aria-label="Choose video" onChange={(event) => selectFile(event.target.files?.[0])} />
                {!file ? <>
                  <span className="upload-mark"><CloudUpload size={30} strokeWidth={1.25} /></span>
                  <div><h2>Drop a short monocular video</h2><p>Sideways camera motion and textured, static scenes produce the strongest geometry.</p></div>
                  <label className="primary-button" htmlFor="video-input">Choose video <span><ArrowUpRight size={15} strokeWidth={1.7} /></span></label>
                  <div className="constraints"><span>MP4 · MOV · WEBM</span><span>≤ 50 MB</span><span>≤ 30 seconds</span></div>
                </> : <>
                  <div className="file-preview">{previewUrl && <video src={previewUrl} muted playsInline preload="metadata" aria-label="Selected video preview" />}<span>LOCAL PREVIEW</span></div>
                  <div className="file-copy"><p className="eyebrow">Ready for reconstruction</p><h2>{file.name}</h2><p>{formatBytes(file.size)} · {formatType(file)}</p></div>
                  <button className="icon-button remove-button" type="button" onClick={reset} aria-label="Remove video"><Trash2 size={18} strokeWidth={1.5} /></button>
                </>}
              </div>
            </div>

            <aside className="control-panel">
              <div className="control-heading"><span><Focus size={17} strokeWidth={1.4} /></span><div><p className="eyebrow">Calibration</p><h2>Camera model</h2></div></div>
              <p className="control-copy">By default, focal length is approximated from frame dimensions and labeled in the result.</p>
              <button className="text-button" type="button" onClick={() => setAdvanced((value) => !value)} aria-expanded={advanced}>Advanced calibration <ChevronDown size={16} className={advanced ? "rotate" : ""} /></button>
              {advanced && <fieldset className="calibration-grid"><legend className="visually-hidden">Camera intrinsics</legend>{(["fx", "fy", "cx", "cy"] as const).map((name) => <label key={name}>{name === "fx" ? "Focal length fx" : name === "fy" ? "Focal length fy" : name === "cx" ? "Principal point cx" : "Principal point cy"}<input inputMode="decimal" value={calibration[name]} onChange={(event) => setCalibration((current) => ({ ...current, [name]: event.target.value }))} placeholder={name === "fx" || name === "fy" ? "e.g. 720" : name === "cx" ? "e.g. 320" : "e.g. 180"} /></label>)}</fieldset>}
              {error && <div className="error-box" role="alert"><CircleAlert size={18} strokeWidth={1.6} /><div><strong>Reconstruction needs attention</strong><p>{error}</p></div></div>}
              {phase === "uploading" && <div className="progress-block" aria-live="polite"><div><span>Uploading video · {uploadProgress}%</span><span>{uploadProgress}%</span></div><progress max="100" value={uploadProgress} /></div>}
              <button className="primary-button wide-button" type="button" disabled={!file || phase === "uploading" || (advanced && !parsedCalibration)} onClick={() => void start()}>Start reconstruction <span><ArrowUpRight size={15} strokeWidth={1.7} /></span></button>
              <p className="privacy-note">Video bytes are deleted after processing. Only bounded run metadata is persisted.</p>
            </aside>
          </section>
        )}

        {phase === "processing" && <section className="processing-shell" aria-live="polite"><div className="scanner"><i /><Aperture size={34} strokeWidth={1.1} /></div><div><p className="eyebrow">Geometric pipeline active</p><h2>{processingLabel}</h2><p>Feature tracks are being validated against a persistent sparse map.</p></div><ol><li className="done"><Check size={15} /> Video validated</li><li className="active"><Activity size={15} /> {processingLabel}</li><li>Local bundle adjustment</li></ol></section>}

        {phase === "result" && run && result && <section className="result-layout">
          <div className="result-header"><div><p className="eyebrow">{run.status === "PARTIAL" ? "Partial reconstruction" : "Reconstruction complete"}</p><h2>{run.source_filename}</h2><p>{run.status === "PARTIAL" ? "A valid partial map was retained after tracking became unreliable." : "Trajectory and sparse landmarks are ready to inspect."}</p></div><button className="secondary-button" type="button" onClick={reset}><RotateCcw size={15} strokeWidth={1.5} /> New reconstruction</button></div>
          <div className="explorer-grid"><div className="double-bezel scene-bezel"><div className="scene-core"><Suspense fallback={<div className="scene-skeleton">Preparing 3D explorer…</div>}><LazyScene result={result} layers={layers} resetSignal={resetSignal} /></Suspense><div className="scene-caption"><span>Orbit · pan · zoom</span><button type="button" onClick={() => setResetSignal((value) => value + 1)}><RotateCcw size={13} /> Reset view</button></div></div></div>
          <aside className="inspector"><div className="metric-grid"><div><span>Poses</span><strong>{run.pose_count}</strong></div><div><span>Landmarks</span><strong>{run.point_count.toLocaleString()}</strong></div><div><span>Keyframes</span><strong>{run.keyframe_count}</strong></div><div><span>Compute</span><strong>{run.processing_seconds?.toFixed(2)}s</strong></div></div>
          <div className="layer-panel"><h3>Scene layers</h3>{([['points','Point cloud'],['trajectory','Camera trajectory'],['keyframes','Keyframe markers'],['grid','Reference grid']] as [keyof SceneLayers,string][]).map(([key,label]) => <label key={key}><input type="checkbox" checked={layers[key]} onChange={() => setLayers((current) => ({ ...current, [key]: !current[key] }))} /> <span>{label}</span></label>)}</div>
          <div className="disclosure"><strong>Coordinate frame</strong><p>Coordinates use arbitrary relative units. Monocular reconstruction does not recover metric scale.</p>{result.rendered_point_count < result.point_count && <p>Showing {result.rendered_point_count.toLocaleString()} of {result.point_count.toLocaleString()} points for responsive rendering.</p>}</div></aside></div>
          <div className="text-summary" aria-label="Accessible reconstruction summary"><h3>Reconstruction summary</h3><p>{run.pose_count} camera poses, {run.keyframe_count} keyframes, and {run.point_count.toLocaleString()} sparse landmarks were recovered in {run.processing_seconds?.toFixed(2)} seconds. Scale is arbitrary.</p></div>
        </section>}
      </main>
      <footer><span>O-HIVE / Assignment 02</span><span>OpenCV · SciPy · Three.js</span></footer>
    </div>
  );
}
