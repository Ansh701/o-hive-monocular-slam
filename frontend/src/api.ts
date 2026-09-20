export interface CalibrationInput {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
}

export interface RunAccepted {
  id: string;
  status: string;
  stage: string;
  source_filename?: string;
}

export interface PublicPoint {
  id: number;
  position: [number, number, number];
  observations: number;
  reprojection_error: number | null;
}

export interface PublicPose {
  frame_index: number;
  timestamp_seconds: number;
  keyframe: boolean;
  world_to_camera: number[][];
}

export interface FrameDiagnostic {
  frame_index: number;
  tracked_features: number;
  inliers: number;
  reprojection_error: number | null;
  stage: string;
}

export interface SlamPayload {
  status: "completed" | "partial";
  scale: "arbitrary";
  processing_seconds?: number;
  pose_count: number;
  point_count: number;
  rendered_point_count: number;
  points: PublicPoint[];
  poses: PublicPose[];
  diagnostics: FrameDiagnostic[];
}

export interface RunView extends RunAccepted {
  source_filename: string;
  width: number;
  height: number;
  frame_count: number;
  duration_seconds: number;
  processing_seconds: number | null;
  pose_count: number;
  point_count: number;
  keyframe_count: number;
  scale: "arbitrary";
  result: SlamPayload | null;
  error: { code: string; message: string } | null;
}

function parseError(request: XMLHttpRequest): Error {
  try {
    const body = JSON.parse(request.responseText) as { detail?: { message?: string } };
    return new Error(body.detail?.message ?? "The upload could not be completed.");
  } catch {
    return new Error("The upload could not be completed.");
  }
}

export function createRun(
  file: File,
  calibration: CalibrationInput | null,
  onProgress: (percentage: number) => void,
): Promise<RunAccepted> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    const form = new FormData();
    form.append("video", file, file.name);
    if (calibration) {
      for (const [name, value] of Object.entries(calibration)) form.append(name, String(value));
    }
    request.open("POST", "/api/slam-runs");
    request.responseType = "json";
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 100));
    });
    request.addEventListener("load", () => {
      if (request.status >= 200 && request.status < 300) {
        resolve(request.response as RunAccepted);
      } else {
        reject(parseError(request));
      }
    });
    request.addEventListener("error", () => reject(new Error("The server could not be reached.")));
    request.send(form);
  });
}

export async function getRun(id: string): Promise<RunView> {
  const response = await fetch(`/api/slam-runs/${encodeURIComponent(id)}`, {
    headers: { Accept: "application/json" },
  });
  if (!response.ok) throw new Error("Reconstruction status could not be loaded.");
  return (await response.json()) as RunView;
}
