import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import * as api from "./api";

vi.mock("./api");
vi.mock("./components/Scene", () => ({
  Scene: () => <div data-testid="three-scene">3D map</div>,
}));

const accepted = { id: "0194f4b2-55a2-7000-8000-123456789abc", status: "UPLOADED", stage: "queued" };
const completed: api.RunView = {
  ...accepted,
  status: "COMPLETED",
  source_filename: "walk.mp4",
  width: 640,
  height: 360,
  frame_count: 80,
  duration_seconds: 10,
  processing_seconds: 4.2,
  pose_count: 14,
  point_count: 1800,
  keyframe_count: 5,
  scale: "arbitrary",
  result: {
    status: "completed",
    scale: "arbitrary",
    point_count: 1800,
    rendered_point_count: 1200,
    pose_count: 14,
    points: [],
    poses: [],
    diagnostics: [],
  },
  error: null,
};

function chooseVideo(name = "walk.mp4") {
  const file = new File([new Uint8Array(2048)], name, { type: "video/mp4" });
  fireEvent.change(screen.getByLabelText(/choose video/i), { target: { files: [file] } });
  return file;
}

beforeEach(() => {
  vi.resetAllMocks();
  localStorage.clear();
  document.documentElement.dataset.theme = "";
});

describe("SLAM workspace", () => {
  it("opens in a purpose-built empty upload state", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: /reconstruct camera motion/i })).toBeInTheDocument();
    expect(screen.getByText(/drop a short monocular video/i)).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("validates files, shows metadata, and can clear a selection", async () => {
    render(<App />);
    chooseVideo();
    expect(await screen.findByText("walk.mp4")).toBeInTheDocument();
    expect(screen.getByText("2 KB · MP4")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /remove video/i }));
    expect(screen.getByText(/drop a short monocular video/i)).toBeInTheDocument();

    const bad = new File(["unsafe"], "page.html", { type: "text/html" });
    fireEvent.change(screen.getByLabelText(/choose video/i), { target: { files: [bad] } });
    expect(await screen.findByRole("alert")).toHaveTextContent(/mp4, mov, or webm/i);
  });

  it("reveals labeled advanced intrinsics", async () => {
    render(<App />);
    chooseVideo();
    await userEvent.click(screen.getByRole("button", { name: /advanced calibration/i }));
    expect(screen.getByLabelText("Focal length fx")).toBeInTheDocument();
    expect(screen.getByLabelText("Principal point cy")).toBeInTheDocument();
  });

  it("reports real upload and processing stages", async () => {
    let finishUpload!: (value: typeof accepted) => void;
    let finishProcessing!: (value: api.RunView) => void;
    vi.mocked(api.createRun).mockImplementation((_file, _intrinsics, onProgress) => {
      onProgress(62);
      return new Promise((resolve) => {
        finishUpload = resolve;
      });
    });
    vi.mocked(api.getRun)
      .mockResolvedValueOnce({ ...completed, status: "PROCESSING", stage: "tracking", result: null })
      .mockImplementationOnce(() => new Promise((resolve) => {
        finishProcessing = resolve;
      }));
    render(<App pollIntervalMs={1} />);
    chooseVideo();
    await userEvent.click(screen.getByRole("button", { name: /start reconstruction/i }));
    expect(await screen.findByText(/uploading video · 62%/i)).toBeInTheDocument();
    act(() => finishUpload(accepted));
    expect(await screen.findByRole("heading", { name: /tracking camera motion/i })).toBeInTheDocument();
    act(() => finishProcessing(completed));
    expect(await screen.findByTestId("three-scene")).toBeInTheDocument();
  });

  it("shows actionable errors without losing the selected filename", async () => {
    vi.mocked(api.createRun).mockResolvedValue(accepted);
    vi.mocked(api.getRun).mockResolvedValue({
      ...completed,
      status: "FAILED",
      result: null,
      error: { code: "INITIALIZATION_FAILED", message: "Add sideways motion and more texture." },
    });
    render(<App pollIntervalMs={1} />);
    chooseVideo();
    await userEvent.click(screen.getByRole("button", { name: /start reconstruction/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/sideways motion/i);
    expect(screen.getByText("walk.mp4")).toBeInTheDocument();
  });

  it("discloses partial maps and visualization downsampling", async () => {
    vi.mocked(api.createRun).mockResolvedValue(accepted);
    vi.mocked(api.getRun).mockResolvedValue({ ...completed, status: "PARTIAL" });
    render(<App pollIntervalMs={1} />);
    chooseVideo();
    await userEvent.click(screen.getByRole("button", { name: /start reconstruction/i }));
    expect(await screen.findByText(/partial reconstruction/i)).toBeInTheDocument();
    expect(screen.getByText(/showing 1,200 of 1,800 points/i)).toBeInTheDocument();
    expect(screen.getByText(/arbitrary relative units/i)).toBeInTheDocument();
  });

  it("toggles visualization layers and resets the workspace", async () => {
    vi.mocked(api.createRun).mockResolvedValue(accepted);
    vi.mocked(api.getRun).mockResolvedValue(completed);
    render(<App pollIntervalMs={1} />);
    chooseVideo();
    await userEvent.click(screen.getByRole("button", { name: /start reconstruction/i }));
    const points = await screen.findByRole("checkbox", { name: /point cloud/i });
    expect(points).toBeChecked();
    await userEvent.click(points);
    expect(points).not.toBeChecked();
    await userEvent.click(screen.getByRole("button", { name: /new reconstruction/i }));
    expect(screen.getByText(/drop a short monocular video/i)).toBeInTheDocument();
  });

  it("defaults to light and persists an intentional dark choice", async () => {
    const { unmount } = render(<App />);
    expect(document.documentElement.dataset.theme).toBe("light");
    await userEvent.click(screen.getByRole("button", { name: /switch to dark theme/i }));
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(localStorage.getItem("slam-theme")).toBe("dark");
    unmount();
    render(<App />);
    await waitFor(() => expect(document.documentElement.dataset.theme).toBe("dark"));
  });
});
