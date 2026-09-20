import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type { SlamPayload } from "../api";

export interface SceneLayers {
  points: boolean;
  trajectory: boolean;
  keyframes: boolean;
  grid: boolean;
}

interface SceneProps {
  result: SlamPayload;
  layers: SceneLayers;
  resetSignal: number;
}

function cameraCenter(matrix: number[][]): THREE.Vector3 {
  const rotation = new THREE.Matrix3().set(
    matrix[0][0], matrix[0][1], matrix[0][2],
    matrix[1][0], matrix[1][1], matrix[1][2],
    matrix[2][0], matrix[2][1], matrix[2][2],
  );
  const inverse = rotation.clone().transpose();
  return new THREE.Vector3(matrix[0][3], matrix[1][3], matrix[2][3])
    .applyMatrix3(inverse)
    .multiplyScalar(-1);
}

export function Scene({ result, layers, resetSignal }: SceneProps) {
  const hostRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return undefined;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(52, 1, 0.01, 1000);
    camera.position.set(4, 3, 6);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.append(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const root = new THREE.Group();
    scene.add(root);
    const pointPositions = new Float32Array(result.points.length * 3);
    result.points.forEach((point, index) => pointPositions.set(point.position, index * 3));
    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute("position", new THREE.BufferAttribute(pointPositions, 3));
    pointGeometry.computeBoundingSphere();
    const pointCloud = new THREE.Points(
      pointGeometry,
      new THREE.PointsMaterial({ color: 0x27c5a8, size: 0.025, sizeAttenuation: true }),
    );
    pointCloud.visible = layers.points;
    root.add(pointCloud);

    const centers = result.poses.map((pose) => cameraCenter(pose.world_to_camera));
    const trajectoryGeometry = new THREE.BufferGeometry().setFromPoints(centers);
    const trajectory = new THREE.Line(
      trajectoryGeometry,
      new THREE.LineBasicMaterial({ color: 0xff6b45, transparent: true, opacity: 0.92 }),
    );
    trajectory.visible = layers.trajectory;
    root.add(trajectory);

    const keyframeGroup = new THREE.Group();
    result.poses.forEach((pose, index) => {
      if (!pose.keyframe) return;
      const marker = new THREE.Mesh(
        new THREE.ConeGeometry(0.055, 0.16, 4),
        new THREE.MeshBasicMaterial({ color: index === 0 ? 0xff6b45 : 0xf3c969 }),
      );
      marker.position.copy(centers[index]);
      marker.rotation.x = Math.PI / 2;
      keyframeGroup.add(marker);
    });
    keyframeGroup.visible = layers.keyframes;
    root.add(keyframeGroup);

    const grid = new THREE.GridHelper(12, 24, 0x769187, 0xb6c5bf);
    grid.material.transparent = true;
    grid.material.opacity = 0.18;
    grid.visible = layers.grid;
    root.add(grid);

    const bounds = new THREE.Box3().setFromObject(root);
    const center = bounds.isEmpty() ? new THREE.Vector3() : bounds.getCenter(new THREE.Vector3());
    const size = bounds.isEmpty() ? 2 : Math.max(...bounds.getSize(new THREE.Vector3()).toArray());
    root.position.sub(center);
    camera.position.set(size * 1.2, size * 0.8, size * 1.5);
    controls.target.set(0, 0, 0);
    controls.update();

    const resize = () => {
      const width = Math.max(1, host.clientWidth);
      const height = Math.max(1, host.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();
    let animationFrame = 0;
    const render = () => {
      controls.update();
      renderer.render(scene, camera);
      animationFrame = requestAnimationFrame(render);
    };
    render();
    return () => {
      cancelAnimationFrame(animationFrame);
      observer.disconnect();
      controls.dispose();
      pointGeometry.dispose();
      (pointCloud.material as THREE.Material).dispose();
      trajectoryGeometry.dispose();
      (trajectory.material as THREE.Material).dispose();
      keyframeGroup.children.forEach((child) => {
        const mesh = child as THREE.Mesh;
        mesh.geometry.dispose();
        (mesh.material as THREE.Material).dispose();
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [result, layers.points, layers.trajectory, layers.keyframes, layers.grid, resetSignal]);

  return <div ref={hostRef} className="scene" role="img" aria-label="Interactive sparse point cloud and camera trajectory" />;
}
