"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";

export default function AntigravityBackground({
  count = 280,
  magnetRadius = 1,
  ringRadius = 10,
  waveSpeed = 0.4,
  waveAmplitude = 1,
  particleSize = 1.8,
  lerpSpeed = 0.08,
  color = "#94a3b8",
  autoAnimate = true,
  particleVariance = 1,
  rotationSpeed = 0.05,
  depthFactor = 1,
  pulseSpeed = 2.5,
  fieldStrength = 10,
}) {
  const containerRef = useRef(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let width = window.innerWidth;
    let height = window.innerHeight;

    // Scene & Camera
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, width / height, 0.1, 1000);
    camera.position.set(0, 0, 50);

    // Renderer
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // Viewport calculation at camera distance 50
    const vHeight = 2 * Math.tan((camera.fov * Math.PI) / 360) * camera.position.z;
    const vWidth = vHeight * camera.aspect;

    // Geometry & Material (matching React Bits Antigravity capsule specification)
    const geometry = new THREE.CapsuleGeometry(0.08, 0.35, 4, 8);
    const material = new THREE.MeshBasicMaterial({
      color: new THREE.Color(color),
      transparent: true,
      opacity: 0.65,
    });

    const mesh = new THREE.InstancedMesh(geometry, material, count);
    scene.add(mesh);

    const dummy = new THREE.Object3D();

    // Particle field generation
    const particles = [];
    for (let i = 0; i < count; i++) {
      const t = Math.random() * 100;
      const speed = 0.01 + Math.random() / 200;
      const x = (Math.random() - 0.5) * vWidth * 1.3;
      const y = (Math.random() - 0.5) * vHeight * 1.3;
      const z = (Math.random() - 0.5) * 20;
      const randomRadiusOffset = (Math.random() - 0.5) * 2;

      particles.push({
        t,
        speed,
        mx: x,
        my: y,
        mz: z,
        cx: x,
        cy: y,
        cz: z,
        randomRadiusOffset,
      });
    }

    // Mouse Tracking
    const mouse = { x: 0, y: 0 };
    const lastMousePos = { x: 0, y: 0 };
    let lastMouseMoveTime = Date.now();
    const virtualMouse = { x: 0, y: 0 };

    const handleMouseMove = (e) => {
      mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

      const dist = Math.hypot(mouse.x - lastMousePos.x, mouse.y - lastMousePos.y);
      if (dist > 0.001) {
        lastMouseMoveTime = Date.now();
        lastMousePos.x = mouse.x;
        lastMousePos.y = mouse.y;
      }
    };

    window.addEventListener("mousemove", handleMouseMove, { passive: true });

    // Resize Handler
    const handleResize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };

    window.addEventListener("resize", handleResize);

    // Animation Loop
    let animId;
    let clock = new THREE.Clock();

    const animate = () => {
      animId = requestAnimationFrame(animate);

      const elapsedTime = clock.getElapsedTime();
      const curVHeight = 2 * Math.tan((camera.fov * Math.PI) / 360) * camera.position.z;
      const curVWidth = curVHeight * camera.aspect;

      let destX = (mouse.x * curVWidth) / 2;
      let destY = (mouse.y * curVHeight) / 2;

      // Gentle auto-float if mouse is idle
      if (autoAnimate && Date.now() - lastMouseMoveTime > 1800) {
        destX = Math.sin(elapsedTime * 0.4) * (curVWidth / 4);
        destY = Math.cos(elapsedTime * 0.4 * 1.5) * (curVHeight / 4);
      }

      const smoothFactor = 0.06;
      virtualMouse.x += (destX - virtualMouse.x) * smoothFactor;
      virtualMouse.y += (destY - virtualMouse.y) * smoothFactor;

      const targetX = virtualMouse.x;
      const targetY = virtualMouse.y;
      const globalRotation = elapsedTime * rotationSpeed;

      for (let i = 0; i < count; i++) {
        const p = particles[i];
        p.t += p.speed / 2;

        const projectionFactor = 1 - p.cz / 50;
        const projectedTargetX = targetX * projectionFactor;
        const projectedTargetY = targetY * projectionFactor;

        const dx = p.mx - projectedTargetX;
        const dy = p.my - projectedTargetY;
        const dist = Math.hypot(dx, dy);

        let targetPosX = p.mx;
        let targetPosY = p.my;
        let targetPosZ = p.mz * depthFactor;

        if (dist < magnetRadius) {
          const angle = Math.atan2(dy, dx) + globalRotation;
          const wave = Math.sin(p.t * waveSpeed + angle) * (0.5 * waveAmplitude);
          const deviation = p.randomRadiusOffset * (5 / (fieldStrength + 0.1));
          const currentRingRadius = ringRadius + wave + deviation;

          targetPosX = projectedTargetX + currentRingRadius * Math.cos(angle);
          targetPosY = projectedTargetY + currentRingRadius * Math.sin(angle);
          targetPosZ = p.mz * depthFactor + Math.sin(p.t) * (1 * waveAmplitude * depthFactor);
        }

        p.cx += (targetPosX - p.cx) * lerpSpeed;
        p.cy += (targetPosY - p.cy) * lerpSpeed;
        p.cz += (targetPosZ - p.cz) * lerpSpeed;

        dummy.position.set(p.cx, p.cy, p.cz);
        dummy.lookAt(projectedTargetX, projectedTargetY, p.cz);
        dummy.rotateX(Math.PI / 2);

        const currentDistToMouse = Math.hypot(p.cx - projectedTargetX, p.cy - projectedTargetY);
        const distFromRing = Math.abs(currentDistToMouse - ringRadius);
        let scaleFactor = Math.max(0, Math.min(1, 1 - distFromRing / 10));

        const finalScale =
          scaleFactor * (0.8 + Math.sin(p.t * pulseSpeed) * 0.2 * particleVariance) * particleSize;
        dummy.scale.set(finalScale, finalScale, finalScale);

        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
      }

      mesh.instanceMatrix.needsUpdate = true;
      renderer.render(scene, camera);
    };

    animate();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("resize", handleResize);
      renderer.dispose();
      geometry.dispose();
      material.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [
    count,
    magnetRadius,
    ringRadius,
    waveSpeed,
    waveAmplitude,
    particleSize,
    lerpSpeed,
    color,
    autoAnimate,
    particleVariance,
    rotationSpeed,
    depthFactor,
    pulseSpeed,
    fieldStrength,
  ]);

  return (
    <div
      ref={containerRef}
      className="fixed inset-0 pointer-events-none z-0 overflow-hidden"
      style={{ opacity: 0.85 }}
    />
  );
}

