import React, { useState, useEffect, useRef } from "react";
import * as THREE from "three";
import anime from 'animejs';
import { Play, Activity, ShieldCheck, FileText, Database, Code, Terminal } from "lucide-react";

export default function App() {
  const [logs, setLogs] = useState([]);
  const [isTriaging, setIsTriaging] = useState(false);
  const mountRef = useRef(null);
  const meshRef = useRef(null);

  useEffect(() => {
    // 1. Scene Setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#001219');
    scene.fog = new THREE.FogExp2('#005f73', 0.015);

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 5, 18);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    if (mountRef.current) mountRef.current.appendChild(renderer.domElement);

    // 2. Lighting (Cyan pulse & Warm Amber key)
    const pointLight = new THREE.PointLight('#94d2bd', 2, 50);
    pointLight.position.set(0, 0, 0);
    scene.add(pointLight);

    const keyLight = new THREE.DirectionalLight('#e9d8a6', 1.5);
    keyLight.position.set(10, 10, 10);
    scene.add(keyLight);
    scene.add(new THREE.AmbientLight('#005f73', 0.5));

    // 3. 3D Cube Lattice (6x6x6)
    const gridSize = 6;
    const count = gridSize * gridSize * gridSize;
    const geometry = new THREE.BoxGeometry(0.7, 0.7, 0.7);
    const material = new THREE.MeshPhysicalMaterial({
      color: '#0a9396', metalness: 0.2, roughness: 0.1, transmission: 0.9, thickness: 0.5,
    });

    const instancedMesh = new THREE.InstancedMesh(geometry, material, count);
    meshRef.current = instancedMesh;
    scene.add(instancedMesh);

    const dummy = new THREE.Object3D();
    const instances = [];
    const offset = (gridSize - 1) / 2;
    let index = 0;

    for (let x = 0; x < gridSize; x++) {
      for (let y = 0; y < gridSize; y++) {
        for (let z = 0; z < gridSize; z++) {
          const posX = (x - offset) * 1.5;
          const posY = (y - offset) * 1.5;
          const posZ = (z - offset) * 1.5;

          instances.push({ x: posX, y: posY, z: posZ, baseX: posX, baseY: posY, baseZ: posZ });
          dummy.position.set(posX, posY, posZ);
          dummy.updateMatrix();
          instancedMesh.setMatrixAt(index, dummy.matrix);

          // Amber accent cubes
          if (Math.random() > 0.9) instancedMesh.setColorAt(index, new THREE.Color('#ee9b00'));
          else instancedMesh.setColorAt(index, new THREE.Color('#94d2bd'));
          index++;
        }
      }
    }
    instancedMesh.instanceMatrix.needsUpdate = true;
    instancedMesh.instanceColor.needsUpdate = true;

    // 4. Anime.js Animations
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!prefersReducedMotion) {
      anime({ targets: pointLight, intensity: [1, 5], direction: 'alternate', loop: true, easing: 'easeInOutSine', duration: 2000 });
      anime({ targets: instancedMesh.rotation, y: Math.PI * 2, x: Math.PI * 2, duration: 25000, loop: true, easing: 'linear' });

      anime({
        targets: instances,
        x: (el) => el.baseX * 2,
        y: (el) => el.baseY * 2,
        z: (el) => el.baseZ * 2,
        delay: anime.stagger(15, { grid: [gridSize, gridSize, gridSize], from: 'center' }),
        direction: 'alternate',
        loop: true,
        easing: 'easeInOutQuad',
        duration: 3000,
        update: () => {
          instances.forEach((inst, i) => {
            dummy.position.set(inst.x, inst.y, inst.z);
            dummy.updateMatrix();
            instancedMesh.setMatrixAt(i, dummy.matrix);
          });
          instancedMesh.instanceMatrix.needsUpdate = true;
        }
      });
    }

    // 5. Interaction
    const onMouseMove = (e) => {
      if (prefersReducedMotion) return;
      const x = (e.clientX / window.innerWidth) * 2 - 1;
      const y = -(e.clientY / window.innerHeight) * 2 + 1;
      anime({ targets: camera.position, x: x * 6, y: 5 + y * 6, duration: 500, easing: 'easeOutQuad' });
      camera.lookAt(0, 0, 0);
    };
    window.addEventListener('mousemove', onMouseMove);

    // 6. Render Loop
    const animateLoop = () => {
      requestAnimationFrame(animateLoop);
      renderer.render(scene, camera);
    };
    animateLoop();

    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      if (mountRef.current) mountRef.current.removeChild(renderer.domElement);
    };
  }, []);

  const startTriage = () => {
    setIsTriaging(true);
    setLogs([]);
    
    // Dynamically grabs the server IP so it works from your laptop
    const serverUrl = `http://${window.location.hostname}:8000/api/v1/triage/PT-101`;
    const eventSource = new EventSource(serverUrl);

    eventSource.onmessage = (e) => {
      setLogs((prev) => [...prev, e.data]);
      if (e.data.includes("TRIAGE COMPLETE") || e.data.includes("ERROR")) {
        eventSource.close();
        setIsTriaging(false);
      }
    };

    eventSource.onerror = (err) => {
      setLogs((prev) => [...prev, "❌ ERROR: Cannot connect to FastAPI Server. Did you enable CORS?"]);
      eventSource.close();
      setIsTriaging(false);
    };
  };

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh', overflow: 'hidden', backgroundColor: '#001219', color: '#e9d8a6', fontFamily: 'system-ui, sans-serif' }}>
      <div ref={mountRef} style={{ position: 'absolute', top: 0, left: 0, zIndex: 0 }} />

      <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 1, pointerEvents: 'none', display: 'flex', flexDirection: 'column' }}>
        
        {/* Header */}
        <div style={{ padding: '2.5rem', background: 'linear-gradient(180deg, rgba(0,18,25,0.9) 0%, rgba(0,0,0,0) 100%)' }}>
          <h1 style={{ margin: 0, fontSize: '3.5rem', fontWeight: 800, background: 'linear-gradient(90deg, #e9d8a6, #ee9b00, #ca6702)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <Activity size={48} color="#ee9b00" /> Aegis-Med
          </h1>
          <p style={{ margin: '0.5rem 0 0 0', fontSize: '1.25rem', color: '#94d2bd', maxWidth: '700px' }}>
            H100-Powered Autonomous Clinical Triage Agent. Dynamic LoRA Swapping & PostgreSQL Hybrid RAG.
          </p>
        </div>

        {/* Start Button */}
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', padding: '0 2.5rem' }}>
          <div style={{ pointerEvents: 'auto' }}>
            <button
              onClick={startTriage}
              disabled={isTriaging}
              style={{
                display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '1.25rem 2.5rem', fontSize: '1.25rem', fontWeight: 700,
                color: '#001219', background: isTriaging ? '#94d2bd' : '#ee9b00', border: 'none', borderRadius: '12px', cursor: isTriaging ? 'not-allowed' : 'pointer',
                boxShadow: isTriaging ? 'none' : '0 0 30px rgba(238, 155, 0, 0.4)', transition: 'all 0.3s ease'
              }}
            >
              {isTriaging ? <Activity size={24} style={{ animation: 'spin 2s linear infinite' }} /> : <Play size={24} />}
              {isTriaging ? 'Agent Orchestrating...' : 'Initialize Triage FSM'}
            </button>
          </div>
        </div>

        {/* Live SSE Logs Terminal */}
        <div style={{
          height: '45%', background: 'rgba(0, 18, 25, 0.85)', backdropFilter: 'blur(12px)', borderTop: '1px solid #0a9396',
          padding: '2rem', overflowY: 'auto', pointerEvents: 'auto', fontFamily: 'monospace', display: 'flex', flexDirection: 'column', gap: '0.75rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#e9d8a6', marginBottom: '1rem', fontSize: '1.2rem', fontWeight: 700 }}>
            <Terminal size={24} /> FSM Orchestrator Stream
          </div>
          {logs.length === 0 && <div style={{ color: '#0a9396', fontSize: '1.1rem' }}>Awaiting execution trigger...</div>}
          
          {logs.map((log, idx) => {
            let color = '#e9d8a6';
            let Icon = Code;
            if (log.includes('ERROR')) { color = '#ca6702'; Icon = Activity; }
            else if (log.includes('GUARDRAIL') || log.includes('CITATION') || log.includes('COMPLETE')) { color = '#94d2bd'; Icon = ShieldCheck; }
            else if (log.includes('ACTION:')) { color = '#ee9b00'; Icon = Database; }
            else if (log.includes('RAW LLM')) { color = '#0a9396'; Icon = FileText; }

            return (
              <div key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem', color, padding: '0.75rem', background: 'rgba(255,255,255,0.03)', borderRadius: '6px', fontSize: '1.1rem' }}>
                <Icon size={18} style={{ marginTop: '0.2rem', flexShrink: 0 }} />
                <span style={{ whiteSpace: 'pre-wrap', lineHeight: '1.4' }}>{log}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
