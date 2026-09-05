"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  Camera,
  Upload,
  Plus,
  Trash2,
  Play,
  RotateCcw,
  Sparkles,
  Sliders,
  Download,
  CheckCircle2,
  AlertCircle,
  Video,
  VideoOff,
  Layers,
  Cpu,
  Zap,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

export default function CustomVisionStudio() {
  // Classes state: [{ id: "c1", name: "Class 1", samples: [b64, ...] }]
  const [classes, setClasses] = useState([
    { id: "c1", name: "Class 1", samples: [] },
    { id: "c2", name: "Class 2", samples: [] },
  ]);
  const [activeClassId, setActiveClassId] = useState("c1");

  // Model & Training Configuration
  const [backbone, setBackbone] = useState("mobilenet_v3_small");
  const [epochs, setEpochs] = useState(10);
  const [learningRate, setLearningRate] = useState(0.001);
  const [batchSize, setBatchSize] = useState(8);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Unified Capture Camera
  const [isCaptureCamOpen, setIsCaptureCamOpen] = useState(false);
  const [isRecordingBurst, setIsRecordingBurst] = useState(false);
  const captureVideoRef = useRef(null);
  const captureStreamRef = useRef(null);
  const burstIntervalRef = useRef(null);

  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [trainProgress, setTrainProgress] = useState(0);
  const [trainStatusText, setTrainStatusText] = useState("");
  const [trainedModel, setTrainedModel] = useState(null);
  const [trainError, setTrainError] = useState(null);

  // Preview & Live Inference state
  const [previewMode, setPreviewMode] = useState("webcam");
  const [previewActive, setPreviewActive] = useState(false);
  const previewVideoRef = useRef(null);
  const previewStreamRef = useRef(null);
  const previewLoopRef = useRef(null);
  const runInferenceRef = useRef(null);
  const [testFilePreview, setTestFilePreview] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [topPrediction, setTopPrediction] = useState(null);
  const [inferenceLatency, setInferenceLatency] = useState(null);

  // Past models list
  const [pastModels, setPastModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState(null);

  const activeTestingModelId = trainedModel ? trainedModel.model_id : selectedModelId;
  const activeModelIdRef = useRef(activeTestingModelId);
  const isPreviewActiveRef = useRef(previewActive);

  useEffect(() => {
    activeModelIdRef.current = activeTestingModelId;
  }, [activeTestingModelId]);

  useEffect(() => {
    isPreviewActiveRef.current = previewActive;
  }, [previewActive]);

  // Active class object
  const currentClass = classes.find((c) => c.id === activeClassId) || classes[0];

  // Fetch past models on mount
  const fetchModels = useCallback(async () => {
    try {
      const res = await fetch("http://localhost:8000/api/classifier/models");
      if (res.ok) {
        const data = await res.json();
        if (data?.models) {
          setPastModels(data.models);
        }
      }
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    fetch("http://localhost:8000/api/classifier/models")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (mounted && data?.models) {
          setPastModels(data.models);
        }
      })
      .catch(() => { });
    return () => {
      mounted = false;
    };
  }, []);

  // -------------------------------------------------------------------------
  // Capture Webcam Lifecycle (Shared between classes)
  // -------------------------------------------------------------------------
  const startCaptureCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240, facingMode: "user" },
      });
      captureStreamRef.current = stream;
      if (captureVideoRef.current) {
        captureVideoRef.current.srcObject = stream;
      }
      setIsCaptureCamOpen(true);
    } catch (err) {
      alert("Unable to access camera: " + err.message);
    }
  };

  const stopCaptureCamera = useCallback(() => {
    if (burstIntervalRef.current) {
      clearInterval(burstIntervalRef.current);
      burstIntervalRef.current = null;
    }
    setIsRecordingBurst(false);
    if (captureStreamRef.current) {
      captureStreamRef.current.getTracks().forEach((t) => t.stop());
      captureStreamRef.current = null;
    }
    setIsCaptureCamOpen(false);
  }, []);

  const stopPreviewWebcam = useCallback(() => {
    if (previewLoopRef.current) {
      cancelAnimationFrame(previewLoopRef.current);
      previewLoopRef.current = null;
    }
    if (previewStreamRef.current) {
      previewStreamRef.current.getTracks().forEach((t) => t.stop());
      previewStreamRef.current = null;
    }
    setPreviewActive(false);
  }, []);

  useEffect(() => {
    if (isCaptureCamOpen && captureVideoRef.current && captureStreamRef.current) {
      captureVideoRef.current.srcObject = captureStreamRef.current;
    }
  }, [isCaptureCamOpen]);

  // Clean up streams on unmount
  useEffect(() => {
    return () => {
      stopCaptureCamera();
      stopPreviewWebcam();
    };
  }, [stopCaptureCamera, stopPreviewWebcam]);

  // Frame Capture Utility
  const captureFrame = (videoEl) => {
    if (!videoEl || videoEl.readyState < 2) return null;
    const canvas = document.createElement("canvas");
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.85);
  };

  const captureSingleSnapshot = () => {
    if (!captureVideoRef.current || !currentClass) return;
    const frame = captureFrame(captureVideoRef.current);
    if (frame) {
      setClasses((prev) =>
        prev.map((c) =>
          c.id === currentClass.id ? { ...c, samples: [...c.samples, frame] } : c
        )
      );
    }
  };

  const startBurstRecording = () => {
    if (!captureVideoRef.current || !currentClass) return;
    setIsRecordingBurst(true);
    captureSingleSnapshot();
    burstIntervalRef.current = setInterval(() => {
      if (captureVideoRef.current) {
        const frame = captureFrame(captureVideoRef.current);
        if (frame) {
          setClasses((prev) =>
            prev.map((c) =>
              c.id === currentClass.id ? { ...c, samples: [...c.samples, frame] } : c
            )
          );
        }
      }
    }, 100);
  };

  const stopBurstRecording = () => {
    if (burstIntervalRef.current) {
      clearInterval(burstIntervalRef.current);
      burstIntervalRef.current = null;
    }
    setIsRecordingBurst(false);
  };

  // File upload for active class
  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length || !currentClass) return;

    files.forEach((file) => {
      const reader = new FileReader();
      reader.onload = (ev) => {
        const b64 = ev.target.result;
        setClasses((prev) =>
          prev.map((c) =>
            c.id === currentClass.id ? { ...c, samples: [...c.samples, b64] } : c
          )
        );
      };
      reader.readAsDataURL(file);
    });
    e.target.value = "";
  };

  const removeSample = (classId, indexToRemove) => {
    setClasses((prev) =>
      prev.map((c) =>
        c.id === classId
          ? { ...c, samples: c.samples.filter((_, idx) => idx !== indexToRemove) }
          : c
      )
    );
  };

  const clearCurrentSamples = () => {
    if (!currentClass) return;
    setClasses((prev) =>
      prev.map((c) => (c.id === currentClass.id ? { ...c, samples: [] } : c))
    );
  };

  const addClass = () => {
    const nextNum = classes.length + 1;
    const newId = `c_${Date.now()}`;
    setClasses((prev) => [...prev, { id: newId, name: `Class ${nextNum}`, samples: [] }]);
    setActiveClassId(newId);
  };

  const removeClass = (classId) => {
    if (classes.length <= 2) {
      alert("A classifier requires at least 2 classes to compare.");
      return;
    }
    const remaining = classes.filter((c) => c.id !== classId);
    setClasses(remaining);
    if (activeClassId === classId) {
      setActiveClassId(remaining[0].id);
    }
  };

  const updateClassName = (classId, newName) => {
    setClasses((prev) =>
      prev.map((c) => (c.id === classId ? { ...c, name: newName } : c))
    );
  };

  // -------------------------------------------------------------------------
  // Model Training
  // -------------------------------------------------------------------------
  const totalSamplesCount = classes.reduce((acc, c) => acc + c.samples.length, 0);
  const emptyClasses = classes.filter((c) => c.samples.length === 0);
  const canTrain = classes.length >= 2 && emptyClasses.length === 0 && !isTraining;

  const handleTrainClick = async () => {
    if (emptyClasses.length > 0) {
      alert(
        `Cannot train yet: Class "${emptyClasses[0].name}" has 0 samples.\n\n` +
        `Click on "${emptyClasses[0].name}" above, record some samples, then click Train.`
      );
      return;
    }

    setIsTraining(true);
    setTrainError(null);
    setTrainProgress(15);
    setTrainStatusText("Structuring tensors & augmentations...");

    const payload = {
      classes: {},
      backbone: backbone,
      epochs: epochs,
      lr: learningRate,
      batch_size: batchSize,
    };

    classes.forEach((c) => {
      payload.classes[c.name] = c.samples;
    });

    try {
      setTrainProgress(45);
      setTrainStatusText(`Training transfer head on ${backbone === "resnet18" ? "ResNet18" : "MobileNetV3"}...`);

      const res = await fetch("http://localhost:8000/api/classifier/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      setTrainProgress(85);
      setTrainStatusText("Evaluating validation accuracy & exporting model...");

      if (res.ok) {
        const data = await res.json();
        setTrainedModel(data.model);
        setSelectedModelId(data.model.model_id);
        setTrainProgress(100);
        setTrainStatusText("Training complete! Model is ready.");
        fetchModels();
        // Start live preview automatically
        setPreviewActive(true);
      } else {
        const err = await res.json().catch(() => ({}));
        setTrainError(err.detail || "Training failed.");
      }
    } catch (err) {
      setTrainError("Network error: " + err.message);
    } finally {
      setIsTraining(false);
    }
  };

  // -------------------------------------------------------------------------
  // Live Testing & Inference Lifecycle
  // -------------------------------------------------------------------------
  const startPreviewWebcam = async () => {
    if (!activeModelIdRef.current) {
      alert("Please train your classes first on the left.");
      return;
    }
    // Stop capture camera to free up webcam device
    stopCaptureCamera();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240, facingMode: "user" },
      });
      previewStreamRef.current = stream;
      if (previewVideoRef.current) {
        previewVideoRef.current.srcObject = stream;
      }
      setPreviewActive(true);
    } catch (err) {
      alert("Unable to open preview camera: " + err.message);
    }
  };



  const runInferenceIteration = useCallback(async () => {
    const currentModelId = activeModelIdRef.current;
    if (!isPreviewActiveRef.current || !previewVideoRef.current || !currentModelId) return;

    const frame = captureFrame(previewVideoRef.current);
    if (frame) {
      try {
        const res = await fetch("http://localhost:8000/api/classifier/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: frame,
            model_id: currentModelId,
          }),
        });

        if (res.ok) {
          const data = await res.json();
          setPredictions(data.predictions || []);
          setTopPrediction(data.top_class);
          setInferenceLatency(data.speed_ms);
        }
      } catch {
        // Drop frames silently
      }
    }

    if (isPreviewActiveRef.current && activeModelIdRef.current && runInferenceRef.current) {
      setTimeout(() => {
        previewLoopRef.current = requestAnimationFrame(runInferenceRef.current);
      }, 100);
    }
  }, []);

  useEffect(() => {
    runInferenceRef.current = runInferenceIteration;
  }, [runInferenceIteration]);

  useEffect(() => {
    if (previewActive && previewMode === "webcam" && activeModelIdRef.current) {
      if (previewVideoRef.current && previewStreamRef.current) {
        previewVideoRef.current.srcObject = previewStreamRef.current;
      } else if (!previewStreamRef.current) {
        startPreviewWebcam();
      }
      previewLoopRef.current = requestAnimationFrame(runInferenceIteration);
    } else {
      if (previewLoopRef.current) {
        cancelAnimationFrame(previewLoopRef.current);
        previewLoopRef.current = null;
      }
    }
    return () => {
      if (previewLoopRef.current) {
        cancelAnimationFrame(previewLoopRef.current);
      }
    };
  }, [previewActive, previewMode, runInferenceIteration]);

  // File test prediction
  const handleTestFileUpload = async (e) => {
    if (!activeTestingModelId) {
      alert("Train a model first to test images.");
      return;
    }
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (ev) => {
      const b64 = ev.target.result;
      setTestFilePreview(b64);

      try {
        const res = await fetch("http://localhost:8000/api/classifier/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: b64,
            model_id: activeTestingModelId,
          }),
        });
        if (res.ok) {
          const data = await res.json();
          setPredictions(data.predictions || []);
          setTopPrediction(data.top_class);
          setInferenceLatency(data.speed_ms);
        }
      } catch (err) {
        alert("Prediction failed: " + err.message);
      }
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  };

  return (
    <div className="flex flex-col gap-4 text-[#E0E0E0] max-w-7xl mx-auto">
      {/* Minimal Top Header */}
      <div className="flex items-center justify-between border-b border-[#222222] pb-2.5 flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Layers className="text-[#00E5FF]" size={16} />
            <h2 className="text-sm font-mono font-bold tracking-wider text-white">
              CUSTOM VISION CLASSIFIER
            </h2>
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 bg-[#00E5FF]/10 text-[#00E5FF] border border-[#00E5FF]/30">
            PYTORCH
          </span>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          <span className="text-[#777777]">
            Classes: <strong className="text-white">{classes.length}</strong>
          </span>
          <span className="text-[#777777]">
            Samples:{" "}
            <strong className={totalSamplesCount > 0 ? "text-[#00E5FF]" : "text-[#777777]"}>
              {totalSamplesCount}
            </strong>
          </span>
        </div>
      </div>

      {/* Main Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* ================================================================= */}
        {/* LEFT COLUMN: CLASSES & STREAMLINED CAPTURE (7 cols)               */}
        {/* ================================================================= */}
        <div className="lg:col-span-7 flex flex-col gap-3">
          {/* Class Navigation Strip */}
          <div className="flex items-center gap-1.5 overflow-x-auto pb-1 border-b border-[#1A1A1A]">
            {classes.map((cls, idx) => {
              const isActive = cls.id === activeClassId;
              const hasNoSamples = cls.samples.length === 0;

              return (
                <button
                  key={cls.id}
                  type="button"
                  onClick={() => setActiveClassId(cls.id)}
                  className={`px-3 py-1.5 text-xs font-mono transition-all flex items-center gap-2 shrink-0 border ${isActive
                    ? "border-[#00E5FF] bg-[#00E5FF]/10 text-white font-bold shadow-[0_0_10px_rgba(0,229,255,0.15)]"
                    : "border-[#252525] bg-[#121212] text-[#888888] hover:border-[#444444] hover:text-white"
                    }`}
                >
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${hasNoSamples ? "bg-amber-400 animate-pulse" : "bg-emerald-400"
                      }`}
                  />
                  <span>
                    #{idx + 1} {cls.name}
                  </span>
                  <span
                    className={`text-[10px] px-1.5 py-0.2 rounded-xs ${hasNoSamples
                      ? "bg-amber-950/60 text-amber-300"
                      : "bg-[#222222] text-[#AAAAAA]"
                      }`}
                  >
                    {cls.samples.length}
                  </span>
                </button>
              );
            })}

            <button
              type="button"
              onClick={addClass}
              title="Add new class"
              className="px-2.5 py-1.5 text-xs font-mono bg-[#141414] hover:bg-[#202020] border border-[#2A2A2A] hover:border-[#00E5FF] text-white transition-colors flex items-center gap-1 shrink-0"
            >
              <Plus size={13} /> Add
            </button>
          </div>

          {/* Active Class Studio Card */}
          {currentClass && (
            <div className="border border-[#222222] bg-[#111111] p-4 flex flex-col gap-3">
              {/* Class Edit Header */}
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <div className="flex items-center gap-2 flex-1">
                  <span className="text-xs font-mono text-[#555555]">Class Name:</span>
                  <input
                    type="text"
                    value={currentClass.name}
                    onChange={(e) => updateClassName(currentClass.id, e.target.value)}
                    className="bg-[#181818] border border-[#2D2D2D] focus:border-[#00E5FF] text-white px-2.5 py-1 text-xs font-mono font-bold outline-none max-w-[200px]"
                    placeholder="e.g. Hand, Object, Background"
                  />
                  <span
                    className={`text-[11px] font-mono px-2 py-0.5 ${currentClass.samples.length === 0
                      ? "text-amber-400 bg-amber-950/30 border border-amber-800/40"
                      : "text-emerald-400 bg-emerald-950/20 border border-emerald-800/30"
                      }`}
                  >
                    {currentClass.samples.length} sample{currentClass.samples.length !== 1 ? "s" : ""}
                    {currentClass.samples.length === 0 ? " (Empty)" : " (Ready)"}
                  </span>
                </div>

                <div className="flex items-center gap-1.5">
                  {currentClass.samples.length > 0 && (
                    <button
                      type="button"
                      onClick={clearCurrentSamples}
                      title="Clear all samples for this class"
                      className="px-2 py-1 text-[11px] font-mono text-[#777777] hover:text-amber-400 border border-transparent hover:border-[#333333] transition-colors flex items-center gap-1"
                    >
                      <RotateCcw size={11} /> Clear
                    </button>
                  )}
                  {classes.length > 2 && (
                    <button
                      type="button"
                      onClick={() => removeClass(currentClass.id)}
                      title="Delete class"
                      className="px-2 py-1 text-[11px] font-mono text-[#777777] hover:text-red-400 border border-transparent hover:border-[#333333] transition-colors flex items-center gap-1"
                    >
                      <Trash2 size={11} /> Delete
                    </button>
                  )}
                </div>
              </div>

              {/* Camera Viewfinder & Instant Controls */}
              <div className="border border-[#262626] bg-black p-3 flex flex-col md:flex-row gap-4 items-center">
                <div className="relative w-44 h-32 bg-[#0A0A0A] border border-[#2A2A2A] overflow-hidden shrink-0 flex items-center justify-center">
                  <video
                    ref={captureVideoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-full object-cover"
                  />

                  {!isCaptureCamOpen && (
                    <div className="absolute inset-0 bg-[#0E0E0E] flex flex-col items-center justify-center gap-1 text-center p-2">
                      <Camera size={18} className="text-[#555555]" />
                      <span className="text-[10px] font-mono text-[#777777]">Camera off</span>
                    </div>
                  )}

                  {isRecordingBurst && (
                    <div className="absolute top-1.5 right-1.5 flex items-center gap-1 bg-red-600 text-white text-[9px] px-1.5 py-0.5 font-mono animate-pulse">
                      <span className="w-1.5 h-1.5 rounded-full bg-white" />
                      RECORDING
                    </div>
                  )}
                </div>

                {/* Camera Actions */}
                <div className="flex flex-col gap-2 flex-1 w-full">
                  <div className="flex items-center gap-2">
                    {!isCaptureCamOpen ? (
                      <button
                        type="button"
                        onClick={startCaptureCamera}
                        className="text-xs font-mono px-3 py-1.5 bg-[#00E5FF] text-black font-bold hover:bg-[#00cbe2] transition-colors flex items-center gap-1.5"
                      >
                        <Camera size={13} /> OPEN CAMERA
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={stopCaptureCamera}
                        className="text-xs font-mono px-3 py-1.5 bg-red-950/30 hover:bg-red-900/40 border border-red-800 text-red-300 transition-colors flex items-center gap-1.5"
                      >
                        <VideoOff size={13} /> CLOSE CAMERA
                      </button>
                    )}

                    <label className="text-xs font-mono px-3 py-1.5 bg-[#181818] hover:bg-[#222222] border border-[#333333] hover:border-[#888888] text-white transition-colors cursor-pointer flex items-center gap-1.5">
                      <Upload size={13} className="text-[#888888]" /> UPLOAD IMAGES
                      <input
                        type="file"
                        multiple
                        accept="image/*"
                        onChange={handleFileUpload}
                        className="hidden"
                      />
                    </label>
                  </div>

                  {/* Recording buttons when camera is active */}
                  {isCaptureCamOpen && (
                    <div className="flex items-center gap-2 pt-1 border-t border-[#1C1C1C]">
                      <button
                        type="button"
                        onMouseDown={startBurstRecording}
                        onMouseUp={stopBurstRecording}
                        onTouchStart={startBurstRecording}
                        onTouchEnd={stopBurstRecording}
                        className={`flex-1 text-xs font-mono font-bold py-2 px-3 uppercase tracking-wider transition-all select-none flex items-center justify-center gap-1.5 ${isRecordingBurst
                          ? "bg-red-600 text-white scale-[0.98]"
                          : "bg-[#00E5FF] hover:bg-[#00cbe2] text-black"
                          }`}
                      >
                        <Camera size={13} />
                        {isRecordingBurst ? "RECORDING..." : `HOLD TO RECORD FOR "${currentClass.name.toUpperCase()}"`}
                      </button>
                      <button
                        type="button"
                        onClick={captureSingleSnapshot}
                        className="text-xs font-mono px-3 py-2 bg-[#1C1C1C] hover:bg-[#252525] border border-[#333333] text-white transition-colors whitespace-nowrap"
                      >
                        1 SHOT
                      </button>
                    </div>
                  )}

                  {!isCaptureCamOpen && (
                    <p className="text-[11px] font-mono text-[#666666]">
                      Click <strong>OPEN CAMERA</strong> to record snapshots directly into{" "}
                      <span className="text-[#00E5FF]">&quot;{currentClass.name}&quot;</span>.
                    </p>
                  )}
                </div>
              </div>

              {/* Horizontal Filmstrip of Collected Samples */}
              <div className="flex flex-col gap-1.5">
                <div className="flex justify-between text-[10px] font-mono text-[#777777]">
                  <span>SAMPLE FILMSTRIP ({currentClass.samples.length})</span>
                  {currentClass.samples.length > 0 && <span>Hover image to delete</span>}
                </div>

                {currentClass.samples.length > 0 ? (
                  <div className="flex gap-1.5 overflow-x-auto p-1.5 bg-[#0A0A0A] border border-[#1A1A1A] max-h-20">
                    {currentClass.samples.map((s, sIdx) => (
                      <div
                        key={sIdx}
                        className="group relative w-12 h-12 bg-[#181818] border border-[#2A2A2A] overflow-hidden shrink-0"
                      >
                        <img
                          src={s}
                          alt={`sample-${sIdx}`}
                          className="w-full h-full object-cover"
                        />
                        <button
                          type="button"
                          onClick={() => removeSample(currentClass.id, sIdx)}
                          className="absolute inset-0 bg-red-900/80 text-white opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity"
                        >
                          <Trash2 size={11} />
                        </button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-[11px] font-mono text-amber-400/80 py-2.5 px-3 border border-dashed border-amber-800/30 bg-amber-950/10 text-center">
                    No samples for &quot;{currentClass.name}&quot;. Open camera above to capture samples.
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* ================================================================= */}
        {/* RIGHT COLUMN: TRAINING & LIVE TESTING (5 cols)                    */}
        {/* ================================================================= */}
        <div className="lg:col-span-5 flex flex-col gap-3">
          {/* Card 1: Training Controls */}
          <div className="border border-[#222222] bg-[#111111] p-3.5 flex flex-col gap-2.5">
            <div className="flex items-center justify-between">
              <span className="text-xs uppercase font-mono font-bold tracking-wider text-[#AAAAAA] flex items-center gap-1.5">
                <Cpu size={13} className="text-[#00E5FF]" />
                TRAIN MODEL
              </span>

              {/* Advanced toggle */}
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="text-[10px] font-mono text-[#777777] hover:text-[#00E5FF] flex items-center gap-1 transition-colors"
              >
                <Sliders size={11} />
                {showAdvanced ? "Hide Params" : "Hyperparams"}
                {showAdvanced ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
              </button>
            </div>

            {/* Backbone Selection Pills */}
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setBackbone("mobilenet_v3_small")}
                className={`py-1.5 px-2 text-center text-xs font-mono transition-colors border ${backbone === "mobilenet_v3_small"
                  ? "border-[#00E5FF] bg-[#00E5FF]/10 text-white font-bold"
                  : "border-[#252525] bg-[#141414] text-[#777777] hover:border-[#3A3A3A]"
                  }`}
              >
                MobileNetV3 (Fast)
              </button>

              <button
                type="button"
                onClick={() => setBackbone("resnet18")}
                className={`py-1.5 px-2 text-center text-xs font-mono transition-colors border ${backbone === "resnet18"
                  ? "border-[#00E5FF] bg-[#00E5FF]/10 text-white font-bold"
                  : "border-[#252525] bg-[#141414] text-[#777777] hover:border-[#3A3A3A]"
                  }`}
              >
                ResNet18 (Deep)
              </button>
            </div>

            {/* Collapsible Advanced Hyperparameters */}
            {showAdvanced && (
              <div className="grid grid-cols-2 gap-2 p-2 bg-[#0D0D0D] border border-[#1E1E1E] text-[11px] font-mono">
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between">
                    <span className="text-[#777777]">Epochs:</span>
                    <span className="text-[#00E5FF] font-bold">{epochs}</span>
                  </div>
                  <input
                    type="range"
                    min={3}
                    max={30}
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value))}
                    className="accent-[#00E5FF]"
                  />
                </div>

                <div className="flex flex-col gap-1">
                  <div className="flex justify-between">
                    <span className="text-[#777777]">Batch:</span>
                    <span className="text-[#00E5FF] font-bold">{batchSize}</span>
                  </div>
                  <select
                    value={batchSize}
                    onChange={(e) => setBatchSize(parseInt(e.target.value))}
                    className="bg-[#181818] border border-[#2D2D2D] text-white px-1 py-0.5 text-[10px]"
                  >
                    <option value={4}>4</option>
                    <option value={8}>8</option>
                    <option value={16}>16</option>
                  </select>
                </div>
              </div>
            )}

            {/* Big Train Button */}
            <button
              type="button"
              onClick={handleTrainClick}
              className={`w-full py-2.5 px-3 font-mono text-xs uppercase tracking-wider font-bold transition-all flex items-center justify-center gap-1.5 cursor-pointer ${canTrain
                ? "bg-[#00E5FF] hover:bg-[#00cbe2] text-black shadow-[0_0_15px_rgba(0,229,255,0.25)]"
                : "bg-amber-500/15 hover:bg-amber-500/25 text-amber-300 border border-amber-500/40"
                }`}
            >
              <Play size={13} fill={canTrain ? "black" : "currentColor"} />
              {isTraining
                ? "TRAINING PYTORCH MODEL..."
                : canTrain
                  ? `TRAIN CLASSIFIER (${classes.length} CLASSES, ${totalSamplesCount} SAMPLES)`
                  : `SAMPLES REQUIRED (CLICK FOR INFO)`}
            </button>

            {/* Validation warning if empty */}
            {emptyClasses.length > 0 && !isTraining && (
              <p className="text-[10px] font-mono text-amber-400/90 text-center">
                ⚠️ &quot;{emptyClasses[0].name}&quot; needs samples to enable training.
              </p>
            )}

            {/* Progress */}
            {isTraining && (
              <div className="flex flex-col gap-1 p-2 bg-[#0E0E0E] border border-[#1F1F1F]">
                <div className="flex justify-between text-[10px] font-mono">
                  <span className="text-[#888888]">{trainStatusText}</span>
                  <span className="text-[#00E5FF] font-bold">{trainProgress}%</span>
                </div>
                <div className="w-full h-1 bg-[#222222] overflow-hidden">
                  <div
                    className="h-full bg-[#00E5FF] transition-all duration-200"
                    style={{ width: `${trainProgress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Trained Model Badge */}
            {trainedModel && (
              <div className="p-2 bg-emerald-950/20 border border-emerald-800/40 text-[11px] font-mono flex items-center justify-between">
                <span className="flex items-center gap-1 text-emerald-400 font-bold">
                  <CheckCircle2 size={13} /> READY ({(trainedModel.top1_accuracy * 100).toFixed(0)}% ACC)
                </span>
                <a
                  href={`http://localhost:8000/api/classifier/${trainedModel.model_id}/download`}
                  download
                  className="px-2 py-0.5 bg-[#181818] hover:bg-[#222222] border border-[#333333] text-white text-[10px] flex items-center gap-1"
                >
                  <Download size={10} /> .pth
                </a>
              </div>
            )}
          </div>

          {/* Card 2: Live Testing Studio */}
          <div className="border border-[#222222] bg-[#111111] p-3.5 flex flex-col gap-2.5">
            <div className="flex items-center justify-between">
              <span className="text-xs uppercase font-mono font-bold tracking-wider text-[#AAAAAA] flex items-center gap-1.5">
                <Sparkles size={13} className="text-[#00E5FF]" />
                LIVE TESTING
              </span>

              {inferenceLatency && previewActive && (
                <span className="text-[10px] font-mono px-1.5 py-0.2 bg-[#181818] border border-[#2A2A2A] text-[#00E5FF]">
                  {inferenceLatency} ms
                </span>
              )}
            </div>

            {/* Test Mode Switcher */}
            <div className="grid grid-cols-2 gap-1 border-b border-[#1A1A1A] pb-1.5">
              <button
                type="button"
                onClick={() => setPreviewMode("webcam")}
                className={`py-1 text-xs font-mono transition-colors text-center ${previewMode === "webcam"
                  ? "text-[#00E5FF] border-b border-[#00E5FF] font-bold"
                  : "text-[#777777] hover:text-white"
                  }`}
              >
                Webcam Test
              </button>
              <button
                type="button"
                onClick={() => {
                  stopPreviewWebcam();
                  setPreviewMode("file");
                }}
                className={`py-1 text-xs font-mono transition-colors text-center ${previewMode === "file"
                  ? "text-[#00E5FF] border-b border-[#00E5FF] font-bold"
                  : "text-[#777777] hover:text-white"
                  }`}
              >
                Image File Test
              </button>
            </div>

            {/* Preview Box */}
            {previewMode === "webcam" ? (
              <div className="flex flex-col gap-2">
                <div className="relative w-full aspect-[4/3] bg-black border border-[#222222] overflow-hidden flex items-center justify-center">
                  <video
                    ref={previewVideoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-full object-cover"
                  />

                  {!previewActive && (
                    <div className="absolute inset-0 bg-[#0A0A0A]/90 flex flex-col items-center justify-center gap-2 p-3 text-center">
                      <Video size={20} className="text-[#555555]" />
                      <span className="text-[11px] font-mono text-[#777777]">
                        {!activeTestingModelId
                          ? "Train your model first to test"
                          : "Preview is paused"}
                      </span>
                      <button
                        type="button"
                        onClick={startPreviewWebcam}
                        disabled={!activeTestingModelId}
                        className={`text-xs font-mono px-3 py-1 font-bold uppercase transition-colors ${activeTestingModelId
                          ? "bg-[#00E5FF] text-black hover:bg-[#00cbe2]"
                          : "bg-[#1C1C1C] text-[#555555] cursor-not-allowed"
                          }`}
                      >
                        START PREVIEW
                      </button>
                    </div>
                  )}

                  {previewActive && topPrediction && (
                    <div className="absolute top-2 left-2 bg-black/80 border border-[#00E5FF]/60 px-2 py-0.5 text-xs font-mono text-[#00E5FF] flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                      {topPrediction}
                    </div>
                  )}
                </div>

                {previewActive && (
                  <button
                    type="button"
                    onClick={stopPreviewWebcam}
                    className="text-[11px] font-mono py-1 bg-[#161616] hover:bg-[#202020] border border-[#2A2A2A] text-white transition-colors"
                  >
                    Pause Preview
                  </button>
                )}
              </div>
            ) : (
              <div className="flex flex-col gap-2">
                <div className="relative w-full aspect-[4/3] bg-black border border-[#222222] overflow-hidden flex items-center justify-center">
                  {testFilePreview ? (
                    <img
                      src={testFilePreview}
                      alt="test upload"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="text-center p-3 text-[11px] font-mono text-[#666666]">
                      Choose an image below to test
                    </div>
                  )}
                </div>

                <label
                  className={`text-xs font-mono py-1.5 border transition-colors text-center block ${activeTestingModelId
                    ? "bg-[#181818] hover:bg-[#222222] border-[#333333] hover:border-[#00E5FF] text-white cursor-pointer"
                    : "bg-[#141414] border-[#222222] text-[#555555] cursor-not-allowed"
                    }`}
                >
                  CHOOSE TEST IMAGE
                  <input
                    type="file"
                    accept="image/*"
                    disabled={!activeTestingModelId}
                    onChange={handleTestFileUpload}
                    className="hidden"
                  />
                </label>
              </div>
            )}

            {/* Confidence Probability Bars */}
            <div className="flex flex-col gap-1.5 pt-1.5 border-t border-[#1C1C1C]">
              {predictions.length > 0 ? (
                <div className="flex flex-col gap-1.5">
                  {predictions.map((p, pIdx) => {
                    const pct = Math.round(p.confidence * 100);
                    const isTop = pIdx === 0;

                    return (
                      <div key={p.class} className="flex flex-col gap-0.5">
                        <div className="flex justify-between text-[11px] font-mono">
                          <span className={isTop ? "text-white font-bold" : "text-[#777777]"}>
                            {p.class}
                          </span>
                          <span
                            className={
                              isTop ? "text-[#00E5FF] font-bold" : "text-[#666666]"
                            }
                          >
                            {pct}%
                          </span>
                        </div>
                        <div className="w-full h-1.5 bg-[#1A1A1A] overflow-hidden rounded-[1px]">
                          <div
                            className={`h-full transition-all duration-150 ${isTop ? "bg-[#00E5FF]" : "bg-[#3D3D3D]"
                              }`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-[10px] font-mono text-[#555555] py-1 text-center">
                  {!activeTestingModelId
                    ? "Train your classes on the left to test live."
                    : "Click 'Start Preview' to see live predictions."}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
