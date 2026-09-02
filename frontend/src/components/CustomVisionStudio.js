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
  HelpCircle,
} from "lucide-react";

export default function CustomVisionStudio() {
  // Classes state: [{ id: "c1", name: "Class 1", samples: [b64, ...] }]
  const [classes, setClasses] = useState([
    { id: "c1", name: "Class 1", samples: [] },
    { id: "c2", name: "Class 2", samples: [] },
  ]);

  // Model & Hyperparameter state
  const [backbone, setBackbone] = useState("mobilenet_v3_small"); // "mobilenet_v3_small" | "resnet18"
  const [epochs, setEpochs] = useState(10);
  const [learningRate, setLearningRate] = useState(0.001);
  const [batchSize, setBatchSize] = useState(8);

  // Webcam capture state for training data collection
  const [activeCameraClassId, setActiveCameraClassId] = useState(null);
  const [isRecordingBurst, setIsRecordingBurst] = useState(false);
  const classVideoRef = useRef(null);
  const burstIntervalRef = useRef(null);
  const classStreamRef = useRef(null);

  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [trainProgress, setTrainProgress] = useState(0);
  const [trainStatusText, setTrainStatusText] = useState("");
  const [trainedModel, setTrainedModel] = useState(null);
  const [trainError, setTrainError] = useState(null);

  // Preview & Live Inference state
  const [previewMode, setPreviewMode] = useState("webcam"); // "webcam" | "file"
  const [previewActive, setPreviewActive] = useState(false);
  const previewVideoRef = useRef(null);
  const previewStreamRef = useRef(null);
  const previewLoopRef = useRef(null);
  const [testFilePreview, setTestFilePreview] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [topPrediction, setTopPrediction] = useState(null);
  const [inferenceLatency, setInferenceLatency] = useState(null);

  // Past models list
  const [pastModels, setPastModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState(null);

  // Fetch past models on mount (do NOT auto-select by default to prevent old test model confusion)
  const fetchModels = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/classifier/models");
      if (res.ok) {
        const data = await res.json();
        setPastModels(data.models || []);
      }
    } catch (e) {
      console.warn("Could not fetch models:", e);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  // -------------------------------------------------------------------------
  // Webcam Lifecycle for Training Samples
  // -------------------------------------------------------------------------
  const startClassWebcam = async (classId) => {
    stopClassWebcam();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240, facingMode: "user" },
      });
      classStreamRef.current = stream;
      setActiveCameraClassId(classId);
    } catch (err) {
      alert("Unable to access webcam: " + err.message);
    }
  };

  const stopClassWebcam = () => {
    if (burstIntervalRef.current) {
      clearInterval(burstIntervalRef.current);
      burstIntervalRef.current = null;
    }
    setIsRecordingBurst(false);
    if (classStreamRef.current) {
      classStreamRef.current.getTracks().forEach((t) => t.stop());
      classStreamRef.current = null;
    }
    setActiveCameraClassId(null);
  };

  useEffect(() => {
    if (activeCameraClassId && classVideoRef.current && classStreamRef.current) {
      classVideoRef.current.srcObject = classStreamRef.current;
    }
  }, [activeCameraClassId]);

  // Clean up streams on unmount
  useEffect(() => {
    return () => {
      stopClassWebcam();
      stopPreviewWebcam();
    };
  }, []);

  // Frame capture utility
  const captureFrameFromVideo = (videoEl) => {
    if (!videoEl || videoEl.readyState < 2) return null;
    const canvas = document.createElement("canvas");
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.85);
  };

  const captureSingleSnapshot = (classId) => {
    if (!classVideoRef.current) return;
    const frame = captureFrameFromVideo(classVideoRef.current);
    if (frame) {
      setClasses((prev) =>
        prev.map((c) =>
          c.id === classId ? { ...c, samples: [...c.samples, frame] } : c
        )
      );
    }
  };

  const startBurstRecording = (classId) => {
    if (!classVideoRef.current) return;
    setIsRecordingBurst(true);
    // Capture immediately
    captureSingleSnapshot(classId);
    // Then every 100ms
    burstIntervalRef.current = setInterval(() => {
      if (classVideoRef.current) {
        const frame = captureFrameFromVideo(classVideoRef.current);
        if (frame) {
          setClasses((prev) =>
            prev.map((c) =>
              c.id === classId ? { ...c, samples: [...c.samples, frame] } : c
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

  // -------------------------------------------------------------------------
  // File Upload for Training Samples
  // -------------------------------------------------------------------------
  const handleFileUpload = (classId, e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;

    files.forEach((file) => {
      const reader = new FileReader();
      reader.onload = (ev) => {
        const b64 = ev.target.result;
        setClasses((prev) =>
          prev.map((c) =>
            c.id === classId ? { ...c, samples: [...c.samples, b64] } : c
          )
        );
      };
      reader.readAsDataURL(file);
    });
    // Reset file input
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

  const clearClassSamples = (classId) => {
    setClasses((prev) =>
      prev.map((c) => (c.id === classId ? { ...c, samples: [] } : c))
    );
  };

  const addClass = () => {
    const nextNum = classes.length + 1;
    const newId = `c_${Date.now()}`;
    setClasses((prev) => [...prev, { id: newId, name: `Class ${nextNum}`, samples: [] }]);
  };

  const removeClass = (classId) => {
    if (classes.length <= 2) {
      alert("A classifier requires at least 2 classes to distinguish between objects.");
      return;
    }
    if (activeCameraClassId === classId) {
      stopClassWebcam();
    }
    setClasses((prev) => prev.filter((c) => c.id !== classId));
  };

  const updateClassName = (classId, newName) => {
    setClasses((prev) =>
      prev.map((c) => (c.id === classId ? { ...c, name: newName } : c))
    );
  };

  // -------------------------------------------------------------------------
  // Model Training (In-House PyTorch Transfer Learning)
  // -------------------------------------------------------------------------
  const totalSamplesCount = classes.reduce((acc, c) => acc + c.samples.length, 0);
  const emptyClasses = classes.filter((c) => c.samples.length === 0);
  const canTrain =
    classes.length >= 2 &&
    emptyClasses.length === 0 &&
    !isTraining;

  const handleTrainClick = () => {
    if (emptyClasses.length > 0) {
      alert(
        `Cannot start training yet:\n\nClass "${emptyClasses[0].name}" has 0 samples.\n\n` +
        `A machine learning classifier requires at least 2 distinct classes to learn the difference (for example: "${classes[0].name}" vs "Background / No ${classes[0].name}").\n\n` +
        `Please open the webcam on "${emptyClasses[0].name}" and record samples.`
      );
      return;
    }
    handleTrainModel();
  };

  const handleTrainModel = async () => {
    if (!canTrain) return;
    setIsTraining(true);
    setTrainError(null);
    setTrainProgress(15);
    setTrainStatusText("Structuring training tensors & data augmentations...");

    // Stop active camera during training to maximize GPU throughput
    stopClassWebcam();

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
      setTrainProgress(40);
      setTrainStatusText(`Training transfer head on ${backbone === "resnet18" ? "ResNet18" : "MobileNetV3-Small"}...`);

      const res = await fetch("http://localhost:8000/api/classifier/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      setTrainProgress(85);
      setTrainStatusText("Evaluating validation accuracy & exporting model checkpoint...");

      if (res.ok) {
        const data = await res.json();
        setTrainedModel(data.model);
        setSelectedModelId(data.model.model_id);
        setTrainProgress(100);
        setTrainStatusText("Training complete! Native PyTorch model ready for live inference.");

        // Refresh past models list
        fetchModels();

        // Automatically start preview with newly trained model
        setPreviewActive(true);
      } else {
        const err = await res.json().catch(() => ({}));
        setTrainError(err.detail || "Training failed. Please check your sample data.");
      }
    } catch (err) {
      setTrainError("Network error contacting classifier engine: " + err.message);
    } finally {
      setIsTraining(false);
    }
  };

  // -------------------------------------------------------------------------
  // Live Testing & Inference Lifecycle
  // -------------------------------------------------------------------------
  const activeTestingModelId = trainedModel ? trainedModel.model_id : selectedModelId;
  const activeModelIdRef = useRef(activeTestingModelId);
  activeModelIdRef.current = activeTestingModelId;

  const isPreviewActiveRef = useRef(previewActive);
  isPreviewActiveRef.current = previewActive;

  const startPreviewWebcam = async () => {
    if (!activeModelIdRef.current) {
      alert("Please train a model first using the classes on the left, or select a previously trained model.");
      return;
    }
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

  const stopPreviewWebcam = () => {
    if (previewLoopRef.current) {
      cancelAnimationFrame(previewLoopRef.current);
      previewLoopRef.current = null;
    }
    if (previewStreamRef.current) {
      previewStreamRef.current.getTracks().forEach((t) => t.stop());
      previewStreamRef.current = null;
    }
    setPreviewActive(false);
  };

  // Continuous Inference Loop for Webcam
  const runInferenceIteration = useCallback(async () => {
    const currentModelId = activeModelIdRef.current;
    if (!isPreviewActiveRef.current || !previewVideoRef.current || !currentModelId) return;

    const frame = captureFrameFromVideo(previewVideoRef.current);
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
        // Drop frame silently if busy
      }
    }

    if (isPreviewActiveRef.current && activeModelIdRef.current) {
      setTimeout(() => {
        previewLoopRef.current = requestAnimationFrame(runInferenceIteration);
      }, 100);
    }
  }, []);

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
  }, [previewActive, previewMode]);

  // File Preview Single Prediction
  const handleTestFileUpload = async (e) => {
    if (!activeTestingModelId) {
      alert("Please train a model first using your classes on the left.");
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
    <div className="flex flex-col gap-4 text-[#E0E0E0]">
      {/* Minimal Sleek Header */}
      <div className="flex items-center justify-between border-b border-[#222222] pb-3 flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-mono font-bold tracking-wider text-white flex items-center gap-2">
            <Layers className="text-[#00E5FF]" size={16} />
            CUSTOM VISION CLASSIFIER STUDIO
          </h2>
          <span className="text-[10px] font-mono px-2 py-0.5 bg-[#00E5FF]/10 text-[#00E5FF] border border-[#00E5FF]/30">
            PYTORCH TRANSFER
          </span>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono">
          <span className="text-[#777777]">
            Classes: <strong className="text-white">{classes.length}</strong>
          </span>
          <span className="text-[#777777]">
            Total Samples:{" "}
            <strong className={totalSamplesCount > 0 ? "text-[#00E5FF]" : "text-[#777777]"}>
              {totalSamplesCount}
            </strong>
          </span>
        </div>
      </div>

      {/* Main Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Data Collection Classes (7 cols) */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xs uppercase font-mono tracking-wider text-[#AAAAAA] flex items-center gap-2">
              <Camera size={14} className="text-[#00E5FF]" />
              1. DEFINE CLASSES & RECORD SAMPLES
            </h3>
            <button
              type="button"
              onClick={addClass}
              className="text-xs font-mono px-3 py-1 bg-[#1A1A1A] hover:bg-[#252525] border border-[#333333] hover:border-[#00E5FF] text-white transition-colors flex items-center gap-1.5"
            >
              <Plus size={13} /> ADD CLASS
            </button>
          </div>

          {/* Classes Cards */}
          <div className="flex flex-col gap-3">
            {classes.map((cls, idx) => {
              const isCameraActive = activeCameraClassId === cls.id;
              const hasNoSamples = cls.samples.length === 0;

              return (
                <div
                  key={cls.id}
                  className={`border transition-colors bg-[#111111] p-3.5 ${isCameraActive
                      ? "border-[#00E5FF]/60 shadow-[0_0_15px_rgba(0,229,255,0.08)]"
                      : hasNoSamples
                        ? "border-amber-500/30"
                        : "border-[#222222] hover:border-[#333333]"
                    }`}
                >
                  {/* Class Header */}
                  <div className="flex items-center justify-between mb-2.5 gap-2">
                    <div className="flex items-center gap-2 flex-1 flex-wrap">
                      <span className="font-mono text-xs text-[#555555] select-none">
                        #{idx + 1}
                      </span>
                      <input
                        type="text"
                        value={cls.name}
                        onChange={(e) => updateClassName(cls.id, e.target.value)}
                        className="bg-[#181818] border border-[#2D2D2D] focus:border-[#00E5FF] text-white px-2.5 py-1 text-sm font-semibold tracking-wide outline-none w-full max-w-[200px]"
                        placeholder="Class Name"
                      />
                      <span
                        className={`text-[11px] font-mono px-2 py-0.5 rounded-sm ${hasNoSamples
                            ? "bg-amber-950/40 text-amber-400 border border-amber-800/50"
                            : "bg-[#1E1E1E] text-[#AAAAAA]"
                          }`}
                      >
                        {cls.samples.length} sample{cls.samples.length !== 1 ? "s" : ""}
                        {hasNoSamples && " (Needs samples)"}
                      </span>
                    </div>

                    <div className="flex items-center gap-1">
                      {cls.samples.length > 0 && (
                        <button
                          type="button"
                          onClick={() => clearClassSamples(cls.id)}
                          title="Clear all samples for this class"
                          className="p-1.5 text-[#666666] hover:text-amber-400 transition-colors"
                        >
                          <RotateCcw size={13} />
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => removeClass(cls.id)}
                        title="Delete class"
                        className="p-1.5 text-[#666666] hover:text-red-400 transition-colors"
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                  </div>

                  {/* Capture Controls */}
                  <div className="flex flex-wrap items-center gap-2 mb-2.5">
                    {!isCameraActive ? (
                      <button
                        type="button"
                        onClick={() => startClassWebcam(cls.id)}
                        className={`text-xs font-mono px-3 py-1.5 border transition-colors flex items-center gap-1.5 ${hasNoSamples
                            ? "bg-amber-500/10 border-amber-500/60 text-amber-300 hover:bg-amber-500/20"
                            : "bg-[#181818] hover:bg-[#222222] border-[#333333] hover:border-[#00E5FF] text-white"
                          }`}
                      >
                        <Camera size={13} className={hasNoSamples ? "text-amber-400" : "text-[#00E5FF]"} />
                        OPEN WEBCAM
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={stopClassWebcam}
                        className="text-xs font-mono px-3 py-1.5 bg-red-950/30 hover:bg-red-900/40 border border-red-800 text-red-300 transition-colors flex items-center gap-1.5"
                      >
                        <VideoOff size={13} /> CLOSE WEBCAM
                      </button>
                    )}

                    <label className="text-xs font-mono px-3 py-1.5 bg-[#181818] hover:bg-[#222222] border border-[#333333] hover:border-[#888888] text-white transition-colors cursor-pointer flex items-center gap-1.5">
                      <Upload size={13} className="text-[#888888]" /> UPLOAD IMAGES
                      <input
                        type="file"
                        multiple
                        accept="image/*"
                        onChange={(e) => handleFileUpload(cls.id, e)}
                        className="hidden"
                      />
                    </label>
                  </div>

                  {/* Active Webcam Preview & Burst Capture */}
                  {isCameraActive && (
                    <div className="border border-[#00E5FF]/40 bg-black p-3 mb-2.5 flex flex-col md:flex-row gap-4 items-center">
                      <div className="relative w-48 h-36 bg-black border border-[#262626] overflow-hidden flex items-center justify-center">
                        <video
                          ref={classVideoRef}
                          autoPlay
                          playsInline
                          muted
                          className="w-full h-full object-cover"
                        />
                        {isRecordingBurst && (
                          <div className="absolute top-2 right-2 flex items-center gap-1.5 bg-red-600/90 text-white text-[10px] px-2 py-0.5 rounded font-mono animate-pulse">
                            <span className="w-1.5 h-1.5 rounded-full bg-white" />
                            BURST RECORDING
                          </div>
                        )}
                      </div>

                      <div className="flex flex-col gap-2 flex-1 w-full">
                        <div className="text-xs text-[#999999]">
                          Hold the button to rapidly collect frames as you move or rotate the object:
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onMouseDown={() => startBurstRecording(cls.id)}
                            onMouseUp={stopBurstRecording}
                            onTouchStart={() => startBurstRecording(cls.id)}
                            onTouchEnd={stopBurstRecording}
                            className={`flex-1 text-xs font-mono font-bold py-2.5 px-3 uppercase tracking-wider transition-all select-none flex items-center justify-center gap-2 ${isRecordingBurst
                                ? "bg-red-600 text-white shadow-[0_0_15px_rgba(220,38,38,0.5)] scale-[0.98]"
                                : "bg-[#00E5FF] hover:bg-[#00cbe2] text-black"
                              }`}
                          >
                            <Camera size={14} />
                            {isRecordingBurst ? "RECORDING SAMPLES..." : "HOLD TO RECORD SAMPLES"}
                          </button>
                          <button
                            type="button"
                            onClick={() => captureSingleSnapshot(cls.id)}
                            className="text-xs font-mono px-3 py-2.5 bg-[#1C1C1C] hover:bg-[#282828] border border-[#333333] text-white transition-colors whitespace-nowrap"
                          >
                            SNAPSHOT
                          </button>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Sample Gallery Grid */}
                  {cls.samples.length > 0 ? (
                    <div className="flex flex-wrap gap-1.5 max-h-28 overflow-y-auto p-1 bg-[#0A0A0A] border border-[#1C1C1C]">
                      {cls.samples.map((s, sIdx) => (
                        <div
                          key={sIdx}
                          className="group relative w-11 h-11 bg-[#181818] border border-[#2A2A2A] overflow-hidden shrink-0"
                        >
                          <img
                            src={s}
                            alt={`sample-${sIdx}`}
                            className="w-full h-full object-cover"
                          />
                          <button
                            type="button"
                            onClick={() => removeSample(cls.id, sIdx)}
                            className="absolute inset-0 bg-red-900/80 text-white opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity"
                          >
                            <Trash2 size={11} />
                          </button>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-[11px] font-mono text-amber-400/80 py-2 px-3 border border-dashed border-amber-800/40 bg-amber-950/10 text-center">
                      No samples yet. Open webcam above and hold the button to capture samples for {cls.name}.
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Column: Training Controls & Live Testing (5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-4">
          {/* Section 2: Train Model Panel */}
          <div className="border border-[#262626] bg-[#111111] p-4 flex flex-col gap-3.5">
            <h3 className="text-xs uppercase font-mono tracking-wider text-[#AAAAAA] flex items-center gap-2">
              <Sliders size={14} className="text-[#00E5FF]" />
              2. MODEL & TRAINING
            </h3>

            {/* Backbone Selection */}
            <div className="flex flex-col gap-1">
              <label className="text-[11px] uppercase font-mono text-[#888888] flex items-center gap-1.5">
                <Cpu size={12} className="text-[#00E5FF]" /> Backbone Architecture
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setBackbone("mobilenet_v3_small")}
                  className={`p-2 text-left border text-xs font-mono transition-colors ${backbone === "mobilenet_v3_small"
                      ? "border-[#00E5FF] bg-[#00E5FF]/10 text-white"
                      : "border-[#2A2A2A] bg-[#161616] text-[#777777] hover:border-[#444444]"
                    }`}
                >
                  <div className="font-bold text-white flex items-center justify-between">
                    MobileNetV3
                    {backbone === "mobilenet_v3_small" && (
                      <span className="w-1.5 h-1.5 rounded-full bg-[#00E5FF]" />
                    )}
                  </div>
                  <div className="text-[10px] text-[#888888] mt-0.5">Ultra Fast • ~9MB</div>
                </button>

                <button
                  type="button"
                  onClick={() => setBackbone("resnet18")}
                  className={`p-2 text-left border text-xs font-mono transition-colors ${backbone === "resnet18"
                      ? "border-[#00E5FF] bg-[#00E5FF]/10 text-white"
                      : "border-[#2A2A2A] bg-[#161616] text-[#777777] hover:border-[#444444]"
                    }`}
                >
                  <div className="font-bold text-white flex items-center justify-between">
                    ResNet18
                    {backbone === "resnet18" && (
                      <span className="w-1.5 h-1.5 rounded-full bg-[#00E5FF]" />
                    )}
                  </div>
                  <div className="text-[10px] text-[#888888] mt-0.5">Deep Residual • ~45MB</div>
                </button>
              </div>
            </div>

            {/* Hyperparameter Controls */}
            <div className="grid grid-cols-2 gap-3 pt-1 border-t border-[#1C1C1C]">
              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-[11px] font-mono">
                  <span className="text-[#888888]">Epochs:</span>
                  <span className="text-[#00E5FF] font-bold">{epochs}</span>
                </div>
                <input
                  type="range"
                  min={3}
                  max={30}
                  step={1}
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value))}
                  className="accent-[#00E5FF] cursor-pointer"
                />
              </div>

              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-[11px] font-mono">
                  <span className="text-[#888888]">Batch Size:</span>
                  <span className="text-[#00E5FF] font-bold">{batchSize}</span>
                </div>
                <select
                  value={batchSize}
                  onChange={(e) => setBatchSize(parseInt(e.target.value))}
                  className="bg-[#181818] border border-[#2D2D2D] text-white text-xs font-mono px-2 py-1 outline-none"
                >
                  <option value={4}>4 samples</option>
                  <option value={8}>8 samples</option>
                  <option value={16}>16 samples</option>
                </select>
              </div>
            </div>

            {/* Train Trigger Button */}
            <button
              type="button"
              onClick={handleTrainClick}
              className={`w-full py-3 px-4 font-mono text-xs uppercase tracking-wider font-bold transition-all flex items-center justify-center gap-2 ${canTrain
                  ? "bg-[#00E5FF] hover:bg-[#00cbe2] text-black shadow-[0_0_20px_rgba(0,229,255,0.25)] cursor-pointer"
                  : "bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 cursor-pointer"
                }`}
            >
              <Play size={14} fill={canTrain ? "black" : "currentColor"} />
              {isTraining
                ? "TRAINING PYTORCH MODEL..."
                : canTrain
                  ? "TRAIN CUSTOM CLASSIFIER"
                  : "START TRAINING (SAMPLES REQUIRED)"}
            </button>

            {/* Helpful validation reminder */}
            {emptyClasses.length > 0 && !isTraining && (
              <div className="p-2.5 bg-amber-950/20 border border-amber-800/40 text-amber-300 text-[11px] font-mono flex items-start gap-2">
                <AlertCircle size={14} className="shrink-0 mt-0.5 text-amber-400" />
                <span>
                  <strong>Step required:</strong> Record samples for <u>{emptyClasses.map((c) => c.name).join(", ")}</u>.
                  A classifier needs at least 2 classes to compare against (e.g. "{classes[0].name}" vs "No {classes[0].name}").
                </span>
              </div>
            )}

            {/* Training Progress Bar */}
            {isTraining && (
              <div className="flex flex-col gap-1.5 p-3 bg-[#0E0E0E] border border-[#222222]">
                <div className="flex justify-between text-[11px] font-mono">
                  <span className="text-[#888888]">{trainStatusText}</span>
                  <span className="text-[#00E5FF] font-bold">{trainProgress}%</span>
                </div>
                <div className="w-full h-1.5 bg-[#222222] overflow-hidden">
                  <div
                    className="h-full bg-[#00E5FF] transition-all duration-300"
                    style={{ width: `${trainProgress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Error Message */}
            {trainError && (
              <div className="p-3 bg-red-950/40 border border-red-800 text-red-300 text-xs font-mono flex items-start gap-2">
                <AlertCircle size={14} className="shrink-0 mt-0.5" />
                <span>{trainError}</span>
              </div>
            )}

            {/* Completed Model Card */}
            {trainedModel && (
              <div className="p-3 bg-emerald-950/20 border border-emerald-800/40 text-emerald-300 text-xs font-mono flex flex-col gap-2">
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-1.5 font-bold text-white">
                    <CheckCircle2 size={14} className="text-emerald-400" />
                    MODEL READY
                  </span>
                  <span className="text-[10px] text-[#888888]">{trainedModel.model_id}</span>
                </div>
                <div className="grid grid-cols-3 gap-2 text-[11px] border-t border-emerald-900/40 pt-2">
                  <div>
                    <span className="text-[#666666] block">VAL ACC</span>
                    <span className="text-white font-bold">
                      {(trainedModel.top1_accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-[#666666] block">TRAIN TIME</span>
                    <span className="text-white font-bold">{trainedModel.fit_time_seconds}s</span>
                  </div>
                  <div>
                    <span className="text-[#666666] block">BACKBONE</span>
                    <span className="text-white font-bold uppercase">{trainedModel.backbone}</span>
                  </div>
                </div>
                <a
                  href={`http://localhost:8000/api/classifier/${trainedModel.model_id}/download`}
                  download
                  className="mt-1 py-1 px-2.5 bg-[#1A1A1A] hover:bg-[#252525] border border-[#333333] hover:border-emerald-500 text-white text-[11px] flex items-center justify-center gap-1.5 transition-colors"
                >
                  <Download size={12} className="text-emerald-400" /> DOWNLOAD CHECKPOINT (.PTH)
                </a>
              </div>
            )}
          </div>

          {/* Section 3: Live Preview & Inference Studio */}
          <div className="border border-[#262626] bg-[#111111] p-4 flex flex-col gap-3.5">
            <div className="flex items-center justify-between">
              <h3 className="text-xs uppercase font-mono tracking-wider text-[#AAAAAA] flex items-center gap-2">
                <Sparkles size={14} className="text-[#00E5FF]" />
                3. LIVE TESTING & PREVIEW
              </h3>
              {inferenceLatency && previewActive && (
                <span className="text-[10px] font-mono px-2 py-0.5 bg-[#1A1A1A] border border-[#2D2D2D] text-[#00E5FF]">
                  {inferenceLatency} ms latency
                </span>
              )}
            </div>

            {/* Model Selector if past models exist */}
            {pastModels.length > 0 && (
              <div className="flex items-center justify-between gap-2 border-b border-[#1C1C1C] pb-2.5">
                <label className="text-[11px] font-mono text-[#888888]">Active Model:</label>
                <select
                  value={selectedModelId || (trainedModel ? trainedModel.model_id : "")}
                  onChange={(e) => {
                    setSelectedModelId(e.target.value);
                    setPredictions([]);
                    setTopPrediction(null);
                  }}
                  className="bg-[#181818] border border-[#2D2D2D] text-white text-xs font-mono px-2 py-1 outline-none max-w-[200px]"
                >
                  {pastModels.map((m) => (
                    <option key={m.model_id} value={m.model_id}>
                      {m.model_id} ({m.classes.join(", ")})
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Testing Mode Tabs */}
            <div className="flex border-b border-[#222222]">
              <button
                type="button"
                onClick={() => setPreviewMode("webcam")}
                className={`flex-1 py-2 text-xs font-mono uppercase tracking-wider text-center border-b-2 transition-colors flex items-center justify-center gap-1.5 ${previewMode === "webcam"
                    ? "border-[#00E5FF] text-[#00E5FF] bg-[#00E5FF]/5"
                    : "border-transparent text-[#777777] hover:text-white"
                  }`}
              >
                <Camera size={13} /> Live Camera
              </button>
              <button
                type="button"
                onClick={() => {
                  stopPreviewWebcam();
                  setPreviewMode("file");
                }}
                className={`flex-1 py-2 text-xs font-mono uppercase tracking-wider text-center border-b-2 transition-colors flex items-center justify-center gap-1.5 ${previewMode === "file"
                    ? "border-[#00E5FF] text-[#00E5FF] bg-[#00E5FF]/5"
                    : "border-transparent text-[#777777] hover:text-white"
                  }`}
              >
                <Upload size={13} /> File Test
              </button>
            </div>

            {/* Preview Viewport */}
            {previewMode === "webcam" ? (
              <div className="flex flex-col gap-2.5">
                <div className="relative w-full aspect-[4/3] bg-black border border-[#262626] overflow-hidden flex items-center justify-center">
                  <video
                    ref={previewVideoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-full object-cover"
                  />

                  {!previewActive && (
                    <div className="absolute inset-0 bg-[#0A0A0A]/90 flex flex-col items-center justify-center gap-2 p-4 text-center">
                      <Video size={24} className="text-[#555555]" />
                      <div className="text-xs font-mono text-[#888888]">
                        {!activeTestingModelId
                          ? "No model active yet. Train your classes first!"
                          : "Live camera testing is paused"}
                      </div>
                      <button
                        type="button"
                        onClick={startPreviewWebcam}
                        disabled={!activeTestingModelId}
                        className={`text-xs font-mono px-3 py-1.5 font-bold uppercase tracking-wider transition-colors ${activeTestingModelId
                            ? "bg-[#00E5FF] text-black hover:bg-[#00cbe2]"
                            : "bg-[#1E1E1E] text-[#555555] cursor-not-allowed"
                          }`}
                      >
                        START LIVE PREVIEW
                      </button>
                    </div>
                  )}

                  {previewActive && topPrediction && (
                    <div className="absolute top-2 left-2 bg-black/80 border border-[#00E5FF]/50 px-2.5 py-1 text-xs font-mono text-[#00E5FF] flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                      {topPrediction}
                    </div>
                  )}
                </div>

                {previewActive && (
                  <button
                    type="button"
                    onClick={stopPreviewWebcam}
                    className="text-xs font-mono py-1.5 bg-[#181818] hover:bg-[#222222] border border-[#333333] text-white transition-colors"
                  >
                    PAUSE LIVE PREVIEW
                  </button>
                )}
              </div>
            ) : (
              <div className="flex flex-col gap-2.5">
                <div className="relative w-full aspect-[4/3] bg-black border border-[#262626] overflow-hidden flex items-center justify-center">
                  {testFilePreview ? (
                    <img
                      src={testFilePreview}
                      alt="test upload preview"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="text-center p-4">
                      <Upload size={24} className="mx-auto text-[#444444] mb-2" />
                      <div className="text-xs font-mono text-[#777777]">
                        {!activeTestingModelId
                          ? "Train a model first to test images"
                          : "Upload an image to classify"}
                      </div>
                    </div>
                  )}
                </div>

                <label
                  className={`text-xs font-mono py-2 border transition-colors text-center block ${activeTestingModelId
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

            {/* Prediction Probability Bars */}
            <div className="flex flex-col gap-2 pt-2 border-t border-[#1C1C1C]">
              <div className="text-[11px] font-mono text-[#888888] uppercase tracking-wider">
                Classification Confidence
              </div>

              {predictions.length > 0 ? (
                <div className="flex flex-col gap-2">
                  {predictions.map((p, pIdx) => {
                    const pct = Math.round(p.confidence * 100);
                    const isTop = pIdx === 0;

                    return (
                      <div key={p.class} className="flex flex-col gap-1">
                        <div className="flex justify-between text-xs font-mono">
                          <span className={isTop ? "text-white font-bold" : "text-[#888888]"}>
                            {p.class}
                          </span>
                          <span
                            className={
                              isTop ? "text-[#00E5FF] font-bold" : "text-[#777777]"
                            }
                          >
                            {pct}%
                          </span>
                        </div>
                        <div className="w-full h-2 bg-[#1C1C1C] overflow-hidden rounded-[1px]">
                          <div
                            className={`h-full transition-all duration-150 ${isTop ? "bg-[#00E5FF]" : "bg-[#444444]"
                              }`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-[11px] font-mono text-[#555555] py-2 text-center">
                  {!activeTestingModelId
                    ? "Train your classes on the left to activate live predictions."
                    : "Click 'Start Live Preview' to see predictions."}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
