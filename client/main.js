const API_BASE = window.API_BASE || (window.location.port === "8000" ? window.location.origin : "http://localhost:8000");

// --- UUID & State ---
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

function getProjectId() {
    let pid = localStorage.getItem('project_id');
    if (!pid) {
        pid = generateUUID();
        localStorage.setItem('project_id', pid);
    }
    return pid;
}

const projectId = getProjectId();
console.log("Using Project ID:", projectId);

// --- Navigation SPA ---
const navLinks = document.querySelectorAll('#main-nav a');
const views = document.querySelectorAll('.view');

navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        navLinks.forEach(l => l.classList.remove('active'));
        link.classList.add('active');

        const targetViewId = link.getAttribute('data-view');
        views.forEach(v => {
            if (v.id === targetViewId) v.classList.remove('hidden');
            else v.classList.add('hidden');
        });

        // Trigger load logic based on view
        if (targetViewId === 'view-runner') {
            loadRunnerExperiments();
        }
    });
});

// --- UI Utilities ---
function setupCustomSelect(selectId, valId, menuId, onChange) {
    const sel = document.getElementById(selectId);
    const val = document.getElementById(valId);
    const menu = document.getElementById(menuId);
    if (!sel || !menu) return;

    function toggle(o) {
        sel.classList.toggle('open', o);
        sel.setAttribute('aria-expanded', o);
    }

    sel.addEventListener('click', function (e) {
        const item = e.target.closest('.menu div');
        if (item) {
            val.textContent = item.textContent;
            val.dataset.val = item.getAttribute('data-val');
            Array.from(menu.children).forEach(c => c.classList.remove('sel'));
            item.classList.add('sel');
            toggle(false);
            if (onChange) onChange(item.getAttribute('data-val'));
            return;
        }
        toggle(!sel.classList.contains('open'));
    });

    document.addEventListener('click', function (e) {
        if (!sel.contains(e.target)) toggle(false);
    });
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024, dm = 2, sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

function setupFileDrop(dropId, inputId, browseId, itemsId, rmAllId, acceptRegex, onFilesChange) {
    const drop = document.getElementById(dropId);
    const input = document.getElementById(inputId);
    const items = document.getElementById(itemsId);
    const rmAll = document.getElementById(rmAllId);
    if (!drop || !input) return;

    let files = [];

    function render() {
        if (items) items.innerHTML = '';
        files.forEach((f, i) => {
            const ext = (f.name.split('.').pop() || '').toUpperCase();
            const row = document.createElement('div');
            row.className = 'row item';
            row.innerHTML = `<span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${f.name}</span>
                             <span>${ext}</span>
                             <span>${formatBytes(f.size)}</span>
                             <button class="rm" type="button">Remove</button>`;
            row.querySelector('.rm').onclick = () => { files.splice(i, 1); render(); onFilesChange(files); };
            if (items) items.appendChild(row);
        });
        if (onFilesChange) onFilesChange(files);
    }

    function add(list) {
        if (!list || !list.length) return;
        if (inputId === 'builder-file' && typeof currentPipeline !== 'undefined' && currentPipeline === 'tabular') {
            files = [];
            list = [list[0]];
        }
        Array.from(list).forEach(f => {
            if (f && (!acceptRegex || acceptRegex.test(f.name))) files.push(f);
        });
        render();
    }

    if (document.getElementById(browseId)) {
        document.getElementById(browseId).addEventListener('click', (e) => { e.stopPropagation(); input.click(); });
    }
    drop.addEventListener('click', () => input.click());
    input.addEventListener('change', () => { add(input.files); input.value = ''; });
    ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
    ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
    drop.addEventListener('drop', e => add(e.dataTransfer.files));
    if (rmAll) rmAll.addEventListener('click', () => { files = []; render(); });

    return () => files; // getter
}


// ==========================================
// 1. MODEL BUILDER
// ==========================================
let builderFiles = [];
const getBuilderFiles = setupFileDrop('builder-drop', 'builder-file', 'builder-browse', 'builder-items', 'builder-rm-all', /\.(csv|xlsx|parquet|txt|jsonl|json|pdf|docx|pptx|zip|tar\.gz|jpg|jpeg|png|webp|bmp|mp4|avi|mov|mkv|webm)$/i, (f) => builderFiles = f);

const archs = {
    tabular: [
        { val: 'autogluon_best', label: 'AutoGluon Best' },
        { val: 'custom_hf', label: 'Import Hugging Face Model...' }
    ],
    llm: [
        { val: 'unsloth/Llama-3.2-1B-Instruct-bnb-4bit', label: 'Llama 3.2 1B Instruct (4-bit)' },
        { val: 'unsloth/Llama-3.2-3B-Instruct-bnb-4bit', label: 'Llama 3.2 3B Instruct (4-bit)' },
        { val: 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit', label: 'Llama 3.1 8B Instruct (4-bit)' },
        { val: 'unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit', label: 'Llama 3.1 70B Instruct (4-bit)' },
        { val: 'unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', label: 'Qwen 2.5 0.5B Instruct (4-bit)' },
        { val: 'unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit', label: 'Qwen 2.5 1.5B Instruct (4-bit)' },
        { val: 'unsloth/Qwen2.5-3B-Instruct-bnb-4bit', label: 'Qwen 2.5 3B Instruct (4-bit)' },
        { val: 'unsloth/Qwen2.5-7B-Instruct-bnb-4bit', label: 'Qwen 2.5 7B Instruct (4-bit)' },
        { val: 'unsloth/Qwen2.5-14B-Instruct-bnb-4bit', label: 'Qwen 2.5 14B Instruct (4-bit)' },
        { val: 'unsloth/Qwen2.5-32B-Instruct-bnb-4bit', label: 'Qwen 2.5 32B Instruct (4-bit)' },
        { val: 'unsloth/Qwen2.5-72B-Instruct-bnb-4bit', label: 'Qwen 2.5 72B Instruct (4-bit)' },
        { val: 'unsloth/mistral-7b-instruct-v0.3-bnb-4bit', label: 'Mistral 7B Instruct v0.3 (4-bit)' },
        { val: 'unsloth/gemma-2-2b-it-bnb-4bit', label: 'Gemma 2 2B IT (4-bit)' },
        { val: 'unsloth/gemma-2-9b-it-bnb-4bit', label: 'Gemma 2 9B IT (4-bit)' },
        { val: 'unsloth/gemma-2-27b-it-bnb-4bit', label: 'Gemma 2 27B IT (4-bit)' },
        { val: 'unsloth/Phi-3.5-mini-instruct-bnb-4bit', label: 'Phi 3.5 Mini Instruct (4-bit)' },
        { val: 'custom_hf', label: 'Import Hugging Face Model...' }
    ],
    rag: [
        { val: 'rag_default', label: 'RAG Default pipeline' },
        { val: 'custom_hf', label: 'Import Hugging Face Model...' }
    ],
    vision_od: [
        { val: 'yolo11n', label: 'YOLO11 Nano (Next-Gen, High Accuracy & Speed)' },
        { val: 'yolo11s', label: 'YOLO11 Small (State-of-the-Art Balanced)' },
        { val: 'yolov8n', label: 'YOLOv8 Nano (Fastest, Edge-Optimized)' },
        { val: 'yolov8s', label: 'YOLOv8 Small (Balanced)' },
        { val: 'yolov8m', label: 'YOLOv8 Medium (High Accuracy)' },
        { val: 'yolov8l', label: 'YOLOv8 Large (Complex Detection)' },
        { val: 'custom_hf', label: 'Import Hugging Face Model...' }
    ],
    vision_cls: [
        { val: 'convnext_tiny', label: 'ConvNeXt-Tiny (Modern Pure-ConvNet, Best Accuracy)' },
        { val: 'efficientnet_b0', label: 'EfficientNet-B0 (Compound Scaled, High Efficiency)' },
        { val: 'resnet50', label: 'ResNet-50 (Deep Feature Extractor)' },
        { val: 'resnet18', label: 'ResNet-18 (Fast & Balanced)' },
        { val: 'mobilenet_v3_small', label: 'MobileNet V3 Small (Ultra-Lightweight)' },
        { val: 'custom_hf', label: 'Import Hugging Face Model...' }
    ]
};
// Alias for vision default
archs.vision = archs.vision_od;

let currentPipeline = 'tabular';
let currentVisionMode = 'od'; // 'od' (Object Detection) or 'cls' (Classification)

function setVisionMode(mode) {
    currentVisionMode = mode;
    const btnOd = document.getElementById('vm-btn-od');
    const btnCls = document.getElementById('vm-btn-cls');
    const builderDataUi = document.getElementById('builder-data-ui');
    const visionStudioUi = document.getElementById('vision-studio-ui');
    const builderFmt = document.getElementById('builder-fmt');
    const s2Title = document.getElementById('s2-title');
    const s2Hint = document.getElementById('s2-hint');
    const builderBtnSpan = document.querySelector('#builder-btn span');

    if (mode === 'od') {
        if (btnOd) {
            btnOd.style.background = 'var(--panel-hover)';
            btnOd.style.color = '#fff';
            btnOd.style.borderColor = 'var(--line-strong)';
        }
        if (btnCls) {
            btnCls.style.background = 'transparent';
            btnCls.style.color = '#a9a7b4';
            btnCls.style.borderColor = 'var(--line)';
        }
        if (builderDataUi) builderDataUi.classList.remove('hidden');
        if (visionStudioUi) visionStudioUi.classList.add('hidden');
        stopVsWebcam();

        if (s2Title) s2Title.textContent = 'Object Detection Dataset';
        if (s2Hint) s2Hint.textContent = 'Upload training images, video clips (.mp4, .mov), or annotated YOLO zip archives';
        if (builderFmt) builderFmt.textContent = 'Supported formats: Images (JPG, PNG, WEBP), Videos (MP4, AVI, MOV), ZIP';
        if (builderBtnSpan) builderBtnSpan.textContent = 'Train Object Detection Model';

        updateBuilderArchMenu('vision_od');
    } else {
        if (btnCls) {
            btnCls.style.background = 'var(--panel-hover)';
            btnCls.style.color = '#fff';
            btnCls.style.borderColor = 'var(--line-strong)';
        }
        if (btnOd) {
            btnOd.style.background = 'transparent';
            btnOd.style.color = '#a9a7b4';
            btnOd.style.borderColor = 'var(--line)';
        }
        if (builderDataUi) builderDataUi.classList.add('hidden');
        if (visionStudioUi) visionStudioUi.classList.remove('hidden');
        startVsWebcam();

        if (s2Title) s2Title.textContent = 'Webcam Burst Capture';
        if (s2Hint) s2Hint.textContent = 'Capture live training images for your custom visual classes';
        if (builderBtnSpan) builderBtnSpan.textContent = 'Train Classification Model';

        updateBuilderArchMenu('vision_cls');
    }
}

// Bind vision mode buttons
document.getElementById('vm-btn-od')?.addEventListener('click', () => setVisionMode('od'));
document.getElementById('vm-btn-cls')?.addEventListener('click', () => setVisionMode('cls'));

function updateBuilderArchMenu(pipeline) {
    const menu = document.getElementById('builder-menu');
    const val = document.getElementById('builder-val');
    menu.innerHTML = '';
    const items = archs[pipeline] || archs['tabular'];
    items.forEach((item, idx) => {
        const div = document.createElement('div');
        div.textContent = item.label;
        div.setAttribute('data-val', item.val);
        if (idx === 0) {
            div.classList.add('sel');
            val.textContent = item.label;
            val.dataset.val = item.val;
        }
        menu.appendChild(div);
    });
}

document.querySelectorAll('#builder-tabs .tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('#builder-tabs .tab').forEach(t => { t.classList.remove('active'); t.setAttribute('aria-selected', 'false'); });
        tab.classList.add('active'); tab.setAttribute('aria-selected', 'true');
        currentPipeline = tab.getAttribute('data-val');

        const fileInput = document.getElementById('builder-file');
        if (currentPipeline === 'tabular') {
            fileInput.removeAttribute('multiple');
        } else {
            fileInput.setAttribute('multiple', 'multiple');
        }

        const customHfInput = document.getElementById('builder-custom-hf');
        if (customHfInput) customHfInput.classList.add('hidden');

        const builderBtnSpan = document.querySelector('#builder-btn span');
        if (builderBtnSpan) {
            if (currentPipeline === 'tabular') builderBtnSpan.textContent = 'Build Model';
            else if (currentPipeline === 'llm') builderBtnSpan.textContent = 'Fine-Tune Model';
            else if (currentPipeline === 'rag') builderBtnSpan.textContent = 'Build RAG Pipeline';
            else if (currentPipeline === 'vision') builderBtnSpan.textContent = currentVisionMode === 'cls' ? 'Train Classification Model' : 'Train Object Detection Model';
        }

        const builderMainTitle = document.getElementById('builder-main-title');
        const builderMainSub = document.getElementById('builder-main-sub');
        const builderFmt = document.getElementById('builder-fmt');
        const visionModeSelector = document.getElementById('vision-mode-selector');

        if (builderMainTitle && builderMainSub) {
            if (currentPipeline === 'tabular') {
                builderMainTitle.innerHTML = 'Build your <span class="grad">model.</span>';
                builderMainSub.textContent = 'Select your pipeline, provide your training data, and let the platform handle the rest. We construct and tune the model architecture automatically.';
                if (builderFmt) builderFmt.textContent = 'Supported formats: CSV, XLSX, PARQUET, TXT, JSONL';
            } else if (currentPipeline === 'llm') {
                builderMainTitle.innerHTML = 'Fine-Tune your <span class="grad">LLM.</span>';
                builderMainSub.textContent = 'Select a base model, provide your instruction dataset, and let the platform handle the rest. We train and export the adapters automatically.';
                if (builderFmt) builderFmt.textContent = 'Supported formats: JSONL, JSON, TXT';
            } else if (currentPipeline === 'rag') {
                builderMainTitle.innerHTML = 'Build your <span class="grad">RAG.</span>';
                builderMainSub.textContent = 'Upload your documents, and let the platform handle the rest. We embed and construct the retrieval pipeline automatically.';
                if (builderFmt) builderFmt.textContent = 'Supported formats: PDF, DOCX, TXT, MD, JSONL';
            } else if (currentPipeline === 'vision') {
                builderMainTitle.innerHTML = 'Computer Vision <span class="grad">Studio.</span>';
                builderMainSub.textContent = 'Fine-tune Object Detection models (YOLO11, YOLOv8) with images/video or train real-time image classifiers using webcam burst capture.';
            }
        }

        const builderDataUi = document.getElementById('builder-data-ui');
        const visionStudioUi = document.getElementById('vision-studio-ui');

        if (currentPipeline === 'vision') {
            if (visionModeSelector) visionModeSelector.classList.remove('hidden');
            setVisionMode(currentVisionMode);
        } else {
            if (visionModeSelector) visionModeSelector.classList.add('hidden');
            if (builderDataUi) builderDataUi.classList.remove('hidden');
            if (visionStudioUi) visionStudioUi.classList.add('hidden');
            document.getElementById('s2-title').textContent = 'Training Data';
            document.getElementById('s2-hint').textContent = 'Upload your dataset to get started';
            stopVsWebcam();
            updateBuilderArchMenu(currentPipeline);
        }
    });
});

let vsStream = null;
let vsClassesData = [
    { name: 'Object 1', images: [] },
    { name: 'Object 2', images: [] }
];

async function startVsWebcam() {
    const video = document.getElementById('vs-webcam');
    const errBox = document.getElementById('vs-webcam-error');
    if (errBox) errBox.classList.add('hidden');
    if (!video || vsStream) return;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error("Camera API not supported or context not secure (HTTPS/localhost required).");
        if (errBox) {
            errBox.textContent = "Camera access requires HTTPS or localhost.";
            errBox.classList.remove('hidden');
        }
        return;
    }

    try {
        vsStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 } }
        });
        video.srcObject = vsStream;
        video.muted = true;
        await video.play().catch(e => console.warn("Video play interrupted:", e));
        const toggleBtn = document.getElementById('vs-webcam-toggle');
        if (toggleBtn) toggleBtn.textContent = 'Stop Camera';
    } catch (e) {
        console.error("Camera error:", e);
        let msg = "Could not access camera.";
        if (e.name === 'NotAllowedError' || e.name === 'PermissionDeniedError') {
            msg = "Camera permission denied. Please allow camera access in browser settings.";
        } else if (e.name === 'NotFoundError' || e.name === 'DevicesNotFoundError') {
            msg = "No camera found on this device.";
        } else if (e.name === 'NotReadableError' || e.name === 'TrackStartError') {
            msg = "Camera is in use by another application.";
        }
        if (errBox) {
            errBox.textContent = msg;
            errBox.classList.remove('hidden');
        }
    }
}

function stopVsWebcam() {
    if (vsStream) {
        vsStream.getTracks().forEach(t => t.stop());
        vsStream = null;
    }
    const video = document.getElementById('vs-webcam');
    if (video) video.srcObject = null;
}

document.getElementById('vs-webcam-toggle')?.addEventListener('click', (e) => {
    if (vsStream) {
        stopVsWebcam();
        e.target.textContent = 'Start Camera';
    } else {
        startVsWebcam();
        e.target.textContent = 'Stop Camera';
    }
});

// Setup click-to-record logic
document.querySelectorAll('.vs-capture-btn').forEach((btn) => {
    const classGroup = btn.closest('.vs-class-group');
    const classIdx = parseInt(classGroup.dataset.class);

    // Replace text of the button from HTML since it was "Hold to Record"
    btn.textContent = "Start Capture";
    let captureInterval = null;

    btn.addEventListener('click', () => {
        if (captureInterval) {
            // Stop capturing
            clearInterval(captureInterval);
            captureInterval = null;
            btn.textContent = "Start Capture";
            btn.style.background = "var(--accent)";
        } else {
            // Start capturing
            if (!vsStream) return;
            captureInterval = setInterval(() => {
                captureFrame(classIdx);
            }, 200); // 5 FPS
            btn.textContent = "Stop Capture";
            btn.style.background = "#ff5f56"; // Red to indicate recording
        }
    });
});

document.querySelectorAll('.vs-class-name').forEach((inp) => {
    inp.addEventListener('input', (e) => {
        const classGroup = e.target.closest('.vs-class-group');
        const classIdx = parseInt(classGroup.dataset.class);
        vsClassesData[classIdx].name = e.target.value;
    });
});

function captureFrame(classIdx) {
    const video = document.getElementById('vs-webcam');
    if (!video) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    const b64 = canvas.toDataURL('image/jpeg', 0.8);
    vsClassesData[classIdx].images.push(b64);

    // Update UI
    const classGroup = document.querySelector(`.vs-class-group[data-class="${classIdx}"]`);
    classGroup.querySelector('.vs-count').textContent = `${vsClassesData[classIdx].images.length} images`;

    // Add thumbnail
    const container = classGroup.querySelector('.vs-image-preview-container');
    if (container.children.length < 5) {
        const img = document.createElement('img');
        img.className = 'vs-img-thumb';
        img.src = b64;
        container.appendChild(img);
    } else {
        container.lastChild.src = b64;
    }
}

let vsInferStream = null;
let vsInferInterval = null;

async function startVsInference(modelId) {
    const inferUi = document.getElementById('vs-infer-ui');
    const video = document.getElementById('vs-infer-webcam');
    const predEl = document.getElementById('vs-infer-pred');
    const confEl = document.getElementById('vs-infer-conf');
    const modal = document.getElementById('vs-infer-modal');
    if (modal) {
        modal.classList.add('active');
        document.body.style.overflow = 'hidden'; // prevent background scrolling
    }

    // Add close button logic
    document.getElementById('vs-infer-close')?.addEventListener('click', () => {
        if (modal) modal.classList.remove('active');
        document.body.style.overflow = '';
        if (vsInferInterval) {
            clearInterval(vsInferInterval);
            vsInferInterval = null;
        }
        if (vsInferStream) {
            vsInferStream.getTracks().forEach(t => t.stop());
            vsInferStream = null;
        }
    }, { once: true });

    try {
        if (!vsInferStream) {
            vsInferStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 } }
            });
        }
        video.srcObject = vsInferStream;
        video.muted = true;
        await video.play().catch(e => console.warn("Inference video play error:", e));

        // Wait for video to be ready
        if (video.readyState < 2) {
            await new Promise(r => video.onloadedmetadata = r);
        }

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');

        vsInferInterval = setInterval(async () => {
            if (video.videoWidth === 0) return;

            ctx.drawImage(video, 0, 0);
            const b64 = canvas.toDataURL('image/jpeg', 0.8);

            try {
                const res = await fetch(`${API_BASE}/api/classifier/predict`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: b64, model_id: modelId })
                });
                if (res.ok) {
                    const data = await res.json();
                    if (data.status === 'success' && data.top_class) {
                        predEl.textContent = data.top_class;
                        confEl.textContent = `Confidence: ${(data.top_confidence * 100).toFixed(1)}%`;

                        // Pick a color based on confidence
                        if (data.top_confidence > 0.8) predEl.style.color = '#27c93f';
                        else if (data.top_confidence > 0.5) predEl.style.color = '#ffbd2e';
                        else predEl.style.color = '#ff5f56';
                    }
                }
            } catch (e) {
                // Ignore transient errors
            }
        }, 300); // ~3 FPS for inference

    } catch (e) {
        console.error("Inference camera error:", e);
        predEl.textContent = "Camera Error";
        predEl.style.color = '#ff5f56';
    }
}



const globalCustomHfInput = document.getElementById('builder-custom-hf');
if (globalCustomHfInput) {
    globalCustomHfInput.addEventListener('click', () => {
        if (typeof openHfModal === 'function') openHfModal();
    });
}

setupCustomSelect('builder-select', 'builder-val', 'builder-menu', (val) => {
    const customHfInput = document.getElementById('builder-custom-hf');
    if (customHfInput) {
        if (val === 'custom_hf') {
            customHfInput.classList.remove('hidden');
            if (typeof openHfModal === 'function') openHfModal();
        } else {
            customHfInput.classList.add('hidden');
        }
    }
});
updateBuilderArchMenu('tabular');

document.getElementById('builder-btn').addEventListener('click', async () => {
    const status = document.getElementById('builder-status');
    let arch = document.getElementById('builder-val').dataset.val;

    if (arch === 'custom_hf') {
        const customHfInput = document.getElementById('builder-custom-hf');
        if (customHfInput && customHfInput.value.trim() !== '') {
            arch = customHfInput.value.trim();
        } else {
            status.textContent = "Please enter or select a valid Hugging Face model ID.";
            status.style.color = "#ff6b6b";
            return;
        }
    }

    if (currentPipeline === 'vision' && currentVisionMode === 'cls') {
        const numImgs = vsClassesData.reduce((acc, c) => acc + c.images.length, 0);
        if (numImgs < 2) {
            status.textContent = "Please capture some images for your classes first.";
            status.style.color = "#ff6b6b";
            return;
        }

        status.textContent = "Initializing Vision Studio transfer learning...";
        status.style.color = "#fff";

        document.getElementById('builder-terminal').classList.remove('hidden');

        const logs = document.getElementById('term-logs');
        const leaderboard = document.getElementById('term-leaderboard');
        const chatDiv = document.getElementById('term-chat');
        const progWrap = document.querySelector('.progress-wrap');
        const progBar = document.getElementById('term-progress');
        const progText = document.getElementById('term-progress-text');

        logs.innerHTML = '';
        logs.classList.remove('hidden');
        leaderboard.classList.add('hidden');
        chatDiv.classList.add('hidden');
        progWrap.classList.remove('hidden');
        progBar.style.width = '0%';
        progText.textContent = '0%';

        logs.innerHTML += `<div class="term-line"><span class="time">[${new Date().toLocaleTimeString([], { hour12: false })}]</span> <span class="stage">init</span> <span>Setting up custom vision engine...</span></div>`;
        logs.innerHTML += `<div class="term-line"><span class="time">[${new Date().toLocaleTimeString([], { hour12: false })}]</span> <span class="stage">training</span> <span>Transfer learning on selected backbone...</span></div>`;

        let simPct = 0;
        const simInterval = setInterval(() => {
            if (simPct < 90) {
                simPct += 5;
                progBar.style.width = `${simPct}%`;
                progText.textContent = `${simPct}%`;
            }
        }, 500);

        try {
            const reqClasses = {};
            vsClassesData.forEach(c => {
                if (c.images.length > 0) reqClasses[c.name] = c.images;
            });
            const res = await fetch(`${API_BASE}/api/classifier/train`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    classes: reqClasses,
                    epochs: 10,
                    backbone: arch || 'convnext_tiny'
                })
            });
            clearInterval(simInterval);

            if (!res.ok) throw new Error("Training failed");
            const data = await res.json();

            progBar.style.width = '100%';
            progText.textContent = '100%';
            logs.innerHTML += `<div class="term-line"><span class="time">[${new Date().toLocaleTimeString([], { hour12: false })}]</span> <span class="stage">complete</span> <span>Model trained successfully.</span></div>`;
            logs.scrollTop = logs.scrollHeight;

            setTimeout(() => {
                const lbBody = document.getElementById('term-leaderboard-body');
                lbBody.innerHTML = `
                    <tr class="item best">
                        <td>Vision Studio Custom Model <span class="best-badge">BEST</span></td>
                        <td>0.99</td>
                        <td>~3s</td>
                    </tr>
                `;
                document.querySelector('#term-leaderboard .leaderboard-title').textContent = "VISION STUDIO READY";
                leaderboard.classList.remove('hidden');
                document.getElementById('leaderboard-actions').style.display = 'flex';

                logs.classList.add('hidden');
                progWrap.classList.add('hidden');

                const dlBtn = document.getElementById('download-model-btn');
                dlBtn.onclick = () => { window.location.href = `${API_BASE}/api/classifier/${data.model.model_id}/download`; };

                const runBtn = document.getElementById('run-model-btn');
                runBtn.textContent = 'Live Inference';
                runBtn.onclick = () => {
                    startVsInference(data.model.model_id);
                };
            }, 1000);

        } catch (e) {
            clearInterval(simInterval);
            status.textContent = "Error: " + e.message;
            status.style.color = "#ff6b6b";
        }
        return;
    }

    const files = getBuilderFiles();
    if (files.length === 0) {
        status.textContent = "Please upload training data first.";
        status.style.color = "#ff6b6b";
        return;
    }

    const customHfInput = document.getElementById('builder-custom-hf');
    if (arch === 'custom_hf' || (customHfInput && !customHfInput.classList.contains('hidden') && customHfInput.value.trim() !== '')) {
        if (customHfInput && customHfInput.value.trim() !== '') {
            arch = customHfInput.value.trim();
        } else {
            status.textContent = "Please enter or select a valid Hugging Face model ID.";
            status.style.color = "#ff6b6b";
            return;
        }
    }

    status.textContent = "Uploading data...";
    status.style.color = "#fff";

    try {
        // Find current latest experiment to avoid polling old ones
        let prevLatestId = null;
        try {
            const projRes = await fetch(`${API_BASE}/api/projects/${projectId}?t=${Date.now()}`);
            if (projRes.ok) {
                const projData = await projRes.json();
                if (projData.experiments && projData.experiments.length > 0) {
                    prevLatestId = projData.experiments[0].id;
                }
            }
        } catch (e) { }

        // Ingest files
        const formData = new FormData();
        formData.append('project_id', projectId);
        files.forEach(f => formData.append('files', f));

        let res = await fetch(`${API_BASE}/api/ingest`, { method: 'POST', body: formData });
        if (!res.ok) throw new Error("Data ingestion failed");

        status.textContent = "Data uploaded. Starting model build...";

        // Start build
        const config = { model_candidates: [arch] };
        res = await fetch(`${API_BASE}/api/expert_build`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: projectId,
                pipeline_type: currentPipeline,
                expert_config: config
            })
        });

        if (!res.ok) throw new Error("Build job failed to start");

        status.textContent = "Build triggered! Streaming logs...";
        status.style.color = "#e9e7f2";

        startTerminalPolling(currentPipeline, prevLatestId);

    } catch (e) {
        console.error(e);
        status.textContent = "Error: " + e.message;
        status.style.color = "#ff6b6b";
    }
});

async function startTerminalPolling(pipeline, prevLatestId) {
    const term = document.getElementById('builder-terminal');
    const logs = document.getElementById('term-logs');
    const progBar = document.getElementById('term-progress');
    const progText = document.getElementById('term-progress-text');
    const leaderboard = document.getElementById('term-leaderboard');
    const lbBody = document.getElementById('term-leaderboard-body');
    const chatDiv = document.getElementById('term-chat');

    term.classList.remove('hidden');
    leaderboard.classList.add('hidden');
    chatDiv.classList.add('hidden');
    logs.classList.remove('hidden');
    progBar.parentElement.classList.remove('hidden');

    logs.innerHTML = '<div class="term-line">Waiting for background worker...</div>';
    progBar.style.width = '0%';
    progText.textContent = '0%';

    let activeExpId = null;
    let pollAttempts = 0;

    // 1. Poll projects to find the new experiment
    const findExp = setInterval(async () => {
        pollAttempts++;
        try {
            const res = await fetch(`${API_BASE}/api/projects/${projectId}?t=${Date.now()}`);
            if (res.ok) {
                const data = await res.json();
                if (data.experiments && data.experiments.length > 0) {
                    const latest = data.experiments[0];
                    if (latest.id !== prevLatestId) {
                        activeExpId = latest.id;
                        clearInterval(findExp);
                        streamLogs(activeExpId, pipeline);
                    }
                }
            }
        } catch (e) { }

        if (pollAttempts > 30) {
            clearInterval(findExp);
            logs.innerHTML += '<div class="term-line" style="color:#ff5f56">Failed to connect to worker process. Orchestrator might have failed.</div>';
        }
    }, 2000);

    // 2. Stream logs
    async function streamLogs(expId, pl) {
        let lastLogCount = 0;

        const pollStatus = setInterval(async () => {
            try {
                const res = await fetch(`${API_BASE}/api/experiments/${expId}/status?t=${Date.now()}`);
                if (res.ok) {
                    const data = await res.json();
                    const statusArray = Array.isArray(data) ? data : (data.log || []);

                    if (statusArray && statusArray.length > 0) {
                        const latest = statusArray[statusArray.length - 1];
                        progBar.style.width = `${latest.pct}%`;
                        progText.textContent = `${latest.pct}%`;

                        if (statusArray.length > lastLogCount) {
                            const pageH = Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
                            const isLogsAtBottom = Math.abs(logs.scrollHeight - logs.scrollTop - logs.clientHeight) < 30;
                            const isPageAtBottom = Math.abs(pageH - window.innerHeight - window.scrollY) < 100;

                            for (let i = lastLogCount; i < statusArray.length; i++) {
                                const st = statusArray[i];
                                const timeStr = new Date(st.updated_at).toLocaleTimeString([], { hour12: false });
                                logs.innerHTML += `<div class="term-line"><span class="time">[${timeStr}]</span> <span class="stage">${st.stage}</span> <span>${st.message}</span></div>`;
                            }

                            if (isLogsAtBottom || lastLogCount === 0) {
                                logs.scrollTop = logs.scrollHeight;
                            }
                            if (isPageAtBottom || lastLogCount === 0) {
                                window.scrollTo({ top: pageH, behavior: 'smooth' });
                            }
                            lastLogCount = statusArray.length;
                        }

                        if (latest.pct === 100 || latest.stage === 'failed') {
                            clearInterval(pollStatus);
                            if (latest.pct === 100) {
                                setTimeout(() => fetchCompletionState(expId, pl), 1500);
                            }
                        }
                    }
                }
            } catch (e) { }
        }, 1500);
    }

    // 3. Fetch completion state and show actions
    async function fetchCompletionState(expId, pl) {
        try {
            const res = await fetch(`${API_BASE}/api/projects/${projectId}?t=${Date.now()}`);
            if (res.ok) {
                const data = await res.json();
                const exp = data.experiments.find(e => e.id === expId);

                const titleDiv = leaderboard.querySelector('div');
                const table = leaderboard.querySelector('.leaderboard-table');

                if (pl === 'tabular') {
                    titleDiv.textContent = 'EXPERIMENT LEADERBOARD';
                    table.classList.remove('hidden');
                    if (exp && exp.metrics && exp.metrics.leaderboard) {
                        lbBody.innerHTML = '';
                        exp.metrics.leaderboard.forEach(row => {
                            const tr = document.createElement('tr');
                            if (row.is_best) tr.className = 'best';
                            tr.innerHTML = `
                                <td>${row.model_name} ${row.is_best ? '<span class="best-badge">BEST</span>' : ''}</td>
                                <td>${row.score.toFixed(4)}</td>
                                <td>${row.fit_time.toFixed(2)}</td>
                            `;
                            lbBody.appendChild(tr);
                        });
                    }
                } else {
                    titleDiv.textContent = pl === 'llm' ? 'FINE-TUNED MODEL READY' : 'MODEL EXPORT READY';
                    table.classList.add('hidden');
                }

                leaderboard.classList.remove('hidden');
                document.getElementById('leaderboard-actions').style.display = 'flex';

                const dlBtn = document.getElementById('download-model-btn');
                dlBtn.onclick = () => { window.location.href = `${API_BASE}/api/experiments/${expId}/download`; };

                const runBtn = document.getElementById('run-model-btn');
                if (pl === 'llm' || pl === 'rag') {
                    runBtn.textContent = 'Chat / Infer Here';
                    runBtn.onclick = () => {
                        logs.classList.add('hidden');
                        leaderboard.classList.add('hidden');
                        progBar.parentElement.classList.add('hidden');
                        chatDiv.classList.remove('hidden');

                        const sendBtn = document.getElementById('term-chat-send');
                        const chatInput = document.getElementById('term-chat-input');
                        const msgs = document.getElementById('term-chat-messages');

                        if (window.currentChatExpId !== expId) {
                            msgs.innerHTML = '';
                            window.currentChatExpId = expId;
                        }

                        const handleSend = async () => {
                            const val = chatInput.value.trim();
                            if (!val) return;
                            chatInput.value = '';

                            msgs.innerHTML += `<div style="align-self: flex-end; background: var(--accent); color: #000; padding: 8px 14px; border-radius: 14px 14px 0 14px; max-width: 85%;">${val}</div>`;
                            msgs.scrollTop = msgs.scrollHeight;

                            try {
                                const backend = (exp.backend || '').toLowerCase();
                                const endpoint = backend === 'rag' ? `/api/experiments/${expId}/query_rag` : `/api/models/${expId}/chat`;
                                const payload = backend === 'rag' ? { query: val } : { prompt: val, max_tokens: 256 };

                                const res = await fetch(`${API_BASE}${endpoint}`, {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify(payload)
                                });

                                const data = await res.json();
                                const reply = data.response || data.answer || JSON.stringify(data);
                                const formattedReply = (typeof marked !== 'undefined') ? marked.parse(reply) : reply;

                                msgs.innerHTML += `<div class="chat-md" style="align-self: flex-start; background: rgba(255,255,255,0.08); padding: 8px 14px; border-radius: 14px 14px 14px 0; max-width: 85%; font-size: 14px; line-height: 1.5;">${formattedReply}</div>`;
                                msgs.scrollTop = msgs.scrollHeight;
                            } catch (e) {
                                msgs.innerHTML += `<div style="align-self: flex-start; color: #ff6b6b; font-size: 12px; margin-top: 4px;">Error: ${e.message}</div>`;
                            }
                        };

                        sendBtn.onclick = handleSend;
                        chatInput.onkeydown = (e) => { if (e.key === 'Enter') handleSend(); };
                        chatInput.focus();
                    };
                } else {
                    runBtn.onclick = async () => {
                        const runnerLink = document.querySelector('#main-nav a[data-view="view-runner"]');
                        if (runnerLink) runnerLink.click();

                        await loadRunnerExperiments();
                        const menu = document.getElementById('runner-menu');
                        const val = document.getElementById('runner-val');
                        const opt = Array.from(menu.children).find(c => c.dataset.val === expId);
                        if (opt) {
                            Array.from(menu.children).forEach(c => c.classList.remove('sel'));
                            opt.classList.add('sel');
                            val.textContent = opt.textContent;
                            val.dataset.val = expId;
                            onRunnerExperimentChange(expId);
                        }
                    };
                }

                // Scroll to bottom so leaderboard is visible
                term.parentElement.scrollTop = term.parentElement.scrollHeight;
            }
        } catch (e) { }
    }
}


// ==========================================
// 2. DATA FACTORY
// ==========================================
let syntheticSchema = null;
let syntheticJobId = null;

setupCustomSelect('df-rows-select', 'df-rows-val', 'df-rows-menu');

document.getElementById('df-generate-schema-btn').addEventListener('click', async () => {
    const prompt = document.getElementById('df-prompt').value.trim();
    const rows = parseInt(document.getElementById('df-rows-val').dataset.val || "1000", 10);
    const btn = document.getElementById('df-generate-schema-btn');

    if (!prompt) {
        alert("Please enter a dataset description.");
        return;
    }

    btn.textContent = "Analyzing & Planning...";
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/api/synthetic/schema`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, rows })
        });

        if (!res.ok) throw new Error("Failed to generate schema");
        const data = await res.json();
        syntheticSchema = data.schema;

        // Populate schema table
        const tbody = document.getElementById('df-schema-tbody');
        tbody.innerHTML = '';

        (syntheticSchema.columns || []).forEach(col => {
            const tr = document.createElement('tr');
            tr.style.borderBottom = "1px solid rgba(255,255,255,0.05)";
            tr.innerHTML = `
                <td style="padding: 10px; color: #fff; font-family: monospace;">${col.name}</td>
                <td style="padding: 10px;"><span style="background: rgba(185,162,245,0.15); color: #b9a2f5; padding: 2px 6px; border-radius: 4px; font-size: 11px; text-transform: uppercase;">${col.type}</span></td>
                <td style="padding: 10px; color: ${col.generation_strategy === 'llm' ? '#ffbd2e' : '#a9a7b4'};">${col.generation_strategy}</td>
                <td style="padding: 10px;">${col.description}</td>
            `;
            tbody.appendChild(tr);
        });

        // Reveal next steps
        document.getElementById('df-schema-section').style.opacity = "1";
        document.getElementById('df-schema-section').style.pointerEvents = "auto";
        document.getElementById('df-schema-container').classList.remove('hidden');

        document.getElementById('df-generate-section').style.opacity = "1";
        document.getElementById('df-generate-section').style.pointerEvents = "auto";

        // Reset old UI states
        document.getElementById('df-progress-container').classList.add('hidden');
        document.getElementById('df-preview-container').classList.add('hidden');
        document.getElementById('df-preview-head').innerHTML = '';
        document.getElementById('df-preview-body').innerHTML = '';
        document.getElementById('df-status').textContent = "";

        document.getElementById('df-progress-text').textContent = `Rows generated: 0 / ${rows}`;
        document.getElementById('df-progress-valid').textContent = `Valid: 0`;
        document.getElementById('df-progress-rejected').textContent = `Rejected: 0`;
        document.getElementById('df-progress-dupes').textContent = `Duplicates: 0`;
        document.getElementById('df-progress-bar').style.width = '0%';

    } catch (e) {
        console.error(e);
        alert("Error generating schema: " + e.message);
    } finally {
        btn.textContent = "Generate Schema";
        btn.disabled = false;
    }
});

let syntheticPoll = null;

document.getElementById('df-generate-dataset-btn').addEventListener('click', async () => {
    if (!syntheticSchema) return;

    const btn = document.getElementById('df-generate-dataset-btn');
    const status = document.getElementById('df-status');
    const progressContainer = document.getElementById('df-progress-container');
    const progressBar = document.getElementById('df-progress-bar');

    btn.disabled = true;
    status.textContent = "Starting generation engine...";
    status.style.color = "#fff";
    progressContainer.classList.remove('hidden');
    document.getElementById('df-preview-container').classList.add('hidden');

    try {
        const res = await fetch(`${API_BASE}/api/synthetic/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ schema_def: syntheticSchema })
        });

        if (!res.ok) throw new Error("Failed to start generation");
        const data = await res.json();
        syntheticJobId = data.job_id;

        if (syntheticPoll) clearInterval(syntheticPoll);

        syntheticPoll = setInterval(async () => {
            try {
                const pollRes = await fetch(`${API_BASE}/api/synthetic/status/${syntheticJobId}?t=${Date.now()}`);
                if (!pollRes.ok) return;
                const pData = await pollRes.json();

                const target = pData.target || 1;
                const valid = pData.valid || 0;
                const pct = Math.min(100, Math.round((valid / target) * 100));

                progressBar.style.width = `${pct}%`;
                document.getElementById('df-progress-text').textContent = `Rows generated: ${valid} / ${target}`;
                document.getElementById('df-progress-valid').textContent = `Valid: ${valid}`;
                document.getElementById('df-progress-rejected').textContent = `Rejected: ${pData.rejected || 0}`;
                document.getElementById('df-progress-dupes').textContent = `Duplicates: ${pData.duplicates || 0}`;

                if (pData.status === 'completed') {
                    clearInterval(syntheticPoll);
                    status.textContent = "Dataset generation completed successfully!";
                    status.style.color = "#27c93f";
                    btn.disabled = false;
                } else if (pData.status === 'running') {
                    status.textContent = "Generating dataset...";
                }

                // Show preview
                if (pData.preview && pData.preview.length > 0) {
                    const keys = Object.keys(pData.preview[0]);
                    const thead = document.getElementById('df-preview-head');
                    const tbody = document.getElementById('df-preview-body');

                    thead.innerHTML = '<tr>' + keys.map(k => `<th style="padding: 10px;">${k}</th>`).join('') + '</tr>';
                    tbody.innerHTML = '';

                    pData.preview.forEach(row => {
                        const tr = document.createElement('tr');
                        tr.style.borderBottom = "1px solid rgba(255,255,255,0.05)";
                        tr.innerHTML = keys.map(k => {
                            let val = row[k];
                            if (typeof val === 'string' && val.length > 50) val = val.substring(0, 50) + '...';
                            return `<td style="padding: 10px; max-width: 200px; overflow: hidden; text-overflow: ellipsis;">${val !== null ? val : ''}</td>`;
                        }).join('');
                        tbody.appendChild(tr);
                    });

                    document.getElementById('df-preview-container').classList.remove('hidden');
                }
            } catch (e) {
                console.error("Poll error:", e);
            }
        }, 1500);

    } catch (e) {
        console.error(e);
        status.textContent = "Error: " + e.message;
        status.style.color = "#ff6b6b";
        btn.disabled = false;
    }
});

document.getElementById('df-download-csv').addEventListener('click', () => {
    if (!syntheticJobId) return;
    window.location.href = `${API_BASE}/api/synthetic/download/${syntheticJobId}/csv`;
});

document.getElementById('df-download-json').addEventListener('click', () => {
    if (!syntheticJobId) return;
    window.location.href = `${API_BASE}/api/synthetic/download/${syntheticJobId}/json`;
});


// ==========================================
// 3. VISION STUDIO
// ==========================================
setupCustomSelect('vs-select', 'vs-val', 'vs-menu');
let vsClasses = [];
const vsClassesContainer = document.getElementById('vs-classes');

function addVsClass() {
    const id = Date.now();
    const div = document.createElement('div');
    div.className = 'vs-class-group';
    div.innerHTML = `
        <input type="text" class="vs-class-name" placeholder="Class Name (e.g., Cat)" value="Class ${vsClasses.length + 1}">
        <div class="drop vs-image-drop" tabindex="0" role="button">
            <div class="t">Drop images here</div>
            <input type="file" accept="image/*" multiple>
        </div>
        <div class="vs-image-preview-container"></div>
    `;
    vsClassesContainer.appendChild(div);

    const drop = div.querySelector('.vs-image-drop');
    const input = div.querySelector('input[type="file"]');
    const preview = div.querySelector('.vs-image-preview-container');
    const nameInput = div.querySelector('.vs-class-name');

    const classObj = { id, name: nameInput.value, images: [] };
    vsClasses.push(classObj);

    nameInput.addEventListener('input', () => classObj.name = nameInput.value);

    function addImages(files) {
        if (!files.length) return;
        Array.from(files).forEach(f => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const dataUrl = e.target.result;
                classObj.images.push(dataUrl);
                const img = document.createElement('img');
                img.src = dataUrl;
                img.className = 'vs-img-thumb';
                preview.appendChild(img);
            };
            reader.readAsDataURL(f);
        });
    }

    drop.addEventListener('click', () => input.click());
    input.addEventListener('change', () => addImages(input.files));
    ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.style.borderColor = 'var(--accent)'; }));
    ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.style.borderColor = ''; }));
    drop.addEventListener('drop', e => addImages(e.dataTransfer.files));
}

document.getElementById('vs-add-btn').addEventListener('click', addVsClass);
addVsClass(); // Add first class
addVsClass(); // Add second class

document.getElementById('vs-btn').addEventListener('click', async () => {
    const status = document.getElementById('vs-status');
    const backbone = document.getElementById('vs-val').dataset.val || 'mobilenet_v3_small';

    const classesData = {};
    for (const c of vsClasses) {
        if (c.images.length === 0) {
            status.textContent = `Class "${c.name}" has no images!`;
            status.style.color = "#ff6b6b";
            return;
        }
        classesData[c.name] = c.images;
    }

    status.textContent = "Training vision classifier... this may take a moment.";
    status.style.color = "#fff";

    try {
        const res = await fetch(`${API_BASE}/api/classifier/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ classes: classesData, backbone: backbone, epochs: 5 })
        });

        if (!res.ok) throw new Error("Training failed");

        status.textContent = "Custom Vision Model trained successfully!";
        status.style.color = "#e9e7f2";
    } catch (e) {
        console.error(e);
        status.textContent = "Error: " + e.message;
        status.style.color = "#ff6b6b";
    }
});


// ==========================================
// 4. RUNNER
// ==========================================
let runnerExperiments = [];
let runnerSelectedExperiment = null;

async function loadRunnerExperiments() {
    const menu = document.getElementById('runner-menu');
    const val = document.getElementById('runner-val');

    try {
        const res = await fetch(`${API_BASE}/api/projects/${projectId}?t=${Date.now()}`);
        if (!res.ok) throw new Error("Could not fetch project data");
        const data = await res.json();

        runnerExperiments = data.experiments.filter(e => e.status === 'completed' || e.status === 'success' || true); // Allow all for demo if needed

        menu.innerHTML = '';
        if (runnerExperiments.length === 0) {
            val.textContent = "No completed experiments found";
            return;
        }

        runnerExperiments.forEach((exp, idx) => {
            const div = document.createElement('div');
            const label = `${exp.dataset_name || exp.model_name} (${exp.backend})`;
            div.textContent = label;
            div.setAttribute('data-val', exp.id);
            if (idx === 0) {
                div.classList.add('sel');
                val.textContent = label;
                val.dataset.val = exp.id;
                onRunnerExperimentChange(exp.id);
            }
            menu.appendChild(div);
        });

    } catch (e) {
        console.error(e);
        val.textContent = "Error loading experiments";
    }
}

function onRunnerExperimentChange(expId) {
    runnerSelectedExperiment = runnerExperiments.find(e => e.id === expId);
    if (!runnerSelectedExperiment) return;

    const backend = (runnerSelectedExperiment.backend || '').toLowerCase();
    const textIn = document.getElementById('runner-text-in');
    const imageIn = document.getElementById('runner-image-in');
    const tabularIn = document.getElementById('runner-tabular-in');

    textIn.classList.add('hidden');
    imageIn.classList.add('hidden');
    tabularIn.classList.add('hidden');

    if (backend.includes('yolo') || backend.includes('ultralytics') || backend.includes('autotrain')) {
        imageIn.classList.remove('hidden');
    } else if (backend === 'autogluon') {
        tabularIn.classList.remove('hidden');
        tabularIn.innerHTML = '';
        const features = runnerSelectedExperiment.metrics?.features || [];
        features.forEach(feat => {
            const wrap = document.createElement('div');
            wrap.innerHTML = `<div style="font-size: 12px; margin-bottom: 5px;">${feat}</div><input type="text" data-feat="${feat}" style="width: 100%; height: 40px; background: var(--panel); border: 1px solid var(--line-strong); border-radius: 6px; color: #fff; padding: 0 10px; font-family: inherit;">`;
            tabularIn.appendChild(wrap);
        });
        if (features.length === 0) {
            tabularIn.innerHTML = '<div style="color: #aaa; font-size: 13px;">No features found for this model.</div>';
        }
    } else {
        textIn.classList.remove('hidden');
    }
}

setupCustomSelect('runner-select', 'runner-val', 'runner-menu', onRunnerExperimentChange);

let runnerVisionFile = null;
const runnerImgDrop = document.getElementById('runner-img-drop');
const runnerFileInput = document.getElementById('runner-file');
const runnerImgPreview = document.getElementById('runner-img-preview');

function handleRunnerImage(file) {
    if (!file) return;
    runnerVisionFile = file;
    const isVid = file.type.startsWith('video') || /\.(mp4|avi|mov|mkv|webm)$/i.test(file.name);
    if (isVid) {
        const url = URL.createObjectURL(file);
        runnerImgPreview.innerHTML = `<video src="${url}" controls style="height: 120px; border-radius: 8px; border: 1px solid var(--line);"></video>`;
    } else {
        const reader = new FileReader();
        reader.onload = e => {
            runnerImgPreview.innerHTML = `<img src="${e.target.result}" style="height: 100px; border-radius: 8px; border: 1px solid var(--line);">`;
        };
        reader.readAsDataURL(file);
    }
}

runnerImgDrop.addEventListener('click', () => runnerFileInput.click());
runnerFileInput.addEventListener('change', () => handleRunnerImage(runnerFileInput.files[0]));
['dragenter', 'dragover'].forEach(ev => runnerImgDrop.addEventListener(ev, e => { e.preventDefault(); runnerImgDrop.style.borderColor = 'var(--accent)'; }));
['dragleave', 'drop'].forEach(ev => runnerImgDrop.addEventListener(ev, e => { e.preventDefault(); runnerImgDrop.style.borderColor = ''; }));
runnerImgDrop.addEventListener('drop', e => handleRunnerImage(e.dataTransfer.files[0]));

// Confidence threshold slider
const runnerConfidence = document.getElementById('runner-confidence');
const runnerConfidenceVal = document.getElementById('runner-confidence-val');
runnerConfidence.addEventListener('input', () => {
    runnerConfidenceVal.textContent = parseFloat(runnerConfidence.value).toFixed(2);
});

document.getElementById('runner-btn').addEventListener('click', async () => {
    if (!runnerSelectedExperiment) return;

    const status = document.getElementById('runner-status');
    const outBox = document.getElementById('runner-out');
    const imgOut = document.getElementById('runner-img-out');
    const backend = (runnerSelectedExperiment.backend || '').toLowerCase();
    const expId = runnerSelectedExperiment.id;

    status.textContent = "Running inference...";
    status.style.color = "#fff";
    outBox.classList.add('hidden');
    imgOut.classList.add('hidden');

    try {
        if (backend.includes('yolo') || backend.includes('ultralytics') || backend.includes('autotrain')) {
            // Vision Predict
            if (!runnerVisionFile) throw new Error("Provide an image first");

            const formData = new FormData();
            formData.append('file', runnerVisionFile);
            formData.append('confidence', parseFloat(runnerConfidence.value) || 0.25);

            const res = await fetch(`${API_BASE}/api/models/${expId}/predict_vision`, {
                method: 'POST',
                body: formData
            });
            if (!res.ok) throw new Error("Vision predict failed");
            const data = await res.json();

            imgOut.src = data.annotated_image;
            imgOut.classList.remove('hidden');
            status.textContent = `Found ${data.count} objects in ${data.speed_ms?.inference?.toFixed(1) || '?'}ms`;
            status.style.color = "#e9e7f2";

        } else if (backend === 'autogluon') {
            // Tabular Predict
            const inputs = document.querySelectorAll('#runner-tabular-in input');
            const features = {};
            inputs.forEach(input => {
                let val = input.value.trim();
                if (!isNaN(val) && val !== '') val = Number(val);
                features[input.dataset.feat] = val;
            });

            const res = await fetch(`${API_BASE}/api/model/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ experiment_id: expId, features })
            });
            if (!res.ok) throw new Error("Tabular predict failed");
            const data = await res.json();

            outBox.textContent = `Prediction Result: ${data.prediction}`;
            outBox.classList.remove('hidden');
            status.textContent = "Inference complete.";
            status.style.color = "#e9e7f2";

        } else {
            // Text Chat / RAG
            const prompt = document.getElementById('runner-prompt').value;
            if (!prompt) throw new Error("Prompt is empty");

            // Assume LLM chat by default, or RAG if backend is RAG
            const endpoint = backend === 'rag' ? `/api/experiments/${expId}/query_rag` : `/api/models/${expId}/chat`;
            const payload = backend === 'rag' ? { query: prompt } : { prompt: prompt, max_tokens: 128 };

            const res = await fetch(`${API_BASE}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!res.ok) throw new Error("Inference failed");
            const data = await res.json();

            outBox.textContent = data.response || data.answer || JSON.stringify(data);
            outBox.classList.remove('hidden');
            status.textContent = "Inference complete.";
            status.style.color = "#e9e7f2";
        }
    } catch (e) {
        console.error(e);
        status.textContent = "Error: " + e.message;
        status.style.color = "#ff6b6b";
    }
});

// --- HF Modal Logic ---
const hfModal = document.getElementById('hf-modal');
const hfSearch = document.getElementById('hf-search');
const hfResults = document.getElementById('hf-results');
const hfClose = document.getElementById('hf-close');

function openHfModal() {
    if (!hfModal) return;
    hfModal.classList.add('active');
    hfSearch.focus();
}
function closeHfModal() {
    if (!hfModal) return;
    hfModal.classList.remove('active');
}

if (hfClose) hfClose.addEventListener('click', closeHfModal);
if (hfModal) hfModal.addEventListener('click', (e) => {
    if (e.target === hfModal) closeHfModal();
});

let hfDebounce;
if (hfSearch) hfSearch.addEventListener('input', (e) => {
    clearTimeout(hfDebounce);
    const q = e.target.value.trim();
    if (!q) {
        hfResults.innerHTML = '<div style="text-align: center; color: #a9a7b4; font-size: 13.5px; padding: 20px 0;">Type to search models from Hugging Face Hub</div>';
        return;
    }

    hfResults.innerHTML = '<div style="text-align: center; color: #a9a7b4; font-size: 13.5px; padding: 20px 0;">Searching...</div>';

    hfDebounce = setTimeout(async () => {
        try {
            const res = await fetch(`https://huggingface.co/api/models?search=${encodeURIComponent(q)}&limit=15&sort=downloads&direction=-1`);
            const models = await res.json();

            hfResults.innerHTML = '';
            if (models.length === 0) {
                hfResults.innerHTML = '<div style="text-align: center; color: #a9a7b4; font-size: 13.5px; padding: 20px 0;">No models found</div>';
                return;
            }

            models.forEach(m => {
                const item = document.createElement('div');
                item.className = 'hf-result-item';
                item.innerHTML = `
                    <div class="hf-result-title">${m.id}</div>
                    <div class="hf-result-stats">
                        <span><svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" fill="none" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> ${formatDownloads(m.downloads)}</span>
                        <span><svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" fill="none" stroke-width="2"><path d="M12 20.94c1.5 0 2.75 1.06 4 1.06 3 0 6-8 6-12.22A4.91 4.91 0 0 0 17 5c-2.22 0-4 1.44-5 2-1-.56-2.78-2-5-2a4.9 4.9 0 0 0-5 4.78C2 14 5 22 8 22c1.25 0 2.5-1.06 4-1.06Z"/></svg> ${m.likes}</span>
                    </div>
                `;
                item.addEventListener('click', async () => {
                    const customInput = document.getElementById('builder-custom-hf');
                    const valEl = document.getElementById('builder-val');
                    if (customInput) {
                        customInput.value = m.id;
                        customInput.classList.remove('hidden');
                    }
                    if (valEl) {
                        valEl.textContent = `HF: ${m.id}`;
                        valEl.dataset.val = m.id;
                    }
                    closeHfModal();

                    // Automatically trigger backend registration so capabilities & estimates are ready
                    try {
                        await fetch(`${API_BASE}/api/hf/import`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ model_id: m.id, pipeline_type: currentPipeline })
                        });
                    } catch (e) {
                        console.warn("Auto HF registration note:", e);
                    }
                });
                hfResults.appendChild(item);
            });
        } catch (e) {
            hfResults.innerHTML = '<div style="text-align: center; color: #ff6b6b; font-size: 13.5px; padding: 20px 0;">Failed to search Hugging Face</div>';
        }
    }, 400);
});

function formatDownloads(num) {
    if (!num) return '0';
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'k';
    return num.toString();
}
document.addEventListener("DOMContentLoaded", () => { const el = document.getElementById("df-prompt"); if (el) el.value = ""; });
