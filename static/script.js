// UI Elements
const tabUpload = document.getElementById('tab-upload');
const tabCamera = document.getElementById('tab-camera');
const modeUpload = document.getElementById('mode-upload');
const modeCamera = document.getElementById('mode-camera');

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewSection = document.getElementById('preview-section');
const imagePreview = document.getElementById('image-preview');
const analyzeBtn = document.getElementById('analyze-btn');
const resetBtn = document.getElementById('reset-btn');
const captureBtn = document.getElementById('capture-btn');
const newScanBtn = document.getElementById('new-scan-btn');

const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');

const loader = document.getElementById('loader');
const resultCard = document.getElementById('result-card');
const mainCard = document.querySelector('.main-card');

let currentFiles = [];
let stream = null;

// --- KNOWLEDGE BASE ---
const agriculturalData = {
    apple: { life: "2-4 Weeks", storage: "Refrigerated (1-3°C)", action: "Ready for Retail" },
    banana: { life: "2-5 Days", storage: "Room Temp (15-20°C)", action: "Immediate Sale" },
    bellpepper: { life: "1-2 Weeks", storage: "Refrigerated (7-10°C)", action: "Ready for Retail" },
    carrot: { life: "3-4 Weeks", storage: "Refrigerated (0-2°C)", action: "Ready for Retail" },
    cucumber: { life: "1 Week", storage: "Cool & Dry (10-12°C)", action: "Ready for Retail" },
    grape: { life: "1-2 Weeks", storage: "Refrigerated (-1 to 0°C)", action: "Ready for Retail" },
    guava: { life: "2-4 Days", storage: "Room Temp until ripe", action: "Immediate Sale" },
    jujube: { life: "1 Week", storage: "Refrigerated (0-4°C)", action: "Ready for Retail" },
    mango: { life: "1 Week", storage: "Room Temp until ripe", action: "Immediate Sale" },
    orange: { life: "2-3 Weeks", storage: "Refrigerated (0-4°C)", action: "Ready for Retail" },
    pomegranate: { life: "1-2 Months", storage: "Refrigerated (0-5°C)", action: "Long-term Storage" },
    potato: { life: "2-3 Months", storage: "Cool, Dark & Dry (7-10°C)", action: "Long-term Storage" },
    strawberry: { life: "3-5 Days", storage: "Refrigerated (0-2°C)", action: "Immediate Sale" },
    tomato: { life: "1 Week", storage: "Room Temp (Do not freeze)", action: "Ready for Retail" },
    default: { life: "Unknown", storage: "Standard Control", action: "Quality Evaluation Req." }
};

// --- TAB SWITCHING ---
tabUpload.addEventListener('click', () => {
    tabUpload.classList.add('active'); tabCamera.classList.remove('active');
    modeUpload.classList.remove('hidden'); modeCamera.classList.add('hidden');
    stopCamera();
});

tabCamera.addEventListener('click', () => {
    tabCamera.classList.add('active'); tabUpload.classList.remove('active');
    modeCamera.classList.remove('hidden'); modeUpload.classList.add('hidden');
    startCamera();
});

// --- CAMERA LOGIC ---
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        webcam.srcObject = stream;
    } catch (err) {
        alert("Camera access denied or unavailable.");
        tabUpload.click(); // revert
    }
}

function stopCamera() {
    if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null; }
}

captureBtn.addEventListener('click', () => {
    if (webcam.videoWidth === 0) {
        alert("Camera is still loading or blocked. Please wait a second.");
        return;
    }
    
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    canvas.getContext('2d').drawImage(webcam, 0, 0);
    
    canvas.toBlob((blob) => {
        if (!blob) {
            alert("Failed to capture image.");
            return;
        }
        currentFiles = [new File([blob], "scan.jpg", { type: "image/jpeg" })];
        runAnalysis();
    }, 'image/jpeg', 0.9);
});

// --- UPLOAD LOGIC ---
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files); });
fileInput.addEventListener('change', (e) => { if (e.target.files.length) handleFiles(e.target.files); });

function handleFiles(files) {
    currentFiles = Array.from(files).filter(f => f.type.startsWith('image/'));
    if (currentFiles.length === 0) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        dropZone.classList.add('hidden');
        previewSection.classList.remove('hidden');
        
        if (currentFiles.length > 1) {
            analyzeBtn.innerHTML = `<i class="ri-ai-generate"></i> Analyze Batch (${currentFiles.length} Angles)`;
        } else {
            analyzeBtn.innerHTML = `<i class="ri-ai-generate"></i> Run AI Analysis`;
        }
    };
    reader.readAsDataURL(currentFiles[0]);
}

resetBtn.addEventListener('click', () => {
    currentFiles = []; fileInput.value = '';
    previewSection.classList.add('hidden'); dropZone.classList.remove('hidden');
});

analyzeBtn.addEventListener('click', runAnalysis);

// --- API COMMUNICATION ---
async function runAnalysis() {
    if (currentFiles.length === 0) return;

    analyzeBtn.disabled = true;
    captureBtn.disabled = true;
    loader.classList.remove('hidden');
    resultCard.classList.add('hidden');
    mainCard.classList.add('hidden');

    let overallCondition = "Fresh";
    let finalFruit = "Unknown Object";
    let maxConf = 0;
    let combinedData = null;

    try {
        for (let i = 0; i < currentFiles.length; i++) {
            const formData = new FormData();
            formData.append('file', currentFiles[i]);
            
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const data = await response.json();
            
            // If ANY angle is rotten, the whole produce is marked rotten
            if (data.condition === "Rotten") {
                overallCondition = "Rotten";
            }
            
            if (data.fruit !== "Unknown Object" && data.fruit_confidence > maxConf) {
                finalFruit = data.fruit;
                maxConf = data.fruit_confidence;
            }
            combinedData = data;
        }
        
        // Synthesize results
        combinedData.fruit = finalFruit;
        combinedData.condition = overallCondition;
        if (maxConf > 0) combinedData.fruit_confidence = maxConf;

        displayDashboard(combinedData);
    } catch (error) {
        alert('API Error. Make sure the Python backend is running.');
        mainCard.classList.remove('hidden');
    } finally {
        analyzeBtn.disabled = false;
        captureBtn.disabled = false;
        loader.classList.add('hidden');
    }
}

// --- VOICE ASSISTANT ---
function speak(text) {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
    }
}

// --- DASHBOARD RENDERING ---
function displayDashboard(data) {
    resultCard.classList.remove('hidden');
    
    // AI Vision Metrics
    document.getElementById('res-fruit').textContent = data.fruit;
    setTimeout(() => {
        document.getElementById('fruit-conf-bar').style.width = `${data.fruit_confidence}%`;
        document.getElementById('fruit-conf-text').textContent = `${data.fruit_confidence.toFixed(1)}%`;
    }, 100);

    const condEl = document.getElementById('res-condition');
    const condBar = document.getElementById('cond-conf-bar');
    condEl.textContent = data.condition.toUpperCase();
    
    if (data.condition === "Fresh") {
        condEl.className = 'value status-fresh';
        condBar.style.background = 'var(--primary)';
    } else if (data.condition === "Rotten") {
        condEl.className = 'value status-rotten';
        condBar.style.background = 'var(--rotten)';
    } else {
        condEl.className = 'value';
        condEl.style.color = 'var(--text-muted)';
        condBar.style.background = 'var(--text-muted)';
    }

    setTimeout(() => {
        condBar.style.width = `${data.cond_confidence}%`;
        document.getElementById('cond-conf-text').textContent = `${data.cond_confidence.toFixed(1)}%`;
    }, 100);

    // Explainable AI Heatmap
    const heatmapImg = document.getElementById('heatmap-img');
    const heatmapLabel = document.getElementById('heatmap-label');
    if (data.heatmap) {
        heatmapImg.src = `data:image/jpeg;base64,${data.heatmap}`;
        heatmapImg.style.display = 'block';
        heatmapLabel.style.display = 'block';
    } else {
        heatmapImg.style.display = 'none';
        heatmapLabel.style.display = 'none';
    }

    // Agricultural Analytics Engine
    const fruitKey = data.fruit.toLowerCase();
    const knowledge = agriculturalData[fruitKey] || agriculturalData['default'];
    
    if (data.fruit === "Unknown Object") {
        document.getElementById('analytics-life').textContent = "N/A";
        document.getElementById('analytics-storage').textContent = "N/A";
        document.getElementById('analytics-action').textContent = "Scan valid produce to see analytics";
        document.getElementById('analytics-action').style.color = "var(--text-muted)";
        speak("Unknown object detected. Please scan valid produce.");
    } else {
        document.getElementById('analytics-life').textContent = data.condition === "Rotten" ? "Expired / 0 Days" : knowledge.life;
        document.getElementById('analytics-storage').textContent = knowledge.storage;
        
        if (data.condition === "Rotten") {
            document.getElementById('analytics-action').textContent = "Discard / Compost (Biowaste)";
            document.getElementById('analytics-action').style.color = "var(--rotten)";
        } else {
            document.getElementById('analytics-action').textContent = knowledge.action;
            document.getElementById('analytics-action').style.color = "var(--primary)";
        }
        
        speak(`${data.condition} ${data.fruit} detected. ${data.condition === "Rotten" ? "Discard immediately." : knowledge.action}.`);
    }
}

newScanBtn.addEventListener('click', () => {
    resultCard.classList.add('hidden');
    mainCard.classList.remove('hidden');
    if (tabUpload.classList.contains('active')) {
        resetBtn.click();
    }
});
