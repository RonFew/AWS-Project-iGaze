let faceLandmarker;
let lastGaze = { x: 0, y: 0 };
let isBlinking = false;

// 1. Fixed WASM path: Pointing to the specific WASM directory on jsDelivr
async function setupMediaPipe() {
    // UPDATED: Full path to the WASM folder is required for FilesetResolver
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
    );
    
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { 
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        outputFaceBlendshapes: true
    });
    
    // Start WebGazer AFTER Landmarker is ready to avoid camera lock
    initWebGazer();
}

// ... existing initWebGazer, detectBlink, and handleBlinkClick functions ...

setupMediaPipe();
