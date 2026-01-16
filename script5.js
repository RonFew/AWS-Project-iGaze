import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net";
        
let faceLandmarker;
let lastGaze = { x: 0, y: 0 };
let isBlinking = false;

async function setupMediaPipe() {
    // 1. Initialize Face Landmarker
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/wasm"
    );
    
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        outputFaceBlendshapes: true
    });

    // 2. Start WebGazer after Landmarker is ready
    initWebGazer();
}

function initWebGazer() {
    webgazer.setGazeListener((data, elapsedTime) => {
        if (data) {
            lastGaze = { x: data.x, y: data.y };
        }
    }).begin();

    // 3. Start the MediaPipe detection loop using WebGazer's video element
    const video = document.getElementById('webgazerVideoFeed');
    if (video) {
        detectBlink(video);
    }
}

async function detectBlink(video) {
    if (video.paused || video.ended) return;

    const startTimeMs = performance.now();
    const results = await faceLandmarker.detectForVideo(video, startTimeMs);

    if (results.faceBlendshapes && results.faceBlendshapes.length > 0) {
        const shapes = results.faceBlendshapes[0].categories;
        // eyeBlinkLeft and eyeBlinkRight values are usually between 0.0 and 1.0
        const blinkScore = (shapes.find(s => s.categoryName === "eyeBlinkLeft").score + 
                            shapes.find(s => s.categoryName === "eyeBlinkRight").score) / 2;

        if (blinkScore > 0.4) { // Threshold for blink
            if (!isBlinking) {
                isBlinking = true;
                handleBlinkClick();
            }
        } else {
            isBlinking = false;
        }
    }

    requestAnimationFrame(() => detectBlink(video));
}

function handleBlinkClick() {
    console.log("Blink Click at:", lastGaze.x, lastGaze.y);
    const element = document.elementFromPoint(lastGaze.x, lastGaze.y);
    if (element) element.click();
}

setupMediaPipe();
