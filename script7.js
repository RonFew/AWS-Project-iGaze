// 1. FIXED: Full path with version and ESM flag to prevent CORS/Redirect errors
import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net";

let faceLandmarker;
let lastGaze = { x: 0, y: 0 };
let isBlinking = false;

async function setupMediaPipe() {
    try {
        // 2. FIXED: Point to the specific WASM folder matching the library version
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net"
        );
        
        // 3. FIXED: Specific absolute path to the .task model file
        faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            outputFaceBlendshapes: true
        });

        console.log("MediaPipe initialized for 2026.");
        initWebGazer();
    } catch (error) {
        console.error("MediaPipe initialization failed:", error);
    }
}

function initWebGazer() {
    if (typeof webgazer === 'undefined') {
        setTimeout(initWebGazer, 100);
        return;
    }

    webgazer.setGazeListener((data) => {
        if (data) {
            lastGaze = { x: data.x, y: data.y };
        }
    }).begin();

    webgazer.showVideoPreview(false).showPredictionPoints(true);

    const checkVideo = setInterval(() => {
        const video = document.getElementById('webgazerVideoFeed');
        if (video) {
            clearInterval(checkVideo);
            detectBlink(video);
        }
    }, 500);
}

async function detectBlink(video) {
    if (!faceLandmarker || video.paused || video.ended) {
        requestAnimationFrame(() => detectBlink(video));
        return;
    }

    const results = await faceLandmarker.detectForVideo(video, performance.now());

    // 4. FIXED: results.faceBlendshapes[0].categories accesses the correct face data
    if (results.faceBlendshapes && results.faceBlendshapes.length > 0) {
        const shapes = results.faceBlendshapes[0].categories;
        const left = shapes.find(s => s.categoryName === "eyeBlinkLeft").score;
        const right = shapes.find(s => s.categoryName === "eyeBlinkRight").score;
        
        // Threshold: 0.45 is usually a "voluntary" blink
        if ((left + right) / 2 > 0.45) { 
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
    // 5. IMPLEMENTED: Click logic based on gaze position
    const element = document.elementFromPoint(lastGaze.x, lastGaze.y);
    if (element) {
        // Find the actual button/tile if look target is a child (like an icon)
        const target = element.closest('.tile') || element.closest('button');
        if (target) {
            console.log("Blink Click Triggered on:", target.innerText);
            target.click(); // Triggers your existing TTS/Message logic
            
            // Visual feedback using your gaze-pending class
            target.classList.add('gaze-pending');
            setTimeout(() => target.classList.remove('gaze-pending'), 400);
        }
    }
}

setupMediaPipe();
