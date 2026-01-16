// 1. Fixed Import: Specific version + ESM flag prevents redirects
import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net";

let faceLandmarker;
let lastGaze = { x: 0, y: 0 };
let isBlinking = false;

async function setupMediaPipe() {
    try {
        // 2. Fixed WASM Path: Points to the specific WASM binary folder
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net"
        );
        
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
    if (video.paused || video.ended) {
        requestAnimationFrame(() => detectBlink(video));
        return;
    }

    const results = await faceLandmarker.detectForVideo(video, performance.now());

    // 3. Corrected results navigation for faceBlendshapes
    if (results.faceBlendshapes && results.faceBlendshapes.length > 0) {
        const shapes = results.faceBlendshapes[0].categories;
        const left = shapes.find(s => s.categoryName === "eyeBlinkLeft").score;
        const right = shapes.find(s => s.categoryName === "eyeBlinkRight").score;
        
        if ((left + right) / 2 > 0.45) { // Sensitivity threshold
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
    // 4. Click any button or tile at the gaze coordinate
    const element = document.elementFromPoint(lastGaze.x, lastGaze.y);
    if (element) {
        const target = element.closest('.tile') || element.closest('button');
        if (target) {
            console.log("Blink Click Triggered:", target.getAttribute('data-word') || "button");
            target.click();
            
            // Visual feedback
            target.classList.add('gaze-pending');
            setTimeout(() => target.classList.remove('gaze-pending'), 400);
        }
    }
}

setupMediaPipe();
