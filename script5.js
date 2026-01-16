// At the very top of script.js
import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net";

let faceLandmarker;
let lastGaze = { x: 0, y: 0 };
let isBlinking = false;

async function setupMediaPipe() {
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/wasm"
        );
        
        faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            outputFaceBlendshapes: true
        });

        console.log("MediaPipe Ready");
        initWebGazer();
    } catch (error) {
        console.error("MediaPipe initialization failed:", error);
    }
}

function initWebGazer() {
    // WebGazer is global, but we wait for it to be defined
    if (typeof webgazer === 'undefined') {
        setTimeout(initWebGazer, 100);
        return;
    }

    webgazer.setGazeListener((data) => {
        if (data) {
            lastGaze = { x: data.x, y: data.y };
        }
    }).begin();

    // Hide the WebGazer video feedback if you only want the dot
    webgazer.showVideoPreview(false).showPredictionPoints(true);

    // Wait for WebGazer to create its video element before attaching MediaPipe
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

    if (results.faceBlendshapes && results.faceBlendshapes.length > 0) {
        // Access the first face's categories
        const shapes = results.faceBlendshapes[0].categories;
        const left = shapes.find(s => s.categoryName === "eyeBlinkLeft").score;
        const right = shapes.find(s => s.categoryName === "eyeBlinkRight").score;
        
        if ((left + right) / 2 > 0.4) {
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
    const element = document.elementFromPoint(lastGaze.x, lastGaze.y);
    if (element && element.classList.contains('tile')) {
        console.log("Blink selected tile:", element.getAttribute('data-word'));
        element.click();
    }
}

setupMediaPipe();
