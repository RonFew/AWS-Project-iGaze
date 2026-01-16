// 1. Fixed Import: Must point to the specific package on jsDelivr
import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net";

let faceLandmarker;
let lastGaze = { x: 0, y: 0 };
let isBlinking = false;

async function setupMediaPipe() {
    // 2. Fixed WASM Path: Must point to the specific WASM folder
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

    initWebGazer();
}

function initWebGazer() {
    // Check if webgazer is loaded via script tag in HTML
    if (typeof webgazer !== 'undefined') {
        webgazer.setGazeListener((data, elapsedTime) => {
            if (data) {
                lastGaze = { x: data.x, y: data.y };
            }
        }).begin();

        // Use a small delay to ensure WebGazer has created the video element
        setTimeout(() => {
            const video = document.getElementById('webgazerVideoFeed');
            if (video) detectBlink(video);
        }, 1000);
    }
}

async function detectBlink(video) {
    if (!faceLandmarker || video.paused || video.ended) {
        requestAnimationFrame(() => detectBlink(video));
        return;
    }

    const startTimeMs = performance.now();
    const results = await faceLandmarker.detectForVideo(video, startTimeMs);

    // 3. Fixed Data Access: results.faceBlendshapes is an array of objects
    if (results.faceBlendshapes && results.faceBlendshapes.length > 0) {
        const shapes = results.faceBlendshapes[0].categories;
        
        const leftBlink = shapes.find(s => s.categoryName === "eyeBlinkLeft")?.score || 0;
        const rightBlink = shapes.find(s => s.categoryName === "eyeBlinkRight")?.score || 0;
        const blinkScore = (leftBlink + rightBlink) / 2;

        if (blinkScore > 0.4) { 
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
    if (element && typeof element.click === 'function') {
        element.click();
    }
}

setupMediaPipe();
