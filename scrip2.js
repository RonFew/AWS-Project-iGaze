let faceLandmarker;
let lastGaze = { x: 0, y: 0 };
let isBlinking = false;

// Initialize WebGazer
webgazer.setGazeListener((data) => {
    if (data) lastGaze = { x: data.x, y: data.y };
}).begin();

// Initialize MediaPipe Face Landmarker
async function setupMediaPipe() {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net");
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" },
        runningMode: "VIDEO",
        outputFaceBlendshapes: true // Simplifies blink detection via "eyeBlink" scores
    });
    detectBlink();
}

async function detectBlink() {
    const video = document.getElementById('webgazerVideoFeed');
    if (video && video.readyState >= 2) {
        const result = faceLandmarker.detectForVideo(video, performance.now());
        
        if (result.faceBlendshapes && result.faceBlendshapes.length > 0) {
            const blendshapes = result.faceBlendshapes[0].categories;
            // Extract blink scores (0.0 to 1.0)
            const leftBlink = blendshapes.find(b => b.categoryName === "eyeBlinkLeft").score;
            const rightBlink = blendshapes.find(b => b.categoryName === "eyeBlinkRight").score;

            // Threshold for a confirmed blink (typically > 0.4 for blendshapes)
            if (leftBlink > 0.4 && rightBlink > 0.4) {
                if (!isBlinking) {
                    isBlinking = true;
                    handleBlinkClick();
                }
            } else {
                isBlinking = false;
            }
        }
    }
    requestAnimationFrame(detectBlink);
}

function handleBlinkClick() {
    // Find the element at WebGazer's last known coordinates
    const target = document.elementFromPoint(lastGaze.x, lastGaze.y);
    if (target && target.tagName === 'BUTTON') {
        console.log("Blink detected! Clicking button:", target);
        target.click();
    }
}

setupMediaPipe();



/*
webgazer.setGazeListener((data, elapsedTime) => {
    // Change 'timestamp' to 'elapsedTime'
    if (data == null) { return; }
    console.log(data, elapsedTime);
}).begin();
*/
