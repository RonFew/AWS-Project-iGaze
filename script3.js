let faceLandmarker;
let lastGaze = { x: 0, y: 0 };
let isBlinking = false;

// 1. Fixed WASM path: Must point to the npm package's WASM directory
async function setupMediaPipe() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net"
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

function initWebGazer() {
    webgazer.setGazeListener((data) => {
        if (data) lastGaze = { x: data.x, y: data.y };
    }).begin();
    
    // Start detection loop
    detectBlink();
}

async function detectBlink() {
    const video = document.getElementById('webgazerVideoFeed');
    
    // 2. Safety check: WebGazer may take a few seconds to create the video element
    if (video && video.readyState >= 2) {
        const result = faceLandmarker.detectForVideo(video, performance.now());
        
        if (result.faceBlendshapes && result.faceBlendshapes.length > 0) {
            const blendshapes = result.faceBlendshapes[0].categories;
            const leftBlink = blendshapes.find(b => b.categoryName === "eyeBlinkLeft").score;
            const rightBlink = blendshapes.find(b => b.categoryName === "eyeBlinkRight").score;

            // 3. Threshold: 0.4 is standard; increase to 0.5 if it's too sensitive
            if (leftBlink > 0.45 && rightBlink > 0.45) {
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
    const target = document.elementFromPoint(lastGaze.x, lastGaze.y);
    if (target && (target.tagName === 'BUTTON' || target.closest('button'))) {
        console.log("Blink Click:", target);
        target.click();
    }
}

setupMediaPipe();
