webgazer.setGazeListener(async (data, elapsedTime) => {  
    if (data == null) return;  
  
    // --- 1. GET EYE DATA ---  
    const tracker = webgazer.getTracker();  
    const video = document.getElementById('webgazerVideoFeed');  
    const canvas = webgazer.getCanvas();  
    const patches = await tracker.getEyePatches(video, canvas, video.videoWidth, video.videoHeight);  
  
    if (patches && patches.left) {  
        const currentBrightness = calculateBrightness(patches.left.patch);  
  
        // --- 2. BLINK DETECTION IF STATEMENT ---  
        // We check if the brightness spiked (eyelid closed)  
        if (currentBrightness > lastBrightness * 1.15 && !isBlinking) {  
            isBlinking = true;  
  
            // --- 3. TRIGGER CLICK (Put Stage 3 Here) ---  
            // Use the data.x and data.y provided by the listener  
            triggerGlobalGazeClick(data.x, data.y);  
  
        } else if (currentBrightness < lastBrightness) {  
            isBlinking = false;  
        }  
        lastBrightness = currentBrightness;  
    }  
}).begin();  
  
// Define this function outside the listener to keep code clean  
function triggerGlobalGazeClick(x, y) {  
    const target = document.elementFromPoint(x, y);  
    const clickable = target?.closest("button, a");  
    if (clickable) {  
        clickable.click();  
        // Optional visual feedback  
        clickable.style.outline = "3px solid red";  
        setTimeout(() => clickable.style.outline = "", 200);  
    }  
}
