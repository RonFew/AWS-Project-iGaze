webgazer.setGazeListener((data, elapsedTime) => {
    // Change 'timestamp' to 'elapsedTime'
    if (data == null) { return; }
    console.log(data, elapsedTime);
}).begin();
