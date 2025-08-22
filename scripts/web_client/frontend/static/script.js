const chatBox = document.getElementById('chat-box');
const videoPanel = document.getElementById('video-panel');
const videoFrame = document.getElementById('video-frame');
const chatContainer = document.querySelector('.chat-container');
const messageForm = document.getElementById('message-form');
const messageInput = document.getElementById('message-text');
const settingsPanel = document.getElementById('settings-panel');
const settingsForm = document.getElementById('settings-form');
const settingsButton = document.getElementById('settings-button');
const closeVideoButton = document.getElementById('close-video-button');
const recordButton = document.getElementById('record-button');
const streamingCheckbox = document.getElementById('enable_streaming');
const streamingNote = document.getElementById('streaming-note');
const showVideoCheckbox = document.getElementById('show_video');
const settingsModal = document.getElementById('settings-modal');

// Establish WebSocket connection (protocol aware)
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws`);

let currentStreamingMessageElement = null;
let currentStreamingTimestamp = null;

ws.onopen = (event) => {
    console.log("WebSocket connection established");
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    const timestamp = getCurrentTime();

    // Check if the payload is not empty before displaying
    const hasContent = data.payload && data.payload.trim() !== '';

    if (data.type === 'text') {
        currentStreamingMessageElement = null;
        if (hasContent) { // <-- ADDED CHECK
            addMessage(data.payload, 'server-message', '🤖', timestamp);
        }
    } else if (data.type === 'stream') {
        // For the first chunk of a stream, only create a bubble if it has content
        if (!currentStreamingMessageElement && hasContent) {
            currentStreamingTimestamp = timestamp;
            const wrapper = addMessage('', 'server-message', '🤖', currentStreamingTimestamp);
            // grab the actual message div inside the wrapper
            currentStreamingMessageElement = wrapper.querySelector('.message');
        }
        // If a bubble already exists, append the content (even if it's empty)
        if (currentStreamingMessageElement) {
            currentStreamingMessageElement.textContent += data.payload;
        }

        if (data.done) {
            currentStreamingMessageElement = null;
            currentStreamingTimestamp = null;
        }
    } else if (data.type === 'audio') {
        currentStreamingMessageElement = null;
        const audioBlob = new Blob([new Uint8Array(atob(data.payload).split("").map(char => char.charCodeAt(0)))], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        addAudioMessage(audioUrl, 'server-message', '🤖', timestamp);
    } else if (data.type === 'error') {
        currentStreamingMessageElement = null;
        addErrorMessage(data.payload);
    } else if (data.type === 'video_stream_start') {
        addSuccessMessage("Video stream is available");
    } else if (data.type === 'video_stream_stop') {
        addErrorMessage("Video stream has stopped");
        videoFrame.src = ""; // Clear image
        // If the stream stops, hide the panel and uncheck the box
        videoPanel.style.display = 'none';
        showVideoCheckbox.checked = false;
    } else if (data.type === 'video_frame') {
        videoFrame.src = 'data:image/jpeg;base64,' + data.payload;
    }
};

ws.onerror = (error) => {
    console.error("WebSocket error: ", error);
    addMessage("Connection error. Please refresh the page.", 'server-message');
};

ws.onclose = () => {
    console.log("WebSocket connection closed");
    addErrorMessage("Connection closed. Please refresh the page.", 'server-message');
};

// Handle form submission
messageForm.addEventListener('submit', (event) => {
    console.log("Message form submitted: ", messageInput.value);
    event.preventDefault();
    const message = messageInput.value;
    if (message) {
        ws.send(JSON.stringify({ type: 'text', payload: message }));
        addMessage(message, 'user-message', 'You', getCurrentTime());
        messageInput.value = '';
    }
});

// --- NEW: Event listener for the "Show Video" checkbox ---
showVideoCheckbox.addEventListener('change', () => {
    if (showVideoCheckbox.checked) {
        // Show the video panel
        videoPanel.style.display = 'flex';
    } else {
        // Hide the video panel
        videoPanel.style.display = 'none';
    }
});

// Handle settings
settingsForm.addEventListener('submit', (event) => {
    event.preventDefault();
    const settings = {
        text_trigger: document.getElementById('text_trigger').value,
        text_target: document.getElementById('text_target').value,
        audio_trigger: document.getElementById('audio_trigger').value,
        audio_target: document.getElementById('audio_target').value,
        video_stream_topic: document.getElementById('video_stream_topic').value, // Add video topic
        enable_streaming: streamingCheckbox.checked
    };
    ws.send(JSON.stringify({ type: 'settings', payload: settings }));
    settingsModal.classList.remove('active');
    setTimeout(() => {
        settingsModal.style.display = 'none';
    }, 100); // match transition speed
    addSuccessMessage('Settings updated!');
});

// Show/hide the streaming note based on checkbox state
streamingCheckbox.addEventListener('change', () => {
    streamingNote.style.display = streamingCheckbox.checked ? 'block' : 'none';
});

// Open modal
settingsButton.addEventListener('click', () => {
    settingsModal.style.display = 'flex';
    setTimeout(() => {
        settingsModal.classList.add('active');
    }, 10); // tiny delay for transition
});

// Close modal when clicking outside panel
settingsModal.addEventListener('click', (e) => {
    if (!settingsPanel.contains(e.target)) {
        settingsModal.classList.remove('active');
        setTimeout(() => {
            settingsModal.style.display = 'none';
        }, 100); // match transition speed
    }
});

// Close video view
closeVideoButton.addEventListener('click', () => {
    videoPanel.style.display = 'none';
    showVideoCheckbox.checked = false;
});


// Audio Recording Logic
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let recordingIndicatorEl = null; // reference to the indicator message in DOM


recordButton.addEventListener("click", async () => {
    recordButton.classList.toggle("recording");

    if (isRecording) {
        mediaRecorder.stop();
        recordButton.innerHTML = `<i class="fa fa-microphone"></i>`;
    } else {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.onstart = () => {
                isRecording = true;
                recordButton.innerHTML = `<i class="fa fa-stop"></i> <span>Stop</span> <span class="record-tooltip">End Recording</span>`;

                // ✅ Add "Recording..." indicator to chat
                recordingIndicatorEl = addMessage("🎙 Recording...", "user-message recording-indicator", "You", getCurrentTime());
            };

            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                isRecording = false;

                const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
                const audioUrl = URL.createObjectURL(audioBlob);

                // ✅ Replace indicator with audio message
                if (recordingIndicatorEl) {
                    recordingIndicatorEl.remove(); // remove indicator bubble
                    recordingIndicatorEl = null;
                }
                addAudioMessage(audioUrl, "user-message", "You", getCurrentTime());

                const reader = new FileReader();
                reader.readAsDataURL(audioBlob);
                reader.onloadend = () => {
                    const base64Audio = reader.result.split(",")[1];
                    ws.send(JSON.stringify({ type: "audio", payload: base64Audio }));
                };
                recordButton.innerHTML = `<i class="fa fa-microphone"></i> <span class="record-tooltip">Record</span>`;
            };

            mediaRecorder.start();
        } catch (error) {
            console.error("Error accessing microphone:", error);
            addErrorMessage("Error: Could not access the microphone. Please grant permission.");
        }
    }
});

// Helper functions
function getCurrentTime() {
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    return `${hours}:${minutes}`;
}

function addMessage(text, type, label, date) {
    const wrapper = document.createElement("div");
    wrapper.className = "message-wrapper";
    if (type.includes("user-message")) {
        wrapper.style.alignSelf = "flex-end";
    } else {
        wrapper.style.alignSelf = "flex-start";
    }

    if (label) {
        const lbl = document.createElement("div");
        lbl.className = "message-label";
        lbl.textContent = label;
        wrapper.appendChild(lbl);
    }

    const msg = document.createElement("div");
    msg.className = `message ${type}`;
    msg.textContent = text;
    wrapper.appendChild(msg);

    if (date) {
        const timestamp = document.createElement("div");
        timestamp.className = `message-date-${type}`;
        timestamp.textContent = date;
        wrapper.appendChild(timestamp);
    }

    chatBox.appendChild(wrapper);
    chatBox.scrollTop = chatBox.scrollHeight;

    return wrapper;
}

function addAudioMessage(audioUrl, className, label, timestamp) {
    const wrapper = document.createElement('div');
    wrapper.className = `audio-message-wrapper-${className}`;

    const labelElement = document.createElement('div');
    labelElement.className = 'audio-message-label';
    labelElement.innerHTML = `<span>${label}</span><span>${timestamp}</span>`;

    const messageElement = document.createElement('div');
    messageElement.className = `audio-message`;

    const audioPlayer = document.createElement('audio');
    audioPlayer.controls = true;
    audioPlayer.src = audioUrl;
    audioPlayer.style.width = '100%';

    messageElement.appendChild(audioPlayer);
    wrapper.appendChild(labelElement);
    wrapper.appendChild(messageElement);
    chatBox.appendChild(wrapper);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function addErrorMessage(message) {
    const errorElement = document.createElement('div');
    errorElement.className = 'error-message';
    errorElement.textContent = message;
    chatBox.appendChild(errorElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function addSuccessMessage(message) {
    const errorElement = document.createElement('div');
    errorElement.className = 'success-message';
    errorElement.textContent = message;
    chatBox.appendChild(errorElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// --- Reset settings form on page load ---
window.addEventListener('DOMContentLoaded', () => {
    // Uncheck the checkboxes
    showVideoCheckbox.checked = false;
    streamingCheckbox.checked = false;

    // Hide the video panel
    videoPanel.style.display = 'none';

    // Reset selects in the form
    settingsForm.querySelectorAll('select').forEach(select => {
        select.selectedIndex = 0; // resets to the first option
    });
});
