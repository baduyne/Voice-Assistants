document.addEventListener("DOMContentLoaded", () => {
    let mediaRecorder;
    let audioChunks = [];

    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
    const status = document.getElementById("status");
    const chatBox = document.getElementById("chatBox");

    // Helper to append a message to the chat
    function appendMessage(text, sender = "bot") {
        let formattedText = text
            .replace(/\*\*(.*?)\*\*/g, "\n- $1") 
            .replace(/\*(.*?)\*/g, "\n$1");     

        formattedText = formattedText.replace(/\n/g, "<br>");

        const msgDiv = document.createElement("div");
        msgDiv.innerHTML = formattedText; // dùng innerHTML để render xuống dòng
        msgDiv.className = sender === "user" ? "user-message" : "bot-message";
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight; // auto scroll
    }


    startBtn.onclick = async () => {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
        audioChunks = [];

        mediaRecorder.ondataavailable = e => {
            audioChunks.push(e.data);
        };

        mediaRecorder.onstart = () => {
            status.textContent = "Status: Recording...";
            startBtn.disabled = true;
            stopBtn.disabled = false;
        };

        mediaRecorder.start();
    };

    stopBtn.onclick = async () => {
        mediaRecorder.onstop = async () => {
            status.textContent = "Status: Sending audio...";

            const blob = new Blob(audioChunks, { type: "audio/webm" });
            const formData = new FormData();
            formData.append("file", blob, "recorded_audio.webm");

            try {
                const response = await fetch("/upload-audio/", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();
                if (data.text && data.text.trim()) {
                    appendMessage(data.text, "bot"); // display bot response
                }
            } catch (err) {
                console.error(err);
                appendMessage("Error sending audio or receiving response.", "bot");
            }

            status.textContent = "Status: Done";
            startBtn.disabled = false;
            stopBtn.disabled = true;
        };

        mediaRecorder.stop();
    };
});
