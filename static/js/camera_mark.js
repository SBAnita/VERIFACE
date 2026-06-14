// camera_mark.js
const startMarkBtn = document.getElementById("startMarkBtn");
const stopMarkBtn = document.getElementById("stopMarkBtn");
const markVideo = document.getElementById("markVideo");
const markStatus = document.getElementById("markStatus");
const recognizedList = document.getElementById("recognizedList");

let markStream = null;
let markInterval = null;
let recognizedIds = new Set();

startMarkBtn.addEventListener("click", async () => {
  startMarkBtn.disabled = true;
  stopMarkBtn.disabled = false;
  try {
    markStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    markVideo.srcObject = markStream;
    await markVideo.play();
    markStatus.innerText = "Scanning...";
    markInterval = setInterval(captureAndRecognize, 1200);
  } catch (err) {
    alert("Camera error: " + err.message);
    startMarkBtn.disabled = false;
    stopMarkBtn.disabled = true;
  }
});

stopMarkBtn.addEventListener("click", () => {
  if (markInterval) clearInterval(markInterval);
  if (markStream) markStream.getTracks().forEach(t => t.stop());
  startMarkBtn.disabled = false;
  stopMarkBtn.disabled = true;
  markStatus.innerText = "Stopped";
});

async function captureAndRecognize() {

    const canvas = document.createElement("canvas");
    canvas.width = markVideo.videoWidth || 640;
    canvas.height = markVideo.videoHeight || 480;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(markVideo, 0, 0);

    const blob = await new Promise(resolve =>
        canvas.toBlob(resolve, "image/jpeg", 0.85)
    );

    const fd = new FormData();
    fd.append("image", blob, "snapshot.jpg");

    try {

        const res = await fetch("/recognize_face", {
            method: "POST",
            body: fd
        });

        const data = await res.json();

        if (data.recognized) {

            markStatus.innerHTML =
                `<span class="text-success">
                    ✓ ${data.name}
                 </span>`;

            if (!recognizedIds.has(data.student_id)) {

                recognizedIds.add(data.student_id);

                const li = document.createElement("li");

                li.className =
                    "list-group-item d-flex justify-content-between align-items-center";

                li.innerHTML = `
                    <span>${data.name}</span>
                    <span class="badge bg-success">
                        ${new Date().toLocaleTimeString()}
                    </span>
                `;

                recognizedList.prepend(li);

                // Update present count if element exists
                const countEl =
                    document.getElementById("todayPresentCount");

                if (countEl) {
                    countEl.innerText =
                        recognizedIds.size;
                }
            }

        } else {

            markStatus.innerHTML =
                `<span class="text-danger">
                    Not Recognized
                 </span>`;
        }

    } catch (err) {

        console.error(err);

        markStatus.innerHTML =
            `<span class="text-danger">
                Server Error
             </span>`;
    }
}
