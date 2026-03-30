// Image Preview
document.getElementById("file").addEventListener("change", function () {
  const file = this.files[0];
  const preview = document.getElementById("preview-image");
  const detectBtn = document.getElementById("detect-btn");
  const resultDiv = document.getElementById("result");

  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      preview.src = e.target.result;
      preview.style.display = "block";
      resultDiv.style.display = "none";
      detectBtn.classList.add("show-btn");
      detectBtn.classList.remove("hidden-btn");
    };
    reader.readAsDataURL(file);
  }
});

// Form Submit for Disease Detection
document.getElementById("upload-form").addEventListener("submit", async function (e) {
  e.preventDefault();

  const fileInput = document.getElementById("file");
  const resultDiv = document.getElementById("result");

  if (!fileInput.files.length) {
    alert("Please select an image first!");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  resultDiv.style.display = "block";
  resultDiv.textContent = "🔍 Detecting disease... please wait.";
  resultDiv.classList.remove("glow-healthy", "glow-diseased");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.error) {
      resultDiv.textContent = "❌ Error: " + data.error;
    } else {
      resultDiv.innerHTML = `
        🌿 <b>Predicted Class:</b> ${data.class}<br>
        🧠 <b>Confidence:</b> ${(data.confidence * 100).toFixed(2)}%
      `;

      if (data.class.toLowerCase().includes("healthy")) {
        resultDiv.classList.add("glow-healthy");
      } else {
        resultDiv.classList.add("glow-diseased");
      }
    }
  } catch (err) {
    resultDiv.textContent = "⚠️ Something went wrong!";
  }
});