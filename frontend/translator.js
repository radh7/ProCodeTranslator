console.log("translator.js loaded");
const section = document.getElementById("SourceSection")

document.addEventListener("DOMContentLoaded", () => {
  const button = document.getElementById("translate-btn");

  if (button) {
    button.addEventListener("click", async () => {
      console.log("Translate button clicked");
      const code = document.getElementById("input").value;

      const res = await fetch("/translate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ code }),
      });

      const data = await res.json();
      console.log("Response from server:", data);

      const output = document.getElementById("output");
      output.textContent = data.translated_code || data.error || "Something went wrong";

      // 🟡 Re-highlight using Prism.js
      Prism.highlightElement(output);
    });
  } else {
    console.error("Translate button not found!");
  }


function showSection() {
  section.style.display = 'inline-block';  // Make the "Get Started" button visible
  setTimeout(() => {
    section.classList.add('show');  // Trigger the fade-in effect
  }, 100);  // Delay the fade-in slightly
}

showSection();  // Start typing the first line when the page loads
});
