document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".copy-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      const code = button.parentElement?.querySelector("code")?.innerText || "";
      try {
        await navigator.clipboard.writeText(code);
        const original = button.textContent;
        button.textContent = "Copied";
        setTimeout(() => { button.textContent = original || "Copy"; }, 1500);
      } catch {
        button.textContent = "Copy unavailable";
      }
    });
  });

  document.querySelectorAll("details.collapsible").forEach((details) => {
    details.addEventListener("toggle", () => {
      if (details.open) details.classList.add("visible");
    });
  });
});
