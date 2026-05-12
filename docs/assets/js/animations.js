document.addEventListener("DOMContentLoaded", () => {
  const revealNodes = document.querySelectorAll(".reveal");
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });

  revealNodes.forEach((node) => observer.observe(node));

  document.querySelectorAll("[data-counter]").forEach((counter) => {
    const target = Number(counter.getAttribute("data-counter")) || 0;
    let value = 0;
    const step = Math.max(1, Math.floor(target / 50));
    const tick = () => {
      value += step;
      if (value >= target) {
        counter.textContent = String(target);
      } else {
        counter.textContent = String(value);
        requestAnimationFrame(tick);
      }
    };
    tick();
  });
});
