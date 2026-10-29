(function () {
  function attachScoreDemo() {
    document.querySelectorAll("[data-score-demo]").forEach((demo) => {
      if (demo.dataset.ready) return;
      demo.dataset.ready = "true";
      const input = demo.querySelector("[data-score-input]");
      const value = demo.querySelector("[data-score-value]");
      const label = demo.querySelector("[data-score-label]");
      const explanation = demo.querySelector("[data-score-explanation]");
      const update = () => {
        const score = Number(input.value);
        const useful = score >= 55;
        value.textContent = String(score);
        label.textContent = useful ? "useful" : "noise";
        label.style.background = useful ? "#00897b" : "#8e24aa";
        explanation.textContent = useful
          ? "Inference: the fixed rule maps this input to the useful class."
          : "Inference: the fixed rule maps this input to the noise class.";
      };
      input.addEventListener("input", update);
      update();
    });
  }

  if (typeof document$ !== "undefined") document$.subscribe(attachScoreDemo);
  else document.addEventListener("DOMContentLoaded", attachScoreDemo);
})();

