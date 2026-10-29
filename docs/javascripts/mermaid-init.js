(function () {
  function renderDiagrams() {
    if (typeof mermaid === "undefined") return;

    // Some SuperFences renderers wrap custom diagram content in pre/code tags.
    // Mermaid expects the source as direct text inside its .mermaid container.
    document.querySelectorAll("code.language-mermaid").forEach((code) => {
      const container = code.closest(".mermaid");
      if (container) {
        container.textContent = code.textContent.trim();
        return;
      }

      const pre = code.parentElement;
      const wrapper = pre && pre.parentElement && pre.parentElement.classList.contains("highlight")
        ? pre.parentElement
        : pre;
      if (!wrapper) return;
      const diagram = document.createElement("div");
      diagram.className = "mermaid";
      diagram.textContent = code.textContent.trim();
      wrapper.replaceWith(diagram);
    });

    mermaid.initialize({ startOnLoad: false, theme: "default", securityLevel: "strict" });
    if (typeof document$ !== "undefined") {
      mermaid.run({ querySelector: ".mermaid:not([data-processed])" });
    } else {
      mermaid.run({ querySelector: ".mermaid" });
    }
  }

  if (typeof document$ !== "undefined") document$.subscribe(renderDiagrams);
  else document.addEventListener("DOMContentLoaded", renderDiagrams);
})();
