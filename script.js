const toggle = document.querySelector(".menu-toggle");
const nav = document.querySelector(".nav-links");

if (toggle && nav) {
  toggle.addEventListener("click", () => {
    const open = nav.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(open));
  });
}

// Collapsed technical-detail blocks in Models & Evaluation should appear in
// printed / PDF copies of the page, so open them all before printing.
window.addEventListener("beforeprint", () => {
  document.querySelectorAll("details").forEach((d) => { d.open = true; });
});
