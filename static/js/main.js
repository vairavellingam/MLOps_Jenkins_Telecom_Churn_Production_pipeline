// Animate probability bars on load
document.addEventListener('DOMContentLoaded', () => {
  // Smooth scroll to results if they exist
  const results = document.getElementById('results');
  if (results) {
    setTimeout(() => results.scrollIntoView({ behavior: 'smooth', block: 'start' }), 150);
  }

  // Loading state on form submit
  const form = document.getElementById('churnForm');
  const btn  = document.getElementById('predictBtn');
  if (form && btn) {
    form.addEventListener('submit', () => {
      btn.innerHTML = '<span class="btn-icon">⏳</span> Analysing...';
      btn.disabled = true;
      btn.style.opacity = '0.8';
    });
  }

  // Animate probability bars from 0 → actual width
  document.querySelectorAll('.prob-bar').forEach(bar => {
    const target = bar.style.width;
    bar.style.width = '0%';
    setTimeout(() => {
      bar.style.transition = 'width 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
      bar.style.width = target;
    }, 300);
  });
});
