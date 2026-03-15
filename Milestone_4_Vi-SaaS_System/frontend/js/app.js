/**
 * Vi-SaaS - Shared JavaScript Utilities
 * Toast notifications, auth helpers, and shared utilities
 */

// ---- Toast Notifications ----
function showToast(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toastContainer');
  if (!container) return;
  const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.animation = 'slideToast 0.3s ease reverse';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ---- Auth Guard ----
function requireAuth() {
  if (!localStorage.getItem('visaas_token')) {
    window.location.href = '/pages/login.html';
    return false;
  }
  return true;
}

// ---- Format currency ----
function formatCurrency(val) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);
}

// ---- Scroll reveal ----
document.addEventListener('DOMContentLoaded', () => {
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.style.opacity = '1';
        e.target.style.animationPlayState = 'running';
      }
    });
  }, { threshold: 0.1 });
  document.querySelectorAll('.fade-in').forEach(el => {
    el.style.animationPlayState = 'paused';
    obs.observe(el);
  });
});

// ---- Logo Navigation Guard ----
function goHome(e, basePath = '') {
  e.preventDefault();
  if (localStorage.getItem('visaas_token')) {
    window.location.href = basePath + 'pages/dashboard.html';
  } else {
    window.location.href = basePath + 'index.html';
  }
}

