document.addEventListener('DOMContentLoaded', function () {
  const forms = document.querySelectorAll('form');

  forms.forEach(function (form) {
    form.addEventListener('submit', function () {
      const submitButton = form.querySelector('button[type="submit"]');

      if (!submitButton || submitButton.disabled) {
        return;
      }

      submitButton.dataset.originalText = submitButton.textContent;
      submitButton.innerHTML = '<span class="loading-dot" aria-hidden="true"></span>Working...';
      submitButton.disabled = true;
    });
  });
});
