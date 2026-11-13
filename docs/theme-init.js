// Runs before first paint (loaded synchronously from <head>): applies the saved or system theme to <html>
// so there is no flash. app.js owns the full logic (applyTheme); keep the storage key and rules in sync.
(function () {
  var mode = 'system';
  try { mode = localStorage.getItem('evalboard.themeMode') || 'system'; } catch (e) {}
  var dark = mode === 'dark' || (mode === 'system' && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
  if (mode === 'light' || mode === 'dark' || mode === 'system') document.documentElement.classList.toggle('dark', dark);
})();
