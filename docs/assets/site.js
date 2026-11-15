/* Landing-page behavior. No dependencies, no network requests until the visitor presses play. */
(function () {
  "use strict";
  var cfg = (window.SITE_CONFIG || {});

  // ---- video slot: driven by VIDEO_URL in site-config.js; hidden while empty or invalid ----
  function youtubeId(u) {
    if (!u || typeof u !== "string") return null;
    u = u.trim();
    var m = u.match(/^https?:\/\/(?:www\.|m\.)?(?:youtube\.com\/(?:watch\?(?:.*&)?v=|embed\/|shorts\/|live\/)|youtu\.be\/|youtube-nocookie\.com\/embed\/)([A-Za-z0-9_-]{11})/);
    return m ? m[1] : null;
  }
  var id = youtubeId(cfg.VIDEO_URL);
  var toggles = document.querySelectorAll("[data-video-only]");
  var slot = document.getElementById("video-frame");
  if (id && slot) {
    for (var i = 0; i < toggles.length; i++) toggles[i].hidden = false;
    var btn = slot.querySelector("button");
    btn.addEventListener("click", function () {
      var f = document.createElement("iframe");
      f.src = "https://www.youtube-nocookie.com/embed/" + id + "?autoplay=1&rel=0&modestbranding=1";
      f.title = cfg.VIDEO_TITLE || "Project video";
      f.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; fullscreen";
      f.allowFullscreen = true;
      f.referrerPolicy = "strict-origin-when-cross-origin";
      slot.replaceChildren(f);
      f.focus();
    });
  }

  // ---- copy BibTeX ----
  var copy = document.getElementById("copy-bib");
  if (copy && navigator.clipboard) {
    copy.hidden = false;
    copy.addEventListener("click", function () {
      var text = document.getElementById("bibtex").textContent;
      navigator.clipboard.writeText(text).then(function () {
        copy.textContent = "Copied";
        setTimeout(function () { copy.textContent = "Copy BibTeX"; }, 1800);
      });
    });
  }
})();
