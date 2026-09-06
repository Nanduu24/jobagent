"""Browser driver abstraction.

SAFETY (structural): this interface has NO method that clicks a submit/apply
button. The only submit-related operation is ``locate_submit`` — it FINDS the
button so the human's attention can be drawn to it, and returns without clicking.
There is deliberately no click of any kind in the pre-fill flow.
"""
from __future__ import annotations

from typing import Protocol

from ..logging import get_logger
from .models import FieldType, FormField

log = get_logger(__name__)


class BrowserDriver(Protocol):
    def goto(self, url: str) -> None: ...
    def list_fields(self) -> list[FormField]: ...
    def fill_text(self, selector: str, value: str) -> None: ...
    def choose(self, selector: str, value: str) -> None: ...
    def attach_file(self, selector: str, path: str) -> None: ...
    def locate_submit(self) -> FormField | None: ...
    def screenshot(self, path: str) -> None: ...


# JS run in the page to introspect form controls + their labels. Read-only.
_INTROSPECT_JS = r"""
() => {
  const typeOf = (el) => {
    const t = (el.tagName || '').toLowerCase();
    if (t === 'textarea') return 'textarea';
    if (t === 'select') return 'select';
    const it = (el.getAttribute('type') || 'text').toLowerCase();
    return ['email','tel','file','radio','checkbox'].includes(it) ? it : 'text';
  };
  const labelFor = (el) => {
    if (el.id) { const l = document.querySelector(`label[for="${el.id}"]`); if (l) return l.innerText; }
    const wrap = el.closest('label'); if (wrap) return wrap.innerText;
    if (el.getAttribute('aria-label')) return el.getAttribute('aria-label');
    if (el.getAttribute('placeholder')) return el.getAttribute('placeholder');
    const fg = el.closest('.field, .form-field, [class*="field"]');
    if (fg) { const l = fg.querySelector('label'); if (l) return l.innerText; }
    return (el.name || '');
  };
  const sel = (el) => el.id ? `#${CSS.escape(el.id)}` :
                (el.name ? `${el.tagName.toLowerCase()}[name="${el.name}"]` : '');
  const out = [];
  const seen = new Set();
  document.querySelectorAll('input, select, textarea').forEach((el) => {
    const it = (el.getAttribute('type') || '').toLowerCase();
    if (['hidden','submit','button','reset','search'].includes(it)) return;
    const s = sel(el); if (!s || seen.has(s)) return; seen.add(s);
    let options = [];
    if (el.tagName.toLowerCase() === 'select')
      options = Array.from(el.options).map(o => o.text.trim()).filter(Boolean);
    out.push({ label: (labelFor(el) || '').trim().replace(/\s+/g,' ').slice(0,140),
               selector: s, type: typeOf(el), required: !!el.required, options });
  });
  return out;
}
"""

_SUBMIT_JS = r"""
() => {
  const btns = Array.from(document.querySelectorAll('button, input[type="submit"], [role="button"]'));
  const b = btns.find(x => /submit|apply|send application/i.test((x.innerText||x.value||'')));
  if (!b) return null;
  const r = b.getBoundingClientRect();
  return { label: (b.innerText||b.value||'Submit').trim().slice(0,60),
           x: Math.round(r.x+r.width/2), y: Math.round(r.y+r.height/2) };
}
"""


class PlaywrightDriver:
    """Headed Playwright driver. Never clicks submit (no such method)."""

    def __init__(self, page: object) -> None:
        self._page = page  # a playwright.sync_api.Page

    def goto(self, url: str) -> None:
        self._page.goto(url, wait_until="domcontentloaded", timeout=45000)  # type: ignore[attr-defined]

    def list_fields(self) -> list[FormField]:
        raw = self._page.evaluate(_INTROSPECT_JS)  # type: ignore[attr-defined]
        fields: list[FormField] = []
        for r in raw:
            fields.append(FormField(
                label=r["label"], selector=r["selector"],
                type=FieldType(r["type"]), required=r["required"],
                options=tuple(r.get("options") or ()),
            ))
        return fields

    def fill_text(self, selector: str, value: str) -> None:
        try:
            self._page.fill(selector, value, timeout=8000)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - a blocked/hidden field is skipped, logged
            log.warning("apply.fill_skipped", selector=selector)

    def choose(self, selector: str, value: str) -> None:
        try:
            self._page.select_option(selector, label=value, timeout=5000)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - radio/other: best-effort, skip on failure
            log.warning("apply.choose_skipped", selector=selector, value=value)

    def attach_file(self, selector: str, path: str) -> None:
        self._page.set_input_files(selector, path, timeout=15000)  # type: ignore[attr-defined]

    def locate_submit(self) -> FormField | None:
        found = self._page.evaluate(_SUBMIT_JS)  # type: ignore[attr-defined]
        if not found:
            return None
        # LOCATE ONLY — return the button as a field; we never click it.
        return FormField(label=found["label"], selector="(submit button)",
                         type=FieldType.other)

    def screenshot(self, path: str) -> None:
        self._page.screenshot(path=path, full_page=True)  # type: ignore[attr-defined]
