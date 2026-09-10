// Kontrast- und Layout-Prüfung der Seite (Playwright, Chromium).
// Jedes Text/Grund-Paar muss 4.5:1 erreichen, jede Schrift schwarz oder
// weiss sein, nichts darf über den Rand ragen, nichts waagrecht scrollen.
// Aufruf aus dem Repo:  node site/docs/pruefe-kontrast.mjs

import { chromium } from "/Users/benpohl/Claude/enrich-transcript/frontend/node_modules/playwright/index.mjs";
const b = await chromium.launch();
for (const w of [1440, 820, 390]) {
  const p = await b.newPage({ viewport: { width: w, height: 900 } });
  await p.goto("file:///Users/benpohl/Claude/enrich-transcript/site/index.html?lang=de", { waitUntil: "networkidle" });
  const r = await p.evaluate(() => {
    const lum = (c) => { const [r,g,b] = c.map(v => { v/=255; return v <= .03928 ? v/12.92 : ((v+.055)/1.055)**2.4; }); return .2126*r+.7152*g+.0722*b; };
    const parse = (s) => { const m = s.match(/rgba?\(([^)]+)\)/); if (!m) return null; const v = m[1].split(",").map(parseFloat); return { rgb: v.slice(0,3), a: v.length > 3 ? v[3] : 1 }; };
    const bgOf = (el) => { let e = el; while (e) { const c = parse(getComputedStyle(e).backgroundColor); if (c && c.a > 0) return c.rgb; e = e.parentElement; } return [255,255,255]; };
    const out = [], seen = new Set();
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let n;
    while ((n = walker.nextNode())) {
      if (!n.textContent.trim()) continue;
      const el = n.parentElement; if (!el || ["SCRIPT","STYLE"].includes(el.tagName)) continue;
      const cs = getComputedStyle(el); if (cs.display === "none" || cs.visibility === "hidden") continue;
      const fg = parse(cs.color); if (!fg) continue;
      const bg = bgOf(el);
      const L1 = lum(fg.rgb), L2 = lum(bg); const ratio = (Math.max(L1,L2)+.05)/(Math.min(L1,L2)+.05);
      const key = cs.color + "|" + bg.join(",") + "|" + el.tagName + "." + el.className;
      if (seen.has(key)) continue; seen.add(key);
      out.push({ ratio: +ratio.toFixed(2), fg: cs.color, bg: "rgb(" + bg.join(",") + ")", wo: el.tagName.toLowerCase() + (el.className ? "." + String(el.className).split(" ")[0] : ""), text: n.textContent.trim().slice(0, 40) });
    }
    // Layout: was ragt über den Rand, was überlappt
    const ueber = [...document.querySelectorAll("body *")].filter(e => { const r = e.getBoundingClientRect(); return r.width > 0 && (r.right > innerWidth + 1 || r.left < -1); }).map(e => e.tagName.toLowerCase() + "." + String(e.className).split(" ")[0]).slice(0, 8);
    return { out, ueber, scrollX: document.documentElement.scrollWidth > innerWidth };
  });
  const schlecht = r.out.filter(x => x.ratio < 4.5).sort((a, b) => a.ratio - b.ratio);
  console.log(`\n== ${w}px · ${r.out.length} Text/Grund-Paare · unter 4.5:1: ${schlecht.length} · waagrecht: ${r.scrollX} · ragt hinaus: ${r.ueber.length ? r.ueber.join(" ") : "nichts"}`);
  for (const x of schlecht) console.log(`  ${x.ratio}  ${x.fg} auf ${x.bg}  ${x.wo}  «${x.text}»`);
  const grau = r.out.filter(x => !/rgb\(0, 0, 0\)|rgb\(255, 255, 255\)/.test(x.fg));
  if (grau.length) console.log("  NICHT schwarz/weiss:", [...new Set(grau.map(x => x.fg + " @ " + x.wo))].join(" · "));
  await p.close();
}
await b.close();
