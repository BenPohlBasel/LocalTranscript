// Throwaway review script (review 2026-09-11, video E2E). Run:
//   node pw-video-review.mjs webkit|chromium|chrome
import { chromium, webkit } from "playwright";

const BASE = "http://127.0.0.1:5631";
const which = process.argv[2] || "webkit";
const ids = JSON.parse(process.argv[3] || "{}"); // { video: id, audio: id }
const log = (...a) => console.log(`[${which}]`, ...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = which === "webkit" ? await webkit.launch()
  : which === "chrome" ? await chromium.launch({ channel: "chrome" })
  : await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1200, height: 700 } });
const page = await ctx.newPage();
page.on("pageerror", (e) => log("PAGEERROR", e.message));
page.on("console", (m) => { if (m.type() === "error") log("CONSOLE", m.text()); });

await page.goto(BASE + "/");
await page.evaluate(() => { try { sessionStorage.clear(); localStorage.clear(); } catch {} });
await page.goto(BASE + "/");
await page.waitForSelector("text=LocalTranscript");

async function openEntry(name) {
  // go to editor list tab, click entry name
  await page.getByRole("radio", { name: /Human-Editor/ }).first().click().catch(() => {});
  const row = page.getByText(name, { exact: true }).first();
  await row.waitFor({ timeout: 10000 });
  await row.click();
  await page.waitForFunction(() => document.querySelector("audio,video") || document.body.innerText.includes("Kein Audio"), null, { timeout: 10000 });
  await sleep(500);
}

const state = async () => page.evaluate(() => {
  const a = document.querySelector("audio"), v = document.querySelector("video");
  return {
    audio: a ? { t: +a.currentTime.toFixed(2), paused: a.paused, rate: a.playbackRate, ready: a.readyState, src: a.src } : null,
    video: v ? { t: +v.currentTime.toFixed(2), paused: v.paused, rate: v.playbackRate, ready: v.readyState, muted: v.muted,
      filter: getComputedStyle(v).filter, src: v.src, w: v.clientWidth, h: v.clientHeight, err: v.error && v.error.code } : null,
    videos: document.querySelectorAll("video").length,
  };
});

log("== open video entry");
await openEntry("interview");
log("initial", await state());
await page.waitForFunction(() => { const v = document.querySelector("video"); return v && v.readyState >= 1; }, null, { timeout: 15000 }).catch(() => log("video never reached HAVE_METADATA"));
log("after load", await state());

// play via play button
const playBtn = page.locator("button[title^='Abspielen'], button[title^='Pause']").first();
await playBtn.click();
await sleep(1500);
log("playing 1.5s", await state());

// segment click (jump)
await page.locator("[data-seg='3']").locator("button").first().click();
await sleep(200);
log("just after seg3 click", await state());
await sleep(1500);
log("1.5s after seg3 click", await state());

// arrow key jumps
await page.keyboard.press("ArrowDown");
await sleep(150);
log("after ArrowDown", await state());
await sleep(1200);
log("1.2s after ArrowDown", await state());

// Ctrl+X rate cycling
for (let i = 0; i < 5; i++) {
  await page.keyboard.press("Control+x");
  await sleep(500);
  log(`after Ctrl+X #${i + 1}`, await state());
}

// pause; then click same segment twice while paused → does blur clear?
await page.keyboard.press("Space").catch(() => {});
const s0 = await state();
if (s0.audio && !s0.audio.paused) await playBtn.click();
await sleep(300);
log("paused", await state());
await page.locator("[data-seg='1']").locator("button").first().click();
await sleep(800);
log("paused, seg1 click", await state());
await page.locator("[data-seg='1']").locator("button").first().click();
await sleep(800);
log("paused, seg1 click again", await state());

// switch tab to Suchen while playing, then back
await playBtn.click().catch(() => {});
await sleep(500);
log("playing before tab switch", await state());
await page.getByText(/Suchen|Search|Rechercher|Cerca/).first().click();
await sleep(1500);
log("on Suchen tab", await state());
await page.getByText(/Sprecher|Speakers|Locuteurs|Parlanti|Intervenants/).first().click();
await sleep(1500);
log("back on Sprecher tab", await state());
await page.locator("[data-seg='4']").locator("button").first().click();
await sleep(1500);
log("after seg4 click post-tab-switch", await state());

// export menu items
await page.getByText(/^Export/).first().click();
await sleep(300);
const items = await page.locator("[role='option']").allTextContents();
log("export items", items);
await page.keyboard.press("Escape");

// side panel: pinned video with many speakers
for (let i = 0; i < 12; i++) {
  await page.locator("button", { hasText: /Neuer Sprecher|New speaker|Nouveau|Nuovo/ }).first().click();
}
await sleep(500);
const geo = await page.evaluate(() => {
  const v = document.querySelector("video"); const r = v.getBoundingClientRect();
  const panel = v.closest(".ui-sidepanel").getBoundingClientRect();
  const scrollers = [...document.querySelectorAll(".ui-sidepanel *")].filter((e) => e.scrollHeight > e.clientHeight + 2 && getComputedStyle(e).overflowY.match(/auto|scroll/)).map((e) => ({ tag: e.tagName, sh: e.scrollHeight, ch: e.clientHeight }));
  return { video: { top: r.top, bottom: r.bottom }, panel: { top: panel.top, bottom: panel.bottom }, winH: innerHeight, scrollers };
});
log("geometry with 12 more speakers", JSON.stringify(geo));
await page.screenshot({ path: `/private/tmp/claude-501/-Users-benpohl-Claude-enrich-transcript/ba44b2d0-f49b-49f7-ad4e-7b5976c2d567/scratchpad/${which}-speakers.png` });

// leave to list (autosave), open audio-only entry, then back
await page.locator("button", { hasText: /Bibliothek|Library/ }).first().click();
await sleep(800);
await openEntry("talk");
log("audio-only entry", await state());
await page.locator("button[title^='Abspielen'], button[title^='Pause']").first().click();
await sleep(800);
log("audio-only playing", await state());
await page.locator("button", { hasText: /Bibliothek|Library/ }).first().click();
await sleep(800);
await openEntry("interview");
await page.waitForFunction(() => { const v = document.querySelector("video"); return v && v.readyState >= 1; }, null, { timeout: 15000 }).catch(() => log("video never reached HAVE_METADATA (2nd)"));
log("video entry again", await state());
await page.locator("button[title^='Abspielen'], button[title^='Pause']").first().click();
await sleep(1500);
log("video entry again playing", await state());

// open the editor with the Suchen tab persisted in sessionStorage
await page.locator("button", { hasText: /Bibliothek|Library/ }).first().click();
await sleep(500);
await openEntry("interview");
await page.getByText(/Suchen|Search|Rechercher|Cerca/).first().click();
await page.locator("button", { hasText: /Bibliothek|Library/ }).first().click();
await sleep(500);
await openEntry("interview");
log("opened with Suchen persisted", await state());
await page.getByText(/Sprecher|Speakers|Locuteurs|Parlanti|Intervenants/).first().click();
await page.waitForFunction(() => { const v = document.querySelector("video"); return v && v.readyState >= 1; }, null, { timeout: 15000 }).catch(() => log("video never reached HAVE_METADATA (3rd)"));
await page.locator("button[title^='Abspielen'], button[title^='Pause']").first().click();
await sleep(2000);
log("Suchen-persisted → Sprecher, playing 2s", await state());
await page.locator("[data-seg='4']").locator("button").first().click();
await sleep(1500);
log("… then seg4 click", await state());

await browser.close();
