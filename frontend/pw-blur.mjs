import { chromium, webkit } from "playwright";
const which = process.argv[2] || "webkit";
const log = (...a) => console.log(`[${which}]`, ...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const browser = which === "webkit" ? await webkit.launch() : await chromium.launch({ channel: "chrome" });
const page = await browser.newPage({ viewport: { width: 1200, height: 700 } });
page.on("pageerror", (e) => log("PAGEERROR", e.message));
const st = () => page.evaluate(() => { const a = document.querySelector("audio"), v = document.querySelector("video");
  return `A t=${a.currentTime.toFixed(2)} paused=${a.paused} | V t=${v ? v.currentTime.toFixed(2) : "-"} ready=${v && v.readyState} paused=${v && v.paused} filter=${v && getComputedStyle(v).filter}`; });
async function open(delayVideoMs) {
  await page.goto("http://127.0.0.1:5631/");
  await page.evaluate(() => { try { sessionStorage.clear(); localStorage.clear(); } catch {} });
  if (delayVideoMs) await page.route("**/api/transcripts/interview/video", async (route) => { await sleep(delayVideoMs); await route.continue(); });
  await page.goto("http://127.0.0.1:5631/");
  await page.getByRole("radio", { name: /Human-Editor/ }).click();
  await page.getByText("interview", { exact: true }).click();
  await page.waitForSelector("audio", { state: "attached" });
  await page.waitForFunction(() => document.querySelector("audio").readyState >= 1);
}
// 1) same-position seek: rewind 5 s while at 0
await open(0);
await page.waitForFunction(() => document.querySelector("video").readyState >= 1);
log("start", await st());
await page.locator("button[title^='5 Sekunden zur'], button[title*='zurück']").first().click();
await sleep(600);
log("after rewind at 0", await st());
// ArrowUp at first segment
await page.locator("[data-seg='0'] button").first().click(); await sleep(400);
await page.locator("button[title^='Pause']").first().click().catch(() => {}); await sleep(300);
log("seg0 clicked & paused", await st());
await page.keyboard.press("ArrowUp"); await sleep(600);
log("ArrowUp at seg0", await st());
// 2) seek before video has metadata
await open(4000);
log("video delayed: before click", await st());
await page.locator("[data-seg='3'] button").first().click();
await sleep(300);
log("video delayed: after seg3 click", await st());
await sleep(5500);
log("video delayed: after video loaded", await st());
await sleep(1500);
log("video delayed: +1.5s", await st());
await browser.close();
