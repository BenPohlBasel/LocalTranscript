import { chromium, webkit } from "playwright";
const which = process.argv[2] || "webkit";
const log = (...a) => console.log(`[${which}]`, ...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const browser = which === "webkit" ? await webkit.launch() : await chromium.launch({ channel: "chrome" });
const page = await browser.newPage({ viewport: { width: 1200, height: 700 } });
const st = () => page.evaluate(() => { const a = document.querySelector("audio"), v = document.querySelector("video");
  return `A t=${a.currentTime.toFixed(2)} paused=${a.paused} | V t=${v ? v.currentTime.toFixed(2) : "-"} ready=${v && v.readyState} paused=${v && v.paused} filter=${v && getComputedStyle(v).filter}`; });
async function open(delayMs) {
  await page.goto("http://127.0.0.1:5631/");
  await page.evaluate(() => { try { sessionStorage.clear(); localStorage.clear(); } catch {} });
  let n = 0;
  if (delayMs) await page.route("**/api/transcripts/interview/video", async (route) => { n += 1; if (n === 1) await sleep(delayMs); await route.continue(); });
  await page.goto("http://127.0.0.1:5631/");
  await page.getByRole("radio", { name: /Human-Editor/ }).click();
  await page.getByText("interview", { exact: true }).click();
  await page.waitForSelector("audio", { state: "attached" });
  await page.waitForFunction(() => document.querySelector("audio").readyState >= 1);
}
for (const playing of [false, true]) {
  await open(3000);
  log(`delayed video, playing=${playing}: before`, await st());
  if (playing) await page.locator("[data-seg='3'] button").first().click();
  else { await page.keyboard.press("ArrowDown"); }
  await sleep(300);
  log("  after seek during readyState 0", await st());
  await page.waitForFunction(() => document.querySelector("video").readyState >= 2, null, { timeout: 30000 }).catch(() => log("  video never loaded"));
  await sleep(800);
  log("  after video loaded", await st());
  if (playing) { await page.locator("button[title^='Pause']").first().click().catch(() => {}); }
  await sleep(1500);
  log("  +1.5s", await st());
}
await browser.close();
