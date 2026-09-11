import { webkit } from "playwright";
const b = await webkit.launch(); const p = await b.newPage({ viewport: { width: 1200, height: 700 } });
await p.goto("http://127.0.0.1:5631/"); await p.waitForTimeout(1500);
console.log(await p.evaluate(() => [...document.querySelectorAll("button,[role]")].slice(0, 40).map((e) => `${e.tagName} role=${e.getAttribute("role")} ${e.textContent.trim().slice(0, 40)}`).join("\n")));
await p.getByText("Human-Editor").first().click(); await p.waitForTimeout(1500);
console.log("---- after tab");
console.log(await p.evaluate(() => document.body.innerText.slice(0, 1200)));
await b.close();
