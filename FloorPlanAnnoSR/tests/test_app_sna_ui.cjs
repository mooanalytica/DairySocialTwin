const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "static", "index.html"));
const css = fs.readFileSync(path.join(root, "static", "app.css"));
const javascript = fs.readFileSync(path.join(root, "static", "app_sna.js"));
const png = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
  "base64"
);

const groups = [];
for (let farmId = 1; farmId <= 3; farmId += 1) {
  for (let cameraId = 1; cameraId <= 5; cameraId += 1) {
    const id = `farm_ID_${farmId}_camera_ID_${cameraId}`;
    groups.push({
      id,
      farmId,
      cameraId,
      label: `Farm ${farmId} / Camera ${cameraId}`,
      frameExists: true,
      zonesExists: id === "farm_ID_1_camera_ID_1",
      zoneCount: id === "farm_ID_1_camera_ID_1" ? 5 : 0,
    });
  }
}

function blankAnnotation(groupId) {
  const match = /^farm_ID_(\d+)_camera_ID_(\d+)$/.exec(groupId);
  const farmId = Number(match[1]);
  const cameraId = Number(match[2]);
  return {
    schemaVersion: "sna_zones.v1",
    groupId,
    farm: String(farmId),
    camera: `Gopro${cameraId}`,
    farmId,
    cameraId,
    coordinate_system: "reference_frame_pixel",
    zones: [],
  };
}

const annotations = new Map(groups.map((group) => [group.id, blankAnnotation(group.id)]));
annotations.get("farm_ID_1_camera_ID_1").zones = Array.from({ length: 5 }, (_, index) => ({
  zone_id: `resource_${index + 1}`,
  zone_type: "resource",
  label: `Resource ${index + 1}`,
  color: "#0b7285",
  polygon: [[0, 0], [1, 0], [1, 1]],
}));

let failNextSave = false;
let holdNextSave = false;
let heldSaveStarted = Promise.resolve();
let resolveHeldSaveStarted = null;
let releaseHeldSave = null;
const successfulSaves = [];

async function main() {
  const browserPath = [
    process.env.PLAYWRIGHT_BROWSER_PATH,
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  ].find((candidate) => candidate && fs.existsSync(candidate));
  assert.ok(browserPath, "No installed Chromium-compatible browser was found");
  const browser = await chromium.launch({ headless: true, executablePath: browserPath });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });

  await page.route("http://sna.test/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/") {
      await route.fulfill({ status: 200, contentType: "text/html; charset=utf-8", body: html });
    } else if (url.pathname === "/static/app.css") {
      await route.fulfill({ status: 200, contentType: "text/css; charset=utf-8", body: css });
    } else if (url.pathname === "/static/app_sna.js") {
      await route.fulfill({ status: 200, contentType: "text/javascript; charset=utf-8", body: javascript });
    } else if (url.pathname === "/frame") {
      await route.fulfill({ status: 200, contentType: "image/png", body: png });
    } else if (url.pathname === "/api/config") {
      const groupId = url.searchParams.get("group") || "farm_ID_1_camera_ID_1";
      const group = groups.find((item) => item.id === groupId);
      const body = {
        schemaVersion: "sna_zones.v1",
        coordinateSystem: "reference_frame_pixel",
        groups,
        activeGroupId: groupId,
        activeGroup: group,
        annotation: annotations.get(groupId),
        defaultColors: ["#0b7285", "#5f3dc4"],
        frame: {
          exists: true,
          url: `/frame?group=${encodeURIComponent(groupId)}`,
          metadata: { sourceVideoName: `${groupId}.MP4`, timestamp: "00:01:00" },
        },
      };
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
    } else if (url.pathname === "/api/save") {
      const payload = request.postDataJSON();
      if (holdNextSave) {
        holdNextSave = false;
        const release = new Promise((resolve) => {
          releaseHeldSave = resolve;
        });
        resolveHeldSaveStarted();
        await release;
      }
      if (failNextSave) {
        failNextSave = false;
        await route.fulfill({
          status: 400,
          contentType: "application/json",
          body: JSON.stringify({ ok: false, error: "simulated save failure" }),
        });
      } else {
        annotations.set(payload.groupId, payload);
        successfulSaves.push(payload);
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            ok: true,
            groupId: payload.groupId,
            updatedAt: "2026-07-20T12:00:00Z",
            zonesPath: `${payload.groupId}/sna_zones.json`,
          }),
        });
      }
    } else {
      await route.abort();
    }
  });

  try {
    await page.goto("http://sna.test/");
    await page.waitForFunction(() => document.querySelector("#saveStatus")?.textContent === "Loaded 5 zones");
    assert.equal(await page.locator("#groupSelect option").count(), 15);
    assert.equal(await page.locator("#zoneList .zone-row").count(), 5);

    await page.locator("#zoneList .zone-row").first().click();
    await page.locator("#zoneLabelInput").fill("Draft label");
    await page.selectOption("#groupSelect", "farm_ID_1_camera_ID_2");
    await page.locator("#switchDialog").waitFor({ state: "visible" });
    await page.locator("#switchCancelBtn").click();
    assert.equal(await page.locator("#groupSelect").inputValue(), "farm_ID_1_camera_ID_1");
    assert.equal(await page.locator("#zoneLabelInput").inputValue(), "Draft label");

    await page.selectOption("#groupSelect", "farm_ID_1_camera_ID_2");
    await page.locator("#switchDiscardBtn").click();
    await page.waitForFunction(() => new URL(location.href).searchParams.get("group") === "farm_ID_1_camera_ID_2");
    assert.equal(successfulSaves.length, 0);
    assert.equal(await page.locator("#zoneList .zone-row").count(), 0);

    await page.selectOption("#groupSelect", "farm_ID_1_camera_ID_1");
    await page.waitForFunction(() => new URL(location.href).searchParams.get("group") === "farm_ID_1_camera_ID_1");
    await page.locator("#zoneList .zone-row").first().click();
    assert.equal(await page.locator("#zoneLabelInput").inputValue(), "Resource 1");
    await page.locator("#zoneLabelInput").fill("Saved Unicode \u8cc7\u6e90\u5340");

    failNextSave = true;
    await page.selectOption("#groupSelect", "farm_ID_1_camera_ID_2");
    await page.locator("#switchSaveBtn").click();
    await page.waitForFunction(() =>
      document.querySelector("#switchDialogStatus")?.textContent.includes("simulated save failure")
    );
    assert.equal(await page.locator("#switchDialog").getAttribute("open"), "");
    assert.equal(await page.locator("#groupSelect").inputValue(), "farm_ID_1_camera_ID_1");

    await page.locator("#switchSaveBtn").click();
    await page.waitForFunction(() => new URL(location.href).searchParams.get("group") === "farm_ID_1_camera_ID_2");
    assert.equal(successfulSaves.length, 1);
    assert.equal(successfulSaves[0].groupId, "farm_ID_1_camera_ID_1");
    assert.equal(successfulSaves[0].zones[0].label, "Saved Unicode \u8cc7\u6e90\u5340");

    await page.selectOption("#groupSelect", "farm_ID_1_camera_ID_1");
    await page.waitForFunction(() => new URL(location.href).searchParams.get("group") === "farm_ID_1_camera_ID_1");
    await page.locator("#zoneList .zone-row").first().click();
    const countBeforeDeleteKey = await page.locator("#zoneList .zone-row").count();
    await page.locator("#zoneLabelInput").press("End");
    await page.locator("#zoneLabelInput").press("Backspace");
    assert.equal(await page.locator("#zoneList .zone-row").count(), countBeforeDeleteKey);

    holdNextSave = true;
    heldSaveStarted = new Promise((resolve) => {
      resolveHeldSaveStarted = resolve;
    });
    await page.locator("#saveBtn").click();
    await heldSaveStarted;
    assert.equal(await page.locator("#zoneTypeInput").isDisabled(), true);
    assert.equal(await page.locator("#zoneLabelInput").isDisabled(), true);
    assert.equal(await page.locator("#zoneColorInput").isDisabled(), true);
    assert.equal(await page.locator("#zoneList .zone-row").first().isDisabled(), true);
    releaseHeldSave();
    await page.waitForFunction(() => !document.querySelector("#saveBtn")?.disabled);
    assert.equal(await page.locator("#zoneLabelInput").isDisabled(), false);

    console.log("UI OK: 15 groups, Cancel, Discard, Save failure, Save & Switch, Unicode, input keyboard safety, busy-state locking");
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
