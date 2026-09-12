const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const OUT = path.join(__dirname, '..', 'docs', 'screenshots');
fs.mkdirSync(OUT, { recursive: true });

const THEMES = ['dark', 'light'];

(async () => {
  const browser = await chromium.launch({ headless: true });

  for (const theme of THEMES) {
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await context.newPage();

    // Pre-seed localStorage before navigating
    await page.addInitScript((t) => {
      localStorage.setItem('gk-theme', t);
    }, theme);

    // 1. Overview
    await page.goto('http://localhost:5173/', { waitUntil: 'networkidle', timeout: 30000 });
    await page.evaluate((t) => document.documentElement.setAttribute('data-theme', t), theme);
    await page.waitForTimeout(2000);
    await page.screenshot({ path: path.join(OUT, `overview-${theme}.png`) });
    console.log(`  overview-${theme}: done`);

    // 2. Repo Detail — click first repo card
    try {
      await page.waitForSelector('.repo-card', { timeout: 5000 });
      await page.click('.repo-card');
      await page.waitForSelector('.stats-grid', { timeout: 5000 });
      await page.waitForTimeout(2000);
      await page.screenshot({ path: path.join(OUT, `repo-detail-${theme}.png`) });
      console.log(`  repo-detail-${theme}: done`);
    } catch (e) { console.log(`  repo-detail-${theme}: ${e.message}`); }

    // 3. Commit Detail — click last commit's Detail button
    try {
      await page.waitForSelector('table tbody tr .toggle-btn', { timeout: 5000 });
      const btns = await page.$$('table tbody tr .toggle-btn');
      if (btns.length) await btns[btns.length - 1].click();
      await page.waitForTimeout(2000);
      await page.screenshot({ path: path.join(OUT, `commit-detail-${theme}.png`) });
      console.log(`  commit-detail-${theme}: done`);
    } catch (e) { console.log(`  commit-detail-${theme}: ${e.message}`); }

    // 4. PRs
    try {
      await page.click('button:text("PRs")');
      await page.waitForTimeout(2000);
      await page.screenshot({ path: path.join(OUT, `prs-${theme}.png`) });
      console.log(`  prs-${theme}: done`);
    } catch (e) { console.log(`  prs-${theme}: ${e.message}`); }

    // 5. File Detail
    try {
      await page.click('button:text("File Detail")');
      await page.waitForTimeout(1500);
      await page.screenshot({ path: path.join(OUT, `file-detail-${theme}.png`) });
      console.log(`  file-detail-${theme}: done`);
    } catch (e) { console.log(`  file-detail-${theme}: ${e.message}`); }

    // 6. Config
    try {
      await page.click('button:text("Config")');
      await page.waitForTimeout(2000);
      await page.screenshot({ path: path.join(OUT, `config-${theme}.png`) });
      console.log(`  config-${theme}: done`);
    } catch (e) { console.log(`  config-${theme}: ${e.message}`); }

    // 7. Model Health
    try {
      await page.click('button:text("Model Health")');
      await page.waitForTimeout(2000);
      await page.screenshot({ path: path.join(OUT, `model-health-${theme}.png`) });
      console.log(`  model-health-${theme}: done`);
    } catch (e) { console.log(`  model-health-${theme}: ${e.message}`); }

    await context.close();
  }

  await browser.close();
  console.log('Done: 14 screenshots');
})();
