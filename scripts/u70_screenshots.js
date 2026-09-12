const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const BASE = 'http://localhost:5174';
const OUT = path.join(__dirname, '..', 'docs', 'screenshots');
fs.mkdirSync(OUT, { recursive: true });

(async () => {
  const browser = await chromium.launch({ headless: true });
  
  for (const theme of ['light', 'dark']) {
    const ctx = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      colorScheme: theme,
    });
    const page = await ctx.newPage();
    await page.goto(BASE, { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);

    await page.evaluate((t) => {
      document.documentElement.setAttribute('data-theme', t);
      localStorage.setItem('theme', t);
    }, theme);
    await page.waitForTimeout(500);

    // 1. Overview
    await page.waitForSelector('text=django', { timeout: 5000 }).catch(() => {});
    await page.waitForTimeout(1000);
    await page.screenshot({ path: path.join(OUT, `overview-${theme}.png`) });
    console.log(`1. overview-${theme}`);

    // 2. Repo Detail
    await page.click('text=django >> nth=0', { timeout: 3000 }).catch(() => {});
    await page.waitForTimeout(2500);
    await page.screenshot({ path: path.join(OUT, `repo-detail-${theme}.png`) });
    console.log(`2. repo-detail-${theme}`);

    // 3. Commit Detail - use locator for .toggle-btn
    try {
      const toggleBtn = page.locator('.toggle-btn').first();
      await toggleBtn.waitFor({ timeout: 3000 });
      await toggleBtn.click();
      await page.waitForTimeout(3000);
      await page.screenshot({ path: path.join(OUT, `commit-detail-${theme}.png`) });
      console.log(`3. commit-detail-${theme}`);
    } catch(e) {
      console.log(`3. commit-detail-${theme} (fallback: ${e.message})`);
      await page.screenshot({ path: path.join(OUT, `commit-detail-${theme}.png`) });
    }

    // Go back
    await page.click('text=Overview', { timeout: 3000 }).catch(() => {});
    await page.waitForTimeout(1500);

    // 4. PRs
    await page.click('text=PRs', { timeout: 3000 }).catch(() => {});
    await page.waitForTimeout(2000);
    await page.screenshot({ path: path.join(OUT, `prs-${theme}.png`) });
    console.log(`4. prs-${theme}`);

    // 5. File Detail
    await page.click('text=File Detail', { timeout: 3000 }).catch(() => {});
    await page.waitForTimeout(1500);
    const searchInput = await page.$('input[placeholder*="File"]');
    if (searchInput) {
      await searchInput.fill('django/db/models/query.py');
      await searchInput.press('Enter');
      await page.waitForTimeout(2500);
    }
    await page.screenshot({ path: path.join(OUT, `file-detail-${theme}.png`) });
    console.log(`5. file-detail-${theme}`);

    // 6. Config
    await page.click('text=Config', { timeout: 3000 }).catch(() => {});
    await page.waitForTimeout(1500);
    await page.screenshot({ path: path.join(OUT, `config-editor-${theme}.png`) });
    console.log(`6. config-editor-${theme}`);

    // 7. Model Health
    await page.click('text=Model Health', { timeout: 3000 }).catch(() => {});
    await page.waitForTimeout(1500);
    await page.screenshot({ path: path.join(OUT, `model-health-${theme}.png`) });
    console.log(`7. model-health-${theme}`);

    await ctx.close();
  }
  
  await browser.close();
  console.log('Done - 14 screenshots');
})();
