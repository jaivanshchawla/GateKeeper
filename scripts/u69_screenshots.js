const { chromium } = require('playwright');
const path = require('path');
const DIR = path.join(__dirname, '..', 'docs', 'screenshots');
const URL = 'http://localhost:5173';

async function captureAll() {
  const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
  
  for (const theme of ['light', 'dark']) {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: theme });
    const page = await ctx.newPage();
    
    // 1. Overview
    await page.goto(URL, { timeout: 30000, waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);
    await page.screenshot({ path: path.join(DIR, 'overview-' + theme + '.png') });
    console.log('overview-' + theme + ': captured');
    
    // Click first repo card to get repo detail
    const repoCard = await page.$('.repo-card');
    if (repoCard) {
      await repoCard.click();
      await page.waitForTimeout(3000);
      await page.screenshot({ path: path.join(DIR, 'repo-detail-' + theme + '.png') });
      console.log('repo-detail-' + theme + ': captured');
    }
    
    // Get all nav buttons
    const navButtons = await page.$$('nav button');
    
    // 3. PRs
    for (const btn of navButtons) {
      const txt = await btn.textContent();
      if (txt && txt.includes('PR')) {
        await btn.click();
        await page.waitForTimeout(2000);
        await page.screenshot({ path: path.join(DIR, 'prs-' + theme + '.png') });
        console.log('prs-' + theme + ': captured');
        break;
      }
    }
    
    // 4. File Detail
    for (const btn of navButtons) {
      const txt = await btn.textContent();
      if (txt && txt.includes('File')) {
        await btn.click();
        await page.waitForTimeout(2000);
        await page.screenshot({ path: path.join(DIR, 'file-detail-' + theme + '.png') });
        console.log('file-detail-' + theme + ': captured');
        break;
      }
    }
    
    // 5. Config Editor
    for (const btn of navButtons) {
      const txt = await btn.textContent();
      if (txt && txt.includes('Config')) {
        await btn.click();
        await page.waitForTimeout(2000);
        await page.screenshot({ path: path.join(DIR, 'config-editor-' + theme + '.png') });
        console.log('config-editor-' + theme + ': captured');
        break;
      }
    }
    
    // 6. Model Health
    for (const btn of navButtons) {
      const txt = await btn.textContent();
      if (txt && txt.includes('Health')) {
        await btn.click();
        await page.waitForTimeout(2000);
        await page.screenshot({ path: path.join(DIR, 'model-health-' + theme + '.png') });
        console.log('model-health-' + theme + ': captured');
        break;
      }
    }
    
    // 7. Commit Detail — go back to overview, click repo, then click Detail on a commit
    for (const btn of navButtons) {
      const txt = await btn.textContent();
      if (txt && txt.includes('Overview')) {
        await btn.click();
        await page.waitForTimeout(2000);
        break;
      }
    }
    const rc = await page.$('.repo-card');
    if (rc) {
      await rc.click();
      await page.waitForTimeout(3000);
      const detailBtn = await page.$('.toggle-btn');
      if (detailBtn) {
        await detailBtn.click();
        await page.waitForTimeout(2000);
        await page.screenshot({ path: path.join(DIR, 'commit-detail-' + theme + '.png') });
        console.log('commit-detail-' + theme + ': captured');
      }
    }
    
    await ctx.close();
  }
  
  await browser.close();
  console.log('\nAll screenshots captured!');
}

captureAll().catch(e => { console.error(e.message); process.exit(1); });
