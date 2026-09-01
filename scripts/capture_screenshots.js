const { chromium } = require('playwright');
const path = require('path');

const SCREENSHOT_DIR = path.join(__dirname, '..', 'docs', 'screenshots');
const BASE_URL = 'http://localhost:5178';

async function waitForContent(page, timeout = 8000) {
  try {
    await page.waitForFunction(() => {
      const body = document.body;
      return body && body.innerText.trim().length > 100;
    }, { timeout });
  } catch(e) {
    // Timeout — proceed anyway
  }
}

async function captureAll() {
  const browser = await chromium.launch({ headless: true });
  
  for (const theme of ['light', 'dark']) {
    const context = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      colorScheme: theme,
    });
    const page = await context.newPage();
    
    // Go to overview
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded', timeout: 15000 });
    await waitForContent(page);
    await page.waitForTimeout(2000);
    
    // 1. Overview
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, `overview-${theme}.png`) });
    const bodyText = await page.evaluate(() => document.body.innerText.substring(0, 200));
    console.log(`overview-${theme}: ${bodyText.substring(0, 80)}...`);
    
    // Navigate to each view via buttons/tabs
    const viewActions = [
      { name: 'repo-detail', search: /repo/i },
      { name: 'commits', search: /commit/i },
      { name: 'pr-view', search: /pr\b|pull request/i },
      { name: 'file-detail', search: /file/i },
      { name: 'config-editor', search: /config/i },
      { name: 'model-health', search: /model|health/i },
    ];
    
    for (const va of viewActions) {
      try {
        // Go back to overview first
        await page.goto(BASE_URL, { waitUntil: 'domcontentloaded', timeout: 10000 });
        await waitForContent(page);
        await page.waitForTimeout(1000);
        
        // Find and click the matching button/tab
        const buttons = await page.$$('button, [role="tab"], nav a, .nav-item');
        for (const btn of buttons) {
          const text = await btn.textContent();
          if (text && va.search.test(text)) {
            await btn.click();
            await page.waitForTimeout(3000);
            await waitForContent(page);
            break;
          }
        }
        
        await page.screenshot({ path: path.join(SCREENSHOT_DIR, `${va.name}-${theme}.png`) });
        const vt = await page.evaluate(() => document.body.innerText.substring(0, 200));
        console.log(`${va.name}-${theme}: ${vt.substring(0, 80)}...`);
      } catch(e) {
        console.log(`${va.name}-${theme}: FAILED: ${e.message.substring(0, 80)}`);
      }
    }
    
    await context.close();
  }
  
  await browser.close();
  console.log('\nAll screenshots captured!');
}

captureAll().catch(e => { console.error(e); process.exit(1); });
