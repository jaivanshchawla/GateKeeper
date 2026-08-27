const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const chromePath = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const outDir = path.resolve(__dirname, 'docs/screenshots');
fs.mkdirSync(outDir, { recursive: true });

const urls = [
  { name: 'overview', url: 'http://localhost:5173/' },
  { name: 'repo-detail', url: 'http://localhost:5173/' },
  { name: 'model-health', url: 'http://localhost:5173/' },
];

urls.forEach(({name, url}) => {
  const outFile = path.join(outDir, name + '.png');
  console.log('Capturing', name, '->', outFile);
  try {
    execSync(`"${chromePath}" --headless=new --disable-gpu --screenshot="${outFile}" --window-size=1440,900 --no-sandbox "${url}"`, { timeout: 15000, stdio: 'pipe' });
    console.log('  OK:', fs.existsSync(outFile) ? 'file exists' : 'no file');
  } catch(e) {
    console.log('  Error:', e.message.slice(0, 200));
  }
});
console.log('Done');
