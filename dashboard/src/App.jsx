import { useState, useEffect, useCallback, useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell,
  AreaChart, Area, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts'

const API = import.meta.env.VITE_API_URL || '/api'
const BAND_COLORS = { high: '#ef4444', medium: '#f59e0b', low: '#22c55e' }
const BAND_BG = { high: 'rgba(239,68,68,0.12)', medium: 'rgba(245,158,11,0.12)', low: 'rgba(34,197,94,0.12)' }
const BAND_DISPLAY = { high: 'HIGH RISK', medium: 'ELEVATED', low: 'NOT FLAGGED' }

// ── API helpers ──────────────────────────────────────────────
async function api(path) {
  const r = await fetch(`${API}${path}`)
  if (!r.ok) throw new Error(`API ${r.status}: ${r.statusText}`)
  return r.json()
}
async function apiPost(path, body) {
  const r = await fetch(`${API}${path}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`API ${r.status}: ${r.statusText}`)
  return r.json()
}

// ── Shared components ────────────────────────────────────────
function BandBadge({ band, size = 'sm' }) {
  return <span className={`status-badge ${band}`}>{BAND_DISPLAY[band] || band}</span>
}

function StatCard({ value, label, risk }) {
  const cls = risk ? `stat-card risk-${risk}` : 'stat-card'
  return <div className={cls}><h3>{value}</h3><p>{label}</p></div>
}

function Loading() { return <div className="loading">Loading...</div> }
function Empty({ text }) { return <div className="empty-state"><h3>{text}</h3></div> }

// ── 1. Overview ──────────────────────────────────────────────
function OverviewView({ onSelectRepo }) {
  const [repos, setRepos] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => { api('/repos').then(setRepos).catch(() => {}).finally(() => setLoading(false)) }, [])
  if (loading) return <Loading />

  const totalIssues = repos.reduce((s, r) => s + (r.open_issues || 0), 0)
  const totalCommits = repos.reduce((s, r) => s + (r.total_commits || 0), 0)

  return (
    <div>
      <h1>Gatekeeper</h1>
      <div className="stats-grid">
        <StatCard value={repos.length} label="Repos" />
        <StatCard value={totalCommits} label="Commits Scored" />
        <StatCard value={totalIssues} label="Open Issues" risk={totalIssues > 0 ? 'high' : 'low'} />
      </div>
      <div className="card">
        <h2>Repositories</h2>
        <div className="repo-grid">
          {repos.map(repo => (
            <div key={repo.id} className="repo-card" onClick={() => onSelectRepo(repo.id)}>
              <h3>{repo.name}</h3>
              <div className="repo-url">{repo.remote_url || 'no remote'}</div>
              <div className="repo-stats">
                <BandBadge band={repo.last_score || 'low'} />
                <span>{repo.open_issues || 0} issues</span>
              </div>
              {repo.risk_trend && (
                <div style={{ marginTop: 8, fontSize: '0.75rem', color: 'var(--text-2)' }}>
                  Trend: {repo.risk_trend}
                </div>
              )}
              <div className="repo-date">Registered {repo.registered_at?.slice(0, 10) || 'unknown'}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── 2. Repo Detail ───────────────────────────────────────────
function RepoDetailView({ repoId, onSelectCommit, onBack }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => { api(`/repos/${repoId}`).then(setData).catch(() => {}).finally(() => setLoading(false)) }, [repoId])
  if (loading) return <Loading />
  if (!data) return <div className="error">Repo not found</div>

  const { repo, commits, band_counts, hotspots, total_commits } = data
  const pieData = [
    { name: 'Not Flagged', value: band_counts?.low || 0 },
    { name: 'Elevated', value: band_counts?.medium || 0 },
    { name: 'High Risk', value: band_counts?.high || 0 },
  ].filter(d => d.value > 0)
  const pieColors = [BAND_COLORS.low, BAND_COLORS.medium, BAND_COLORS.high]

  const timelineData = (commits || []).slice(0, 30).reverse().map(c => ({
    sha: c.sha?.slice(0, 8), score: c.score, band: c.risk_label,
  }))

  const authorMap = {}
  ;(commits || []).forEach(c => {
    const a = c.author || 'unknown'
    if (!authorMap[a]) authorMap[a] = { low: 0, medium: 0, high: 0, total: 0 }
    authorMap[a][c.risk_label || 'low']++
    authorMap[a].total++
  })
  const authorData = Object.entries(authorMap)
    .map(([name, counts]) => ({ name: name.slice(0, 20), ...counts }))
    .sort((a, b) => b.total - a.total).slice(0, 10)

  return (
    <div>
      <div className="header-row">
        <button className="back-btn" onClick={onBack}>← Back</button>
        <h1>{repo.name}</h1>
      </div>
      <div className="repo-url" style={{ marginBottom: 'var(--sp-5)' }}>{repo.remote_url}</div>

      <div className="stats-grid">
        <StatCard value={total_commits || 0} label="Commits Scored" />
        <StatCard value={hotspots?.length || 0} label="File Hotspots" />
        <StatCard value={band_counts?.high || 0} label="High Risk" risk="high" />
        <StatCard value={band_counts?.medium || 0} label="Elevated" risk="med" />
      </div>

      <div className="card-row">
        <div className="card half">
          <h2>Band Distribution</h2>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" outerRadius={70} dataKey="value" label>
                  {pieData.map((_, i) => <Cell key={i} fill={pieColors[i]} />)}
                </Pie>
                <Tooltip contentStyle={{ background: 'var(--surface-2)', border: '1px solid var(--border)' }} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="card half">
          <h2>Score Timeline (last 30)</h2>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="sha" stroke="var(--text-2)" tick={{ fontSize: 9 }} />
                <YAxis stroke="var(--text-2)" />
                <Tooltip contentStyle={{ background: 'var(--surface-2)', border: '1px solid var(--border)' }} />
                <Bar dataKey="score" radius={[2, 2, 0, 0]}>
                  {timelineData.map((d, i) => <Cell key={i} fill={BAND_COLORS[d.band] || 'var(--text-2)'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {hotspots?.length > 0 && (
        <div className="card">
          <h2>File Hotspots (by revert count & change frequency)</h2>
          <table>
            <thead><tr><th>File</th><th>Changes</th><th>Authors</th></tr></thead>
            <tbody>
              {hotspots.slice(0, 15).map((h, i) => (
                <tr key={i}><td><code>{h.file}</code></td><td>{h.changes}</td><td>{h.authors}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {authorData.length > 0 && (
        <div className="card">
          <h2>Commit Activity by Author</h2>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={authorData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis type="number" stroke="var(--text-2)" />
                <YAxis type="category" dataKey="name" stroke="var(--text-2)" width={140} tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ background: 'var(--surface-2)', border: '1px solid var(--border)' }} />
                <Legend />
                <Bar dataKey="low" stackId="a" fill={BAND_COLORS.low} name="Not Flagged" />
                <Bar dataKey="medium" stackId="a" fill={BAND_COLORS.medium} name="Elevated" />
                <Bar dataKey="high" stackId="a" fill={BAND_COLORS.high} name="High Risk" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div className="card">
        <h2>Recent Commits ({(commits || []).length})</h2>
        <table>
          <thead><tr><th>SHA</th><th>Author</th><th>Band</th><th>Score</th><th>Date</th><th></th></tr></thead>
          <tbody>
            {(commits || []).slice(0, 50).map(c => (
              <tr key={c.id}>
                <td><code>{c.sha?.slice(0, 8)}</code></td>
                <td>{c.author}</td>
                <td><BandBadge band={c.risk_label || 'low'} /></td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>{c.score?.toFixed(4)}</td>
                <td style={{ color: 'var(--text-2)' }}>{c.timestamp?.slice(0, 10)}</td>
                <td><button className="toggle-btn" onClick={() => onSelectCommit(c.id)}>Detail</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── 3. PR View ───────────────────────────────────────────────
function PRView() {
  const [prs, setPrs] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => { api('/prs').then(d => setPrs(d.prs || [])).catch(() => {}).finally(() => setLoading(false)) }, [])
  if (loading) return <Loading />
  if (!prs.length) return <Empty text="No PRs scored yet" />

  return (
    <div>
      <h1>Pull Requests</h1>
      <div className="card">
        <table>
          <thead><tr><th>PR</th><th>Repo</th><th>Verdict</th><th>Commits</th><th>Files</th><th>Riskiest</th><th>Date</th></tr></thead>
          <tbody>
            {prs.map(pr => (
              <tr key={pr.id}>
                <td><code>#{pr.number}</code></td>
                <td>{pr.repo}</td>
                <td><BandBadge band={pr.verdict || 'low'} /></td>
                <td>{pr.commit_count || 0}</td>
                <td>{pr.file_count || 0}</td>
                <td><code>{pr.riskiest_sha?.slice(0, 8) || '-'}</code></td>
                <td style={{ color: 'var(--text-2)' }}>{pr.created_at?.slice(0, 10)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── 4. Commit Detail ─────────────────────────────────────────
function CommitDetailView({ commitId, onBack }) {
  const [commit, setCommit] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => { api(`/commits/${commitId}`).then(setCommit).catch(() => {}).finally(() => setLoading(false)) }, [commitId])
  if (loading) return <Loading />
  if (!commit) return <div className="error">Commit not found</div>

  let rules = [], shap = [], files = []
  try { rules = typeof commit.rule_results === 'string' ? JSON.parse(commit.rule_results) : commit.rule_results || [] } catch {}
  try { shap = typeof commit.shap_top3 === 'string' ? JSON.parse(commit.shap_top3) : commit.shap_top3 || [] } catch {}
  try { files = typeof commit.files_touched === 'string' ? JSON.parse(commit.files_touched) : commit.files_touched || [] } catch {}

  return (
    <div>
      <div className="header-row">
        <button className="back-btn" onClick={onBack}>← Back</button>
        <h1>Commit <code>{commit.sha?.slice(0, 12)}</code></h1>
      </div>
      <div className="stats-grid">
        <StatCard value={BAND_DISPLAY[commit.risk_label] || commit.risk_label} label="Band" risk={commit.risk_label === 'high' ? 'high' : commit.risk_label === 'medium' ? 'med' : 'low'} />
        <StatCard value={commit.author || 'unknown'} label="Author" />
        <StatCard value={`${commit.lines_added || 0}+ / ${commit.lines_deleted || 0}-`} label="Lines Changed" />
        <StatCard value={commit.score?.toFixed(4) || '-'} label="Risk Score" />
      </div>
      {commit.message && <div className="card"><h2>Message</h2><p style={{ color: 'var(--text-1)', fontSize: '0.9rem' }}>{commit.message}</p></div>}

      {shap.length > 0 && (
        <div className="card">
          <h2>Top Contributing Factors (SHAP)</h2>
          <ul className="shap-list">
            {shap.map((s, i) => (
              <li key={i}>
                <span className="shap-feature">{s.feature || s.description}</span>
                <span className={`shap-dir ${s.direction}`}>{s.direction === 'elevates' ? '↑' : '↓'}</span>
                <span className="shap-val">{s.shap_value?.toFixed(4) || '-'}</span>
                <span className="shap-readable">{s.human_readable || s.description}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {rules.length > 0 && (
        <div className="card">
          <h2>Rule Results</h2>
          <table>
            <thead><tr><th>Rule</th><th>Severity</th><th>Status</th><th>Message</th></tr></thead>
            <tbody>
              {rules.map((r, i) => (
                <tr key={i}>
                  <td><code>{r.rule}</code></td>
                  <td><span className={`sev-badge ${r.severity}`}>{r.severity}</span></td>
                  <td>{r.passed ? '✓' : '✗'}</td>
                  <td style={{ color: 'var(--text-1)' }}>{r.message}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {files.length > 0 && (
        <div className="card">
          <h2>Files Changed ({files.length})</h2>
          <ul className="file-list">{files.map((f, i) => <li key={i}><code>{f}</code></li>)}</ul>
        </div>
      )}
    </div>
  )
}

// ── 5. File Detail ───────────────────────────────────────────
function FileDetailView({ onBack }) {
  const [search, setSearch] = useState('')
  const [fileData, setFileData] = useState(null)
  const [loading, setLoading] = useState(false)

  const searchFile = async () => {
    if (!search) return
    setLoading(true)
    try { const d = await api(`/files/${encodeURIComponent(search)}`); setFileData(d) }
    catch { setFileData(null) }
    setLoading(false)
  }

  return (
    <div>
      <div className="header-row">
        <button className="back-btn" onClick={onBack}>← Back</button>
        <h1>File Detail</h1>
      </div>
      <div className="filters">
        <input value={search} onChange={e => setSearch(e.target.value)} placeholder="File path..." style={{ flex: 1 }}
          onKeyDown={e => e.key === 'Enter' && searchFile()} />
        <button className="toggle-btn" onClick={searchFile}>Search</button>
      </div>
      {loading && <Loading />}
      {fileData && (
        <div className="card">
          <h2><code>{fileData.path}</code></h2>
          <div className="stats-grid">
            <StatCard value={fileData.total_changes || 0} label="Total Changes" />
            <StatCard value={fileData.revert_count || 0} label="Reverts" risk={fileData.revert_count > 3 ? 'high' : 'low'} />
            <StatCard value={fileData.distinct_authors || 0} label="Authors" />
            <StatCard value={fileData.risk_rate ? (fileData.risk_rate * 100).toFixed(1) + '%' : '-'} label="Risk Rate" />
          </div>
          {fileData.history?.length > 0 && (
            <div>
              <h2>Change History</h2>
              <table>
                <thead><tr><th>SHA</th><th>Author</th><th>Date</th><th>Band</th></tr></thead>
                <tbody>
                  {fileData.history.map((h, i) => (
                    <tr key={i}>
                      <td><code>{h.sha?.slice(0, 8)}</code></td>
                      <td>{h.author}</td>
                      <td style={{ color: 'var(--text-2)' }}>{h.date?.slice(0, 10)}</td>
                      <td><BandBadge band={h.band || 'low'} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
      {!fileData && !loading && <Empty text="Search for a file to see its risk history" />}
    </div>
  )
}

// ── 6. Config Editor ─────────────────────────────────────────
function ConfigEditorView() {
  const [config, setConfig] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState('')

  useEffect(() => { api('/config').then(setConfig).catch(() => {}).finally(() => setLoading(false)) }, [])
  if (loading) return <Loading />

  const save = async () => {
    setSaving(true); setMsg('')
    try {
      await apiPost('/config', config)
      setMsg('Saved')
    } catch { setMsg('Save failed') }
    setSaving(false)
  }

  return (
    <div>
      <h1>Rule Configuration</h1>
      {msg && <div className={msg === 'Saved' ? '' : 'error'} style={{ padding: 'var(--sp-3)', marginBottom: 'var(--sp-4)' }}>{msg}</div>}
      <div className="card">
        <h2>Default Rule Severities</h2>
        {config?.rules && (
          <table>
            <thead><tr><th>Rule</th><th>Severity</th><th>Enabled</th><th>Parameters</th></tr></thead>
            <tbody>
              {Object.entries(config.rules).map(([name, rule]) => (
                <tr key={name}>
                  <td><code>{name}</code></td>
                  <td>
                    <select value={rule.severity || 'info'} onChange={e => {
                      const c = { ...config, rules: { ...config.rules, [name]: { ...rule, severity: e.target.value } } }
                      setConfig(c)
                    }}>
                      <option value="info">info</option>
                      <option value="warn">warn</option>
                      <option value="block">block</option>
                    </select>
                  </td>
                  <td>
                    <input type="checkbox" checked={rule.enabled !== false} onChange={e => {
                      const c = { ...config, rules: { ...config.rules, [name]: { ...rule, enabled: e.target.checked } } }
                      setConfig(c)
                    }} />
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-2)' }}>
                    {Object.entries(rule).filter(([k]) => !['severity', 'enabled'].includes(k)).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ') || '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <div style={{ marginTop: 'var(--sp-4)' }}>
          <button className="toggle-btn resolve" onClick={save} disabled={saving}>{saving ? 'Saving...' : 'Save Config'}</button>
        </div>
      </div>
    </div>
  )
}

// ── 7. Model Health ──────────────────────────────────────────
function ModelHealthView() {
  const [health, setHealth] = useState(null)
  const [drift, setDrift] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      api('/model/health').catch(() => null),
      api('/drift').catch(() => null),
    ]).then(([h, d]) => { setHealth(h); setDrift(d) }).finally(() => setLoading(false))
  }, [])
  if (loading) return <Loading />

  const repos = drift?.repos || {}
  const driftRepos = Object.entries(repos)

  return (
    <div>
      <h1>Model Health</h1>
      <div className="stats-grid">
        <StatCard value={health?.version || 'v8'} label="Model Version" />
        <StatCard value={health?.roc_auc ? health.roc_auc.toFixed(3) : '-'} label="ROC-AUC (LORO)" />
        <StatCard value={health?.oow_auc ? health.oow_auc.toFixed(3) : '-'} label="ROC-AUC (Out-of-Window)" />
        <StatCard value={health?.n_features || 35} label="Features" />
      </div>

      {driftRepos.length > 0 && (
        <div className="card">
          <h2>Per-Repo Drift Status</h2>
          <table>
            <thead><tr><th>Repo</th><th>Reference</th><th>Current</th><th>Drift?</th><th>Drift Share</th><th>Retraining</th></tr></thead>
            <tbody>
              {driftRepos.map(([name, r]) => (
                <tr key={name}>
                  <td><strong>{name}</strong></td>
                  <td>{r.reference_rows}</td>
                  <td>{r.current_rows}</td>
                  <td><BandBadge band={r.dataset_drift ? 'high' : 'low'} /></td>
                  <td>{r.drift_share != null ? (r.drift_share * 100).toFixed(1) + '%' : '-'}</td>
                  <td>{r.needs_retraining ? 'Recommended' : 'OK'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {driftRepos.filter(([, r]) => r.drifted_features?.length > 0).map(([name, r]) => (
        <div key={name} className="card">
          <h2>{name}: Drifted Features</h2>
          <ul className="file-list">
            {r.drifted_features.map((f, i) => <li key={i}><code>{f}</code></li>)}
          </ul>
        </div>
      ))}

      <div className="card">
        <h2>Known Limitations</h2>
        <ul className="file-list">
          <li><strong>React divergence:</strong> OOW ROC-AUC 0.5542, CI includes 0.5 — no measurable signal post-window</li>
          <li><strong>Band-share drift:</strong> Django 7.2%, React 17.8%, Kafka 16.4%, K8s 15.1%, Rust 11.4% — repos above 10% score higher than training</li>
          <li><strong>Merge commits:</strong> git log --numstat returns 0 files for merges — known limitation, labels incomplete for merge-heavy repos</li>
          <li><strong>Calibration:</strong> Brier 0.224 — scores are rankings, not probabilities</li>
        </ul>
      </div>
    </div>
  )
}

// ── Main App ─────────────────────────────────────────────────
export default function App() {
  const [view, setView] = useState('repos')
  const [selectedRepo, setSelectedRepo] = useState(null)
  const [selectedCommit, setSelectedCommit] = useState(null)

  return (
    <div>
      <nav className="top-nav">
        {[
          ['repos', 'Overview'],
          ['repo-detail', 'Repo Detail'],
          ['prs', 'PRs'],
          ['file', 'File Detail'],
          ['config', 'Config'],
          ['health', 'Model Health'],
        ].map(([key, label]) => (
          <button key={key} className={view === key ? 'active' : ''} onClick={() => setView(key)}>{label}</button>
        ))}
      </nav>

      {view === 'repos' && <OverviewView onSelectRepo={id => { setSelectedRepo(id); setView('repo-detail') }} />}
      {view === 'repo-detail' && (
        <RepoDetailView
          repoId={selectedRepo}
          onSelectCommit={id => { setSelectedCommit(id); setView('commit-detail') }}
          onBack={() => setView('repos')}
        />
      )}
      {view === 'commit-detail' && <CommitDetailView commitId={selectedCommit} onBack={() => setView('repo-detail')} />}
      {view === 'prs' && <PRView />}
      {view === 'file' && <FileDetailView onBack={() => setView('repos')} />}
      {view === 'config' && <ConfigEditorView />}
      {view === 'health' && <ModelHealthView />}
    </div>
  )
}
