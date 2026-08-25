import { useState, useEffect, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell,
} from 'recharts'

const API_URL = import.meta.env.VITE_API_URL || '/api'

const COLORS = { low: '#00f2fe', medium: '#ffa726', high: '#f5576c' }
const PIE_COLORS = ['#00f2fe', '#ffa726', '#f5576c']

// ── API helpers ──────────────────────────────────────────────────────

async function api(path) {
  const r = await fetch(`${API_URL}${path}`)
  if (!r.ok) throw new Error(`API ${r.status}: ${r.statusText}`)
  return r.json()
}

async function apiPost(path, body) {
  const r = await fetch(`${API_URL}${path}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`API ${r.status}: ${r.statusText}`)
  return r.json()
}

async function apiPatch(path) {
  const r = await fetch(`${API_URL}${path}`, { method: 'PATCH' })
  if (!r.ok) throw new Error(`API ${r.status}: ${r.statusText}`)
  return r.json()
}

// ── Components ───────────────────────────────────────────────────────

function RepoList({ onSelect, onBack }) {
  const [repos, setRepos] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api('/repos').then(setRepos).catch(() => {}).finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="loading">Loading repos...</div>

  return (
    <div>
      <div className="header-row">
        <h1>🛡️ Gatekeeper Dashboard</h1>
      </div>
      <div className="stats-grid">
        <div className="stat-card"><h3>{repos.length}</h3><p>Repos</p></div>
        <div className="stat-card open">
          <h3>{repos.reduce((s, r) => s + (r.open_issues || 0), 0)}</h3><p>Open Issues</p>
        </div>
      </div>
      <div className="card">
        <h2>Repositories</h2>
        <div className="repo-grid">
          {repos.map(repo => (
            <div key={repo.id} className="repo-card" onClick={() => onSelect(repo.id)}>
              <h3>{repo.name}</h3>
              <p className="repo-url">{repo.remote_url || 'no remote'}</p>
              <div className="repo-stats">
                <span className={`band-dot ${repo.last_score || 'low'}`} />
                <span>{repo.open_issues || 0} open issues</span>
              </div>
              <p className="repo-date">Registered: {repo.registered_at?.slice(0, 10)}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function RepoDetail({ repoId, onSelectCommit, onBack }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api(`/repos/${repoId}`).then(setData).catch(() => {}).finally(() => setLoading(false))
  }, [repoId])

  if (loading) return <div className="loading">Loading repo...</div>
  if (!data) return <div className="error">Repo not found</div>

  const { repo, commits, band_counts, hotspots, total_commits } = data

  const pieData = [
    { name: 'Low', value: band_counts.low },
    { name: 'Medium', value: band_counts.medium },
    { name: 'High', value: band_counts.high },
  ].filter(d => d.value > 0)

  // Timeline data (last 20 commits)
  const timelineData = commits.slice(0, 20).reverse().map(c => ({
    sha: c.sha?.slice(0, 8),
    score: c.score,
    label: c.risk_label,
  }))

  // Author breakdown
  const authorMap = {}
  commits.forEach(c => {
    const a = c.author || 'unknown'
    if (!authorMap[a]) authorMap[a] = { low: 0, medium: 0, high: 0, total: 0 }
    authorMap[a][c.risk_label || 'low']++
    authorMap[a].total++
  })
  const authorData = Object.entries(authorMap)
    .map(([name, counts]) => ({ name: name.slice(0, 15), ...counts }))
    .sort((a, b) => b.total - a.total)
    .slice(0, 10)

  return (
    <div>
      <div className="header-row">
        <button className="back-btn" onClick={onBack}>← Back</button>
        <h1>{repo.name}</h1>
      </div>
      <p className="repo-detail-url">{repo.remote_url}</p>

      <div className="stats-grid">
        <div className="stat-card"><h3>{total_commits}</h3><p>Total Commits</p></div>
        <div className="stat-card"><h3>{hotspots?.length || 0}</h3><p>Hotspot Files</p></div>
      </div>

      {/* Score Distribution */}
      <div className="card-row">
        <div className="card half">
          <h2>Score Distribution</h2>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" outerRadius={80} dataKey="value" label>
                  {pieData.map((entry, i) => <Cell key={i} fill={PIE_COLORS[i]} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="card half">
          <h2>Commit Timeline (last 20)</h2>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="sha" stroke="#666" tick={{ fontSize: 10 }} />
                <YAxis stroke="#666" />
                <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #444' }} />
                <Bar dataKey="score">
                  {timelineData.map((d, i) => (
                    <Cell key={i} fill={COLORS[d.label] || '#666'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* File Hotspots */}
      {hotspots && hotspots.length > 0 && (
        <div className="card">
          <h2>🔥 File Hotspots (by change frequency)</h2>
          <table>
            <thead><tr><th>File</th><th>Changes</th><th>Authors</th></tr></thead>
            <tbody>
              {hotspots.map((h, i) => (
                <tr key={i}>
                  <td><code>{h.file}</code></td>
                  <td>{h.changes}</td>
                  <td>{h.authors}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Per-Author View */}
      {authorData.length > 0 && (
        <div className="card">
          <h2>👥 Commit Activity by Author</h2>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={authorData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis type="number" stroke="#666" />
                <YAxis type="category" dataKey="name" stroke="#666" width={120} tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #444' }} />
                <Legend />
                <Bar dataKey="low" stackId="a" fill={COLORS.low} />
                <Bar dataKey="medium" stackId="a" fill={COLORS.medium} />
                <Bar dataKey="high" stackId="a" fill={COLORS.high} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Recent Commits Table */}
      <div className="card">
        <h2>Recent Commits ({commits.length})</h2>
        <table>
          <thead>
            <tr><th>SHA</th><th>Author</th><th>Band</th><th>Date</th><th>Details</th></tr>
          </thead>
          <tbody>
            {commits.map(c => (
              <tr key={c.id}>
                <td><code>{c.sha?.slice(0, 8)}</code></td>
                <td>{c.author}</td>
                <td><span className={`status-badge ${c.risk_label || 'low'}`}>{c.risk_label || 'low'}</span></td>
                <td>{c.timestamp?.slice(0, 10)}</td>
                <td>
                  <button className="toggle-btn" onClick={() => onSelectCommit(c.id)}>View</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function CommitDetail({ commitId, onBack }) {
  const [commit, setCommit] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api(`/commits/${commitId}`).then(setCommit).catch(() => {}).finally(() => setLoading(false))
  }, [commitId])

  if (loading) return <div className="loading">Loading commit...</div>
  if (!commit) return <div className="error">Commit not found</div>

  let rules = []
  try { rules = typeof commit.rule_results === 'string' ? JSON.parse(commit.rule_results) : commit.rule_results || [] } catch {}
  let shap = []
  try { shap = typeof commit.shap_top3 === 'string' ? JSON.parse(commit.shap_top3) : commit.shap_top3 || [] } catch {}
  let files = []
  try { files = typeof commit.files_touched === 'string' ? JSON.parse(commit.files_touched) : commit.files_touched || [] } catch {}

  return (
    <div>
      <div className="header-row">
        <button className="back-btn" onClick={onBack}>← Back</button>
        <h1>Commit {commit.sha?.slice(0, 12)}</h1>
      </div>

      <div className="stats-grid">
        <div className="stat-card"><h3>{commit.risk_label?.toUpperCase()}</h3><p>Band</p></div>
        <div className="stat-card"><h3>{commit.author}</h3><p>Author</p></div>
        <div className="stat-card"><h3>{commit.lines_added || 0}+ / {commit.lines_deleted || 0}-</h3><p>Lines Changed</p></div>
      </div>

      <p>{commit.message}</p>

      {/* SHAP Explanations */}
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

      {/* Rule Results */}
      {rules.length > 0 && (
        <div className="card">
          <h2>Rule Results</h2>
          <table>
            <thead><tr><th>Rule</th><th>Severity</th><th>Status</th><th>Message</th></tr></thead>
            <tbody>
              {rules.map((r, i) => (
                <tr key={i}>
                  <td>{r.rule}</td>
                  <td><span className={`sev-badge ${r.severity}`}>{r.severity}</span></td>
                  <td>{r.passed ? '✅' : '⚠️'}</td>
                  <td>{r.message}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Files */}
      {files.length > 0 && (
        <div className="card">
          <h2>Files Changed ({files.length})</h2>
          <ul className="file-list">
            {files.map((f, i) => <li key={i}><code>{f}</code></li>)}
          </ul>
        </div>
      )}
    </div>
  )
}

// ── Issues view (existing, kept for compatibility) ───────────────────

function IssuesView() {
  const [issues, setIssues] = useState([])
  const [stats, setStats] = useState({ daily: [], totals: { open: 0, resolved: 0, total: 0 } })
  const [loading, setLoading] = useState(true)
  const [statusFilter, setStatusFilter] = useState('')
  const [repoFilter, setRepoFilter] = useState('')
  const [repos, setRepos] = useState([])

  const fetchIssues = async () => {
    const params = new URLSearchParams()
    if (statusFilter) params.append('status', statusFilter)
    if (repoFilter) params.append('repo', repoFilter)
    params.append('limit', '100')
    const data = await api(`/issues?${params}`)
    setIssues(data.issues)
    setRepos([...new Set(data.issues.map(i => i.repo))])
  }

  const fetchStats = async () => {
    const data = await api('/issues/stats')
    setStats(data)
  }

  const toggleStatus = async (id) => {
    await apiPatch(`/issues/${id}`)
    fetchIssues()
    fetchStats()
  }

  useEffect(() => {
    setLoading(true)
    Promise.all([fetchIssues(), fetchStats()]).finally(() => setLoading(false))
  }, [statusFilter, repoFilter])

  if (loading) return <div className="loading">Loading...</div>

  return (
    <div>
      <div className="stats-grid">
        <div className="stat-card"><h3>{stats.totals.total}</h3><p>Total Issues</p></div>
        <div className="stat-card open"><h3>{stats.totals.open}</h3><p>Open</p></div>
        <div className="stat-card resolved"><h3>{stats.totals.resolved}</h3><p>Resolved</p></div>
      </div>
      <div className="card">
        <h2>Issues Over Time</h2>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={stats.daily}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="date" stroke="#666" />
              <YAxis stroke="#666" />
              <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #444' }} />
              <Legend />
              <Line type="monotone" dataKey="open" stroke="#f5576c" strokeWidth={2} />
              <Line type="monotone" dataKey="resolved" stroke="#00f2fe" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="filters">
        <select value={statusFilter} onChange={e => setStatusFilter(e.target.value)}>
          <option value="">All Statuses</option>
          <option value="open">Open</option>
          <option value="resolved">Resolved</option>
        </select>
        <select value={repoFilter} onChange={e => setRepoFilter(e.target.value)}>
          <option value="">All Repos</option>
          {repos.map(r => <option key={r} value={r}>{r}</option>)}
        </select>
      </div>
      <div className="card">
        <h2>Issues ({issues.length})</h2>
        <table>
          <thead><tr><th>Gate</th><th>Type</th><th>Repo</th><th>Status</th><th>Details</th><th>Action</th></tr></thead>
          <tbody>
            {issues.length === 0 ? (
              <tr><td colSpan={6} style={{ textAlign: 'center', padding: '2rem' }}>No issues</td></tr>
            ) : issues.map(issue => (
              <tr key={issue.id}>
                <td><span className={`gate-badge gate-${issue.gate}`}>Gate {issue.gate}</span></td>
                <td>{issue.type}</td>
                <td>{issue.repo}</td>
                <td><span className={`status-badge ${issue.status}`}>{issue.status}</span></td>
                <td>{issue.details || '-'}</td>
                <td>
                  <button className={`toggle-btn ${issue.status === 'open' ? 'resolve' : 'reopen'}`}
                    onClick={() => toggleStatus(issue.id)}>
                    {issue.status === 'open' ? 'Resolve' : 'Reopen'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Main App ─────────────────────────────────────────────────────────

function DriftView() {
  const [drift, setDrift] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api('/drift').then(setDrift).catch(() => {}).finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="loading">Loading drift data...</div>
  if (!drift) return <div className="card"><h2>No drift data</h2><p>Run <code>scripts/drift_per_repo.py</code> first.</p></div>

  const repos = drift.repos || {}
  const summary = drift.summary || {}

  return (
    <div>
      <h1>📊 Drift Monitoring</h1>
      <p className="repo-detail-url">Generated: {drift.generated_at?.slice(0, 19)}</p>
      <div className="stats-grid">
        <div className="stat-card"><h3>{summary.repos_analyzed || 0}</h3><p>Repos Analyzed</p></div>
        <div className="stat-card open"><h3>{summary.repos_with_drift || 0}</h3><p>With Drift</p></div>
        <div className="stat-card resolved"><h3>{summary.needs_retraining ? 'YES' : 'NO'}</h3><p>Needs Retrain</p></div>
      </div>
      <div className="card">
        <h2>Per-Repo Drift Status</h2>
        <table>
          <thead><tr><th>Repo</th><th>Reference</th><th>Current</th><th>Drift?</th><th>Drift Share</th><th>Drifted Features</th></tr></thead>
          <tbody>
            {Object.entries(repos).map(([name, r]) => (
              <tr key={name}>
                <td><strong>{name}</strong></td>
                <td>{r.reference_rows}</td>
                <td>{r.current_rows}</td>
                <td><span className={`status-badge ${r.dataset_drift ? 'high' : 'low'}`}>{r.dataset_drift ? 'DRIFT' : 'OK'}</span></td>
                <td>{r.drift_share != null ? (r.drift_share * 100).toFixed(1) + '%' : '-'}</td>
                <td>{r.drifted_count != null ? `${r.drifted_count}/${r.total_features}` : '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {/* Show drifted features per repo */}
      {Object.entries(repos).filter(([, r]) => r.drifted_features?.length > 0).map(([name, r]) => (
        <div key={name} className="card">
          <h2>⚠️ {name}: Drifted Features</h2>
          <ul className="file-list">
            {r.drifted_features.map((f, i) => <li key={i}><code>{f}</code></li>)}
          </ul>
        </div>
      ))}
    </div>
  )
}

function App() {
  const [view, setView] = useState('repos')
  const [selectedRepo, setSelectedRepo] = useState(null)
  const [selectedCommit, setSelectedCommit] = useState(null)

  return (
    <div>
      <nav className="top-nav">
        <button className={view === 'repos' ? 'active' : ''} onClick={() => setView('repos')}>Repos</button>
        <button className={view === 'drift' ? 'active' : ''} onClick={() => setView('drift')}>Drift</button>
        <button className={view === 'issues' ? 'active' : ''} onClick={() => setView('issues')}>Issues</button>
      </nav>

      {view === 'repos' && (
        <RepoList onSelect={id => { setSelectedRepo(id); setView('repo-detail') }} />
      )}
      {view === 'repo-detail' && (
        <RepoDetail
          repoId={selectedRepo}
          onSelectCommit={id => { setSelectedCommit(id); setView('commit-detail') }}
          onBack={() => setView('repos')}
        />
      )}
      {view === 'commit-detail' && (
        <CommitDetail commitId={selectedCommit} onBack={() => setView('repo-detail')} />
      )}
      {view === 'drift' && <DriftView />}
      {view === 'issues' && <IssuesView />}
    </div>
  )
}

export default App
