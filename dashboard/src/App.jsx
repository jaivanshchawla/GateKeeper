import { useState, useEffect } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { format } from 'date-fns'

const API_URL = import.meta.env.VITE_API_URL || '/api'

function App() {
  const [issues, setIssues] = useState([])
  const [stats, setStats] = useState({ daily: [], totals: { open: 0, resolved: 0, total: 0 } })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  
  // Filters
  const [statusFilter, setStatusFilter] = useState('')
  const [repoFilter, setRepoFilter] = useState('')
  const [repos, setRepos] = useState([])

  // Fetch issues
  const fetchIssues = async () => {
    try {
      const params = new URLSearchParams()
      if (statusFilter) params.append('status', statusFilter)
      if (repoFilter) params.append('repo', repoFilter)
      params.append('limit', '100')
      
      const response = await fetch(`${API_URL}/issues?${params}`)
      if (!response.ok) throw new Error('Failed to fetch issues')
      
      const data = await response.json()
      setIssues(data.issues)
      
      // Extract unique repos for filter
      const uniqueRepos = [...new Set(data.issues.map(i => i.repo))]
      setRepos(uniqueRepos)
    } catch (err) {
      setError(err.message)
    }
  }

  // Fetch stats
  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_URL}/issues/stats`)
      if (!response.ok) throw new Error('Failed to fetch stats')
      
      const data = await response.json()
      setStats(data)
    } catch (err) {
      setError(err.message)
    }
  }

  // Toggle issue status
  const toggleStatus = async (issueId) => {
    try {
      const response = await fetch(`${API_URL}/issues/${issueId}`, {
        method: 'PATCH',
      })
      if (!response.ok) throw new Error('Failed to toggle status')
      
      // Refresh data
      await fetchIssues()
      await fetchStats()
    } catch (err) {
      setError(err.message)
    }
  }

  // Load data on mount and when filters change
  useEffect(() => {
    const loadData = async () => {
      setLoading(true)
      await Promise.all([fetchIssues(), fetchStats()])
      setLoading(false)
    }
    loadData()
  }, [statusFilter, repoFilter])

  // Format date for display
  const formatDate = (dateStr) => {
    if (!dateStr) return '-'
    try {
      return format(new Date(dateStr), 'MMM d, yyyy HH:mm')
    } catch {
      return dateStr
    }
  }

  // Format date for chart
  const formatChartDate = (dateStr) => {
    try {
      return format(new Date(dateStr), 'MMM d')
    } catch {
      return dateStr
    }
  }

  if (loading) {
    return <div className="loading">Loading dashboard...</div>
  }

  return (
    <div>
      <h1>🛡️ Gatekeeper Dashboard</h1>
      
      {error && <div className="error">{error}</div>}
      
      {/* Stats Cards */}
      <div className="stats-grid">
        <div className="stat-card">
          <h3>{stats.totals.total}</h3>
          <p>Total Issues</p>
        </div>
        <div className="stat-card open">
          <h3>{stats.totals.open}</h3>
          <p>Open Issues</p>
        </div>
        <div className="stat-card resolved">
          <h3>{stats.totals.resolved}</h3>
          <p>Resolved Issues</p>
        </div>
      </div>
      
      {/* Chart */}
      <div className="card">
        <h2>Issues Over Time (Last 30 Days)</h2>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={stats.daily}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis 
                dataKey="date" 
                tickFormatter={formatChartDate}
                stroke="#666"
              />
              <YAxis stroke="#666" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #444' }}
                labelFormatter={formatChartDate}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="open" 
                stroke="#f5576c" 
                strokeWidth={2}
                dot={{ fill: '#f5576c' }}
              />
              <Line 
                type="monotone" 
                dataKey="resolved" 
                stroke="#00f2fe" 
                strokeWidth={2}
                dot={{ fill: '#00f2fe' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* Filters */}
      <div className="filters">
        <select 
          value={statusFilter} 
          onChange={(e) => setStatusFilter(e.target.value)}
        >
          <option value="">All Statuses</option>
          <option value="open">Open</option>
          <option value="resolved">Resolved</option>
        </select>
        
        <select 
          value={repoFilter} 
          onChange={(e) => setRepoFilter(e.target.value)}
        >
          <option value="">All Repos</option>
          {repos.map(repo => (
            <option key={repo} value={repo}>{repo}</option>
          ))}
        </select>
      </div>
      
      {/* Issues Table */}
      <div className="card">
        <h2>Issues ({issues.length})</h2>
        <table>
          <thead>
            <tr>
              <th>Gate</th>
              <th>Type</th>
              <th>Repo</th>
              <th>Status</th>
              <th>Details</th>
              <th>Created</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {issues.length === 0 ? (
              <tr>
                <td colSpan="7" style={{ textAlign: 'center', padding: '2rem' }}>
                  No issues found
                </td>
              </tr>
            ) : (
              issues.map(issue => (
                <tr key={issue.id}>
                  <td>
                    <span className={`gate-badge gate-${issue.gate}`}>
                      Gate {issue.gate}
                    </span>
                  </td>
                  <td>{issue.type}</td>
                  <td>{issue.repo}</td>
                  <td>
                    <span className={`status-badge ${issue.status}`}>
                      {issue.status}
                    </span>
                  </td>
                  <td>{issue.details || '-'}</td>
                  <td>{formatDate(issue.created_at)}</td>
                  <td>
                    <button 
                      className={`toggle-btn ${issue.status === 'open' ? 'resolve' : 'reopen'}`}
                      onClick={() => toggleStatus(issue.id)}
                    >
                      {issue.status === 'open' ? 'Resolve' : 'Reopen'}
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default App
