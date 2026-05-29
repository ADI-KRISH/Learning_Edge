import { useState, useEffect } from 'react'
import { getProfile, getQuizHistory } from '../api.js'

export default function DashboardPage() {
  const [profile, setProfile] = useState({ preferred_style: '', academic_level: '', weak_topics: [], completed_topics: [] })
  const [quizHistory, setQuizHistory] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = async () => {
      const [p, qh] = await Promise.all([getProfile(), getQuizHistory()])
      setProfile(p)
      setQuizHistory(qh)
      setLoading(false)
    }
    load()
  }, [])

  if (loading) {
    return (
      <div className="dashboard-page" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div className="spinner" />
      </div>
    )
  }

  const mastered = profile.completed_topics || []
  const weak = profile.weak_topics || []
  const total = quizHistory.length
  const passed = quizHistory.filter(q => q.passed).length
  const passRate = total > 0 ? Math.round((passed / total) * 100) + '%' : 'N/A'

  return (
    <>
      <div className="page-header">
        <div className="page-title">📊 Learner Dashboard</div>
        <div className="page-subtitle">Your verified academic progress and performance metrics</div>
      </div>

      <div className="dashboard-page">
        {/* Metrics */}
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-label">🏆 Verified Mastered Concepts</div>
            <div className="metric-value">{mastered.length}</div>
            <div className="metric-delta">Passed quiz ≥ 66%</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">📖 Concepts in Review</div>
            <div className="metric-value">{weak.length}</div>
            <div className="metric-delta">Need more study</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">🎯 Quiz Pass Rate</div>
            <div className="metric-value">{passRate}</div>
            <div className="metric-delta">{passed}/{total} passed</div>
          </div>
        </div>

        {/* Mastery + Weak Topics */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
          <div className="section-card">
            <div className="section-title">✅ Verified Mastery</div>
            <div className="section-caption">Topics you've passed quizzes on</div>
            {mastered.length > 0
              ? mastered.map(t => (
                <span key={t} className="topic-tag mastered">🎓 {t}</span>
              ))
              : <div className="alert alert-info">No topics mastered yet. Ask the tutor to quiz you!</div>
            }
          </div>
          <div className="section-card">
            <div className="section-title">⚠️ Focus Areas</div>
            <div className="section-caption">Topics that need more practice</div>
            {weak.length > 0
              ? weak.map(t => (
                <span key={t} className="topic-tag weak">📖 {t}</span>
              ))
              : <div className="alert alert-success">All caught up! No weak topics on record.</div>
            }
          </div>
        </div>

        {/* Quiz History Table */}
        <div className="section-card">
          <div className="section-title">📋 Quiz History Log</div>
          <div className="section-caption">All recorded quiz attempts from your SQLite database</div>
          {quizHistory.length > 0 ? (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Topic</th>
                  <th>Score</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {quizHistory.map((q, i) => (
                  <tr key={i}>
                    <td>{q.timestamp?.split(' ')[0] || '—'}</td>
                    <td>{q.topic}</td>
                    <td>{q.score}/{q.total}</td>
                    <td>
                      <span style={{ color: q.passed ? 'var(--green)' : 'var(--red)', fontWeight: 500 }}>
                        {q.passed ? '🟢 PASSED' : '🔴 FAILED'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="empty-state" style={{ padding: '30px 0' }}>
              <div className="empty-state-text">No quiz records yet. Go to Learn and ask to be quizzed on a topic!</div>
            </div>
          )}
        </div>
      </div>
    </>
  )
}
