import { useState, useEffect } from 'react'
import { getProfile, getQuizHistory, getSubjectsProgress } from '../api.js'

export default function DashboardPage({ activeSubjectId }) {
  const [profile, setProfile] = useState({ preferred_style: '', academic_level: '', weak_topics: [], completed_topics: [] })
  const [quizHistory, setQuizHistory] = useState([])
  const [subProgress, setSubProgress] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [p, qh, progress] = await Promise.all([
          getProfile(activeSubjectId), 
          getQuizHistory(activeSubjectId),
          getSubjectsProgress()
        ])
        setProfile(p)
        setQuizHistory(qh)
        setSubProgress(progress)
      } catch (err) {
        console.error("Dashboard failed to load", err)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [activeSubjectId])

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
        {/* 🏫 Subject Progression Overview */}
        <div className="section-card" style={{ marginBottom: 20 }}>
          <div className="section-title">🏫 Subject Mastery & Progression Overview</div>
          <div className="section-caption">Track your completion, active chapters, and performance statistics across all subjects</div>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 16, marginTop: 12 }}>
            {subProgress.map(sub => {
              const isCompleted = sub.status === 'completed'
              const hasCurriculum = sub.total_parts > 0
              const isActive = sub.subject_id === activeSubjectId
              
              return (
                <div 
                  key={sub.subject_id}
                  style={{
                    padding: 16,
                    background: 'var(--bg-elevated)',
                    borderRadius: 'var(--radius-md)',
                    border: isActive ? '1px solid var(--border-active)' : '1px solid var(--border)',
                    boxShadow: isActive ? '0 4px 20px rgba(99,102,241,.15)' : 'none',
                    position: 'relative',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 10
                  }}
                >
                  {/* Subject Name & Indicator */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-primary)' }}>
                      📚 {sub.subject_name}
                    </div>
                    {isActive && (
                      <span 
                        style={{
                          fontSize: '0.65rem',
                          fontWeight: 700,
                          background: 'var(--accent-glow)',
                          color: 'var(--accent-hover)',
                          padding: '2px 8px',
                          borderRadius: 10,
                          border: '1px solid var(--border-active)'
                        }}
                      >
                        ACTIVE
                      </span>
                    )}
                  </div>
                  
                  {/* Active Subtopic */}
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)' }}>
                    {hasCurriculum ? (
                      isCompleted ? '🎉 Syllabus Fully Mastered!' : `⚡ Studying Part ${sub.active_part_index}: ${sub.active_part_title}`
                    ) : (
                      '⚠️ No syllabus ingested yet. Upload text or PDF!'
                    )}
                  </div>
                  
                  {/* Progress Bar */}
                  {hasCurriculum && (
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: 'var(--text-muted)', marginBottom: 4 }}>
                        <span>{sub.mastered_parts_count} of {sub.total_parts} Mastered</span>
                        <span>{sub.progress_percent}%</span>
                      </div>
                      <div style={{ width: '100%', height: 8, background: 'rgba(255,255,255,.05)', borderRadius: 4, overflow: 'hidden' }}>
                        <div 
                          style={{
                            width: `${sub.progress_percent}%`,
                            height: '100%',
                            background: isCompleted ? 'linear-gradient(90deg, var(--green), #4ade80)' : 'linear-gradient(90deg, var(--accent), #818cf8)',
                            borderRadius: 4,
                            transition: 'width .4s ease'
                          }}
                        />
                      </div>
                    </div>
                  )}
                  
                  <div className="divider" style={{ margin: '4px 0' }} />
                  
                  {/* Stat Badges */}
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, fontSize: '0.7rem' }}>
                    <span style={{ padding: '2px 6px', borderRadius: 4, background: 'var(--green-dim)', color: 'var(--green)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                      🎓 {sub.completed_concepts} Mastered
                    </span>
                    <span style={{ padding: '2px 6px', borderRadius: 4, background: sub.weak_concepts > 0 ? 'var(--red-dim)' : 'rgba(255,255,255,.05)', color: sub.weak_concepts > 0 ? 'var(--red)' : 'var(--text-secondary)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                      📖 {sub.weak_concepts} Focus Area{sub.weak_concepts !== 1 ? 's' : ''}
                    </span>
                    {sub.quiz_count > 0 && (
                      <span style={{ padding: '2px 6px', borderRadius: 4, background: 'rgba(99,102,241,.1)', color: 'var(--accent-hover)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                        🎯 Quiz Avg: {sub.quiz_average}%
                      </span>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
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
