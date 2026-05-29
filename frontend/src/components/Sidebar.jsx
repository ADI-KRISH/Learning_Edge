import { useState, useEffect, useRef } from 'react'
import { getStatus, startOllama, getProfile, updateProfile, deleteSession, renameSession } from '../api.js'

const STYLES = ['default', 'visual', 'step-by-step', 'analogy-based', 'example-heavy', 'concise', 'detailed']
const LEVELS = ['Beginner', 'Intermediate', 'Advanced']

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard', icon: '📊' },
  { id: 'learn', label: 'Learn', icon: '📚' },
  { id: 'quiz', label: 'Active Quiz', icon: '📝' },
  { id: 'graph', label: 'Knowledge Graph', icon: '🕸️' },
  { id: 'agent', label: 'Agent Architecture', icon: '🤖' },
]

export default function Sidebar({
  sessions, activeSessionId, onSelectSession,
  onNewChat, onDeleteSession, onRenameSession,
  page, setPage, quizHasNew
}) {
  const [status, setStatus] = useState({ ollama_running: false, chunk_count: 0, data_files: [] })
  const [profile, setProfile] = useState({ preferred_style: 'default', academic_level: 'Intermediate' })
  const [ollamaStarting, setOllamaStarting] = useState(false)
  const [renaming, setRenaming] = useState(false)
  const [newTitle, setNewTitle] = useState('')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [docsOpen, setDocsOpen] = useState(false)
  const [uploadFiles, setUploadFiles] = useState([])
  const [uploading, setUploading] = useState(false)
  const fileRef = useRef()

  useEffect(() => {
    const load = async () => {
      const [s, p] = await Promise.all([getStatus(), getProfile()])
      setStatus(s)
      setProfile(p)
    }
    load()
    const interval = setInterval(async () => {
      const s = await getStatus()
      setStatus(s)
    }, 8000)
    return () => clearInterval(interval)
  }, [])

  const handleStartOllama = async () => {
    setOllamaStarting(true)
    const res = await startOllama()
    setOllamaStarting(false)
    if (res.success) {
      const s = await getStatus()
      setStatus(s)
    }
  }

  const handleProfileChange = async (key, val) => {
    const updated = { ...profile, [key]: val }
    setProfile(updated)
    await updateProfile({ [key]: val })
  }

  const handleDelete = async (id) => {
    if (!window.confirm('Delete this chat session?')) return
    await deleteSession(id)
    onDeleteSession()
  }

  const handleRename = async () => {
    if (!newTitle.trim()) return
    await renameSession(activeSessionId, newTitle.trim())
    setRenaming(false)
    onRenameSession()
  }

  const handleFileChange = (e) => setUploadFiles(Array.from(e.target.files))

  const handleUploadIngest = async () => {
    if (!uploadFiles.length) return
    setUploading(true)
    const { uploadDocuments, ingestDocuments } = await import('../api.js')
    await uploadDocuments(uploadFiles)
    await ingestDocuments()
    const s = await getStatus()
    setStatus(s)
    setUploadFiles([])
    setUploading(false)
  }

  const activeTitle = sessions.find(s => s.session_id === activeSessionId)?.title || ''

  return (
    <aside className="sidebar">
      {/* Logo */}
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <div className="sidebar-logo-icon">🧠</div>
          Offline AI Tutor
        </div>

        {/* New Chat Button */}
        <button className="btn btn-primary btn-full" id="new-chat-btn" onClick={onNewChat}>
          ✚ New Chat
        </button>
      </div>

      <div className="sidebar-scroll">
        {/* Session List */}
        <div className="sidebar-section">
          <div className="sidebar-section-label">Chat History</div>
          {sessions.map(s => (
            <div
              key={s.session_id}
              className={`session-item ${s.session_id === activeSessionId ? 'active' : ''}`}
              onClick={() => onSelectSession(s.session_id)}
              id={`session-${s.session_id}`}
            >
              <span className="session-item-icon">💬</span>
              <span className="session-item-title">{s.title}</span>
              {s.session_id !== 'default_session' && (
                <button
                  className="session-item-del"
                  onClick={(e) => { e.stopPropagation(); handleDelete(s.session_id) }}
                  title="Delete session"
                >✕</button>
              )}
            </div>
          ))}
          {sessions.length === 0 && (
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', padding: '6px 10px' }}>
              No sessions yet
            </div>
          )}
        </div>

        {/* Rename */}
        {activeSessionId && activeSessionId !== 'default_session' && (
          <div className="sidebar-section">
            {!renaming ? (
              <button
                className="btn btn-secondary btn-full btn-sm"
                onClick={() => { setRenaming(true); setNewTitle(activeTitle) }}
              >
                ✏️ Rename Current Chat
              </button>
            ) : (
              <div>
                <div className="input-row" style={{ marginBottom: 5 }}>
                  <input
                    className="text-input"
                    value={newTitle}
                    onChange={e => setNewTitle(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleRename()}
                    autoFocus
                    placeholder="New title..."
                  />
                </div>
                <div className="btn-row">
                  <button className="btn btn-primary btn-sm" style={{ flex: 1 }} onClick={handleRename}>Save</button>
                  <button className="btn btn-secondary btn-sm" style={{ flex: 1 }} onClick={() => setRenaming(false)}>Cancel</button>
                </div>
              </div>
            )}
          </div>
        )}

        <div className="divider" />

        {/* Navigation */}
        <div className="sidebar-section">
          <div className="sidebar-section-label">Navigation</div>
          {NAV_ITEMS.map(item => (
            <button
              key={item.id}
              id={`nav-${item.id}`}
              className={`nav-item ${page === item.id ? 'active' : ''}`}
              onClick={() => setPage(item.id)}
            >
              <span>{item.icon}</span>
              {item.label}
              {item.id === 'quiz' && quizHasNew && (
                <span className="nav-item-badge">NEW</span>
              )}
            </button>
          ))}
        </div>

        <div className="divider" />

        {/* Personalization */}
        <div className="sidebar-section">
          <div
            className="collapsible-header"
            onClick={() => setSettingsOpen(o => !o)}
          >
            <span>🎯 Personalization</span>
            <span className={`collapsible-arrow ${settingsOpen ? 'open' : ''}`}>▶</span>
          </div>
          {settingsOpen && (
            <>
              <div className="settings-group">
                <div className="settings-label">Learning Style</div>
                <select
                  className="settings-select"
                  value={profile.preferred_style}
                  onChange={e => handleProfileChange('preferred_style', e.target.value)}
                  id="style-select"
                >
                  {STYLES.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
              <div className="settings-group">
                <div className="settings-label">Academic Level</div>
                <select
                  className="settings-select"
                  value={profile.academic_level}
                  onChange={e => handleProfileChange('academic_level', e.target.value)}
                  id="level-select"
                >
                  {LEVELS.map(l => <option key={l} value={l}>{l}</option>)}
                </select>
              </div>
            </>
          )}
        </div>

        <div className="divider" />

        {/* Documents */}
        <div className="sidebar-section">
          <div
            className="collapsible-header"
            onClick={() => setDocsOpen(o => !o)}
          >
            <span>📄 Upload Documents</span>
            <span className={`collapsible-arrow ${docsOpen ? 'open' : ''}`}>▶</span>
          </div>
          {docsOpen && (
            <div style={{ padding: '8px 4px' }}>
              {status.chunk_count > 0 && (
                <div className="alert alert-success" style={{ marginBottom: 8 }}>
                  📦 {status.chunk_count} chunks in vector DB
                </div>
              )}
              {status.data_files.length > 0 && (
                <div style={{ marginBottom: 8 }}>
                  {status.data_files.map(f => (
                    <div key={f} style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', padding: '1px 4px' }}>
                      • {f}
                    </div>
                  ))}
                </div>
              )}
              <div
                className="upload-area"
                onClick={() => fileRef.current.click()}
              >
                {uploadFiles.length > 0
                  ? `${uploadFiles.length} file(s) selected`
                  : 'Click to select PDF / TXT / DOCX'}
              </div>
              <input
                ref={fileRef}
                type="file"
                multiple
                accept=".pdf,.txt,.docx"
                style={{ display: 'none' }}
                onChange={handleFileChange}
              />
              <button
                className="btn btn-primary btn-full"
                style={{ marginTop: 8 }}
                disabled={!uploadFiles.length || uploading}
                onClick={handleUploadIngest}
                id="upload-ingest-btn"
              >
                {uploading ? '⏳ Processing...' : '⚡ Upload & Ingest'}
              </button>
            </div>
          )}
        </div>

        <div className="divider" />

        {/* System Status */}
        <div className="sidebar-section">
          <div className="sidebar-section-label">System Status</div>
          <div className="status-row">
            <div className="status-dot green" />
            LangGraph Orchestrator
          </div>
          <div className="status-row">
            <div className={`status-dot ${status.ollama_running ? 'green' : ollamaStarting ? 'yellow' : 'red'}`} />
            {status.ollama_running ? 'Ollama (llama3.2)' : ollamaStarting ? 'Ollama starting...' : 'Ollama Offline'}
          </div>
          {!status.ollama_running && !ollamaStarting && (
            <button
              className="btn btn-secondary btn-full btn-sm"
              style={{ marginTop: 6 }}
              onClick={handleStartOllama}
              id="start-ollama-btn"
            >
              🔌 Start Ollama
            </button>
          )}
        </div>
      </div>

      <div className="sidebar-footer">
        Offline AI Tutor • 100% local
      </div>
    </aside>
  )
}
