import { useState, useEffect, useRef, useCallback } from 'react'
import { 
  getStatus, startOllama, getProfile, updateProfile, 
  deleteSession, renameSession, getSubjects, createSubject, 
  deleteSubject, getSubjectStudyState, uploadDocuments, ingestDocuments
} from '../api.js'

const STYLES = ['default', 'step-by-step', 'concise', 'detailed']
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
  page, setPage, quizHasNew,
  subjects, activeSubjectId, onSelectSubject, onRefreshSubjects
}) {
  const [status, setStatus] = useState({ ollama_running: false, chunk_count: 0, data_files: [] })
  const [profile, setProfile] = useState({ preferred_style: 'default', academic_level: 'Intermediate' })
  const [ollamaStarting, setOllamaStarting] = useState(false)
  const [renaming, setRenaming] = useState(false)
  const [newTitle, setNewTitle] = useState('')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [docsOpen, setDocsOpen] = useState(false)
  
  // Multi-subject management states
  const [subjectsOpen, setSubjectsOpen] = useState(true)
  const [newSubId, setNewSubId] = useState('')
  const [newSubName, setNewSubName] = useState('')
  const [subjectCreateOpen, setSubjectCreateOpen] = useState(false)

  // Curriculum division states
  const [curriculumOpen, setCurriculumOpen] = useState(true)
  const [ingestMode, setIngestMode] = useState('upload') // 'upload' | 'manual'
  const [uploadFiles, setUploadFiles] = useState([])
  const [uploading, setUploading] = useState(false)
  const [manualTitle, setManualTitle] = useState('')
  const [manualText, setManualText] = useState('')

  // Progression tracking state
  const [studyState, setStudyState] = useState(null)

  const fileRef = useRef()

  const loadStudyState = useCallback(async () => {
    if (activeSubjectId) {
      try {
        const res = await getSubjectStudyState(activeSubjectId)
        setStudyState(res)
      } catch (err) {
        console.error("Failed to load study state", err)
      }
    }
  }, [activeSubjectId])

  useEffect(() => {
    const load = async () => {
      const [s, p] = await Promise.all([getStatus(), getProfile()])
      setStatus(s)
      setProfile(p)
      await loadStudyState()
    }
    load()
    const interval = setInterval(async () => {
      const s = await getStatus()
      setStatus(s)
      await loadStudyState()
    }, 8000)
    return () => clearInterval(interval)
  }, [activeSubjectId, loadStudyState])

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

  const handleCreateSubject = async () => {
    if (!newSubId.trim() || !newSubName.trim()) return
    try {
      const res = await createSubject(newSubId.trim(), newSubName.trim())
      setNewSubId('')
      setNewSubName('')
      setSubjectCreateOpen(false)
      onRefreshSubjects()
      onSelectSubject(res.subject_id)
    } catch (err) {
      alert("Failed to create subject: " + err)
    }
  }

  const handleDeleteSubject = async (e, subId) => {
    e.stopPropagation()
    if (subId === 'default_subject') return
    if (!window.confirm(`Are you absolutely sure you want to delete this subject and all its RAG memory, progression states, and quiz logs?`)) return
    await deleteSubject(subId)
    onRefreshSubjects()
    onSelectSubject('default_subject')
  }

  const handleFileChange = (e) => setUploadFiles(Array.from(e.target.files))

  const handleUploadIngest = async () => {
    if (!uploadFiles.length) return
    setUploading(true)
    try {
      await uploadDocuments(uploadFiles, activeSubjectId)
      const res = await ingestDocuments({
        subject_id: activeSubjectId,
        file_name: uploadFiles[0].name
      })
      const s = await getStatus()
      setStatus(s)
      await loadStudyState()
      setUploadFiles([])
      if (res && res.session_id) {
        onDeleteSession(); // Refresh session list in parent state
        onSelectSession(res.session_id); // Select new session in parent state
      }
    } catch (err) {
      alert("Ingestion failed: " + err)
    } finally {
      setUploading(false)
    }
  }

  const handleManualIngest = async () => {
    if (!manualTitle.trim() || !manualText.trim()) return
    setUploading(true)
    try {
      const res = await ingestDocuments({
        subject_id: activeSubjectId,
        doc_title: manualTitle.trim(),
        manual_text: manualText.trim()
      })
      const s = await getStatus()
      setStatus(s)
      await loadStudyState()
      setManualTitle('')
      setManualText('')
      if (res && res.session_id) {
        onDeleteSession(); // Refresh session list in parent state
        onSelectSession(res.session_id); // Select new session in parent state
      }
    } catch (err) {
      alert("Manual ingestion failed: " + err)
    } finally {
      setUploading(false)
    }
  }

  const activeTitle = sessions.find(s => s.session_id === activeSessionId)?.title || ''

  return (
    <aside className="sidebar">
      {/* Logo Header */}
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <div className="sidebar-logo-icon">🧠</div>
          Offline AI Tutor
        </div>

        {/* New Chat Button */}
        <button className="btn btn-primary btn-full" id="new-chat-btn" onClick={onNewChat} disabled={!activeSessionId && sessions.length === 0 && !activeSubjectId}>
          ✚ New Chat
        </button>
      </div>

      <div className="sidebar-scroll">
        
        {/* 1. Subjects Collapsible Section */}
        <div className="sidebar-section">
          <div className="collapsible-header" onClick={() => setSubjectsOpen(o => !o)}>
            <span>📚 Subjects Manager</span>
            <span className={`collapsible-arrow ${subjectsOpen ? 'open' : ''}`}>▶</span>
          </div>
          {subjectsOpen && (
            <div style={{ padding: '6px 4px' }}>
              <div className="settings-group">
                <div className="settings-label">Active Subject</div>
                <select
                  className="settings-select"
                  value={activeSubjectId}
                  onChange={e => onSelectSubject(e.target.value)}
                  id="subject-select"
                >
                  {subjects.map(sub => (
                    <option key={sub.subject_id} value={sub.subject_id}>
                      {sub.subject_name}
                    </option>
                  ))}
                </select>
              </div>

              {!subjectCreateOpen ? (
                <button
                  className="btn btn-secondary btn-full btn-sm"
                  style={{ marginTop: 8 }}
                  onClick={() => setSubjectCreateOpen(true)}
                >
                  ➕ Create New Subject
                </button>
              ) : (
                <div style={{ marginTop: 8, padding: 8, background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)' }}>
                  <div className="settings-label">Subject ID (no spaces)</div>
                  <input
                    className="text-input"
                    value={newSubId}
                    onChange={e => setNewSubId(e.target.value)}
                    placeholder="e.g. distributed_systems"
                    style={{ marginBottom: 6, fontSize: '0.75rem' }}
                  />
                  <div className="settings-label">Subject Name</div>
                  <input
                    className="text-input"
                    value={newSubName}
                    onChange={e => setNewSubName(e.target.value)}
                    placeholder="e.g. Distributed Systems"
                    style={{ marginBottom: 8, fontSize: '0.75rem' }}
                  />
                  <div className="btn-row">
                    <button className="btn btn-primary btn-sm" style={{ flex: 1 }} onClick={handleCreateSubject}>Add</button>
                    <button className="btn btn-secondary btn-sm" style={{ flex: 1 }} onClick={() => setSubjectCreateOpen(false)}>Cancel</button>
                  </div>
                </div>
              )}

              {activeSubjectId !== 'default_subject' && (
                <button
                  className="btn btn-danger btn-full btn-sm"
                  style={{ marginTop: 6 }}
                  onClick={(e) => handleDeleteSubject(e, activeSubjectId)}
                >
                  🗑️ Delete Active Subject
                </button>
              )}
            </div>
          )}
        </div>

        <div className="divider" />

        {/* 2. Curriculum Ingest Collapsible Section */}
        <div className="sidebar-section">
          <div className="collapsible-header" onClick={() => setCurriculumOpen(o => !o)}>
            <span>📖 Ingest Syllabus / Doc</span>
            <span className={`collapsible-arrow ${curriculumOpen ? 'open' : ''}`}>▶</span>
          </div>
          {curriculumOpen && (
            <div style={{ padding: '6px 4px' }}>
              <div style={{ display: 'flex', gap: 4, marginBottom: 8 }}>
                <button
                  className={`btn btn-sm ${ingestMode === 'upload' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ flex: 1 }}
                  onClick={() => setIngestMode('upload')}
                >
                  📁 File Upload
                </button>
                <button
                  className={`btn btn-sm ${ingestMode === 'manual' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ flex: 1 }}
                  onClick={() => setIngestMode('manual')}
                >
                  ✍️ Manual Text
                </button>
              </div>

              {ingestMode === 'upload' ? (
                <>
                  <div className="upload-area" onClick={() => fileRef.current.click()}>
                    {uploadFiles.length > 0
                      ? `${uploadFiles.length} file(s) selected`
                      : 'Click to select PDF / TXT'}
                  </div>
                  <input
                    ref={fileRef}
                    type="file"
                    multiple
                    accept=".pdf,.txt"
                    style={{ display: 'none' }}
                    onChange={handleFileChange}
                  />
                  <button
                    className="btn btn-primary btn-full"
                    style={{ marginTop: 8 }}
                    disabled={!uploadFiles.length || uploading}
                    onClick={handleUploadIngest}
                  >
                    {uploading ? '⏳ Slicing & Chunking...' : '⚡ Split & Ingest File'}
                  </button>
                </>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  <input
                    className="text-input"
                    value={manualTitle}
                    onChange={e => setManualTitle(e.target.value)}
                    placeholder="Document/Syllabus Title"
                    style={{ fontSize: '0.75rem' }}
                  />
                  <textarea
                    className="text-input"
                    value={manualText}
                    onChange={e => setManualText(e.target.value)}
                    placeholder="Paste full curriculum / syllabus outline here..."
                    rows={4}
                    style={{ fontSize: '0.75rem', fontFamily: 'inherit', resize: 'vertical' }}
                  />
                  <button
                    className="btn btn-primary btn-full"
                    disabled={!manualTitle.trim() || !manualText.trim() || uploading}
                    onClick={handleManualIngest}
                  >
                    {uploading ? '⏳ Processing...' : '⚡ Split & Ingest Text'}
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="divider" />

        {/* 3. Study Progression Stepper Roadmap */}
        {studyState && studyState.parts && studyState.parts.length > 0 && (
          <div className="sidebar-section">
            <div className="sidebar-section-label" style={{ marginBottom: 8 }}>S6 Learning Progression</div>
            
            {/* Progress metric */}
            <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', padding: '0 6px 8px' }}>
              Progress: <strong>
                {Math.round(((studyState.active_part_index - (studyState.status === 'completed' ? 0 : 1)) / studyState.parts.length) * 100)}%
              </strong> ({studyState.status === 'completed' ? studyState.parts.length : studyState.active_part_index - 1} of {studyState.parts.length} Mastered)
            </div>

            {/* Stepper Roadmap list */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, padding: '0 4px' }}>
              {studyState.parts.map((p) => {
                const isCompleted = p.part_index < studyState.active_part_index || studyState.status === 'completed';
                const isActive = p.part_index === studyState.active_part_index && studyState.status !== 'completed';
                const isLocked = p.part_index > studyState.active_part_index && studyState.status !== 'completed';

                let dotIcon = '🔒';

                if (isCompleted) {
                  dotIcon = '✅';
                } else if (isActive) {
                  dotIcon = '⚡';
                }

                return (
                  <div
                    key={p.part_index}
                    title={`Part ${p.part_index}: ${p.part_title}\n\n${p.part_content || 'Content not loaded or locked.'}`}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      padding: '5px 8px',
                      background: isActive ? 'var(--accent-glow)' : 'transparent',
                      border: isActive ? '1px solid var(--border-active)' : '1px solid transparent',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: '0.74rem',
                      color: isLocked ? 'var(--text-muted)' : 'var(--text-primary)',
                      transition: 'all var(--t)',
                      cursor: 'help'
                    }}
                  >
                    <span>{dotIcon}</span>
                    <span style={{ flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      Part {p.part_index}: {p.part_title}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {studyState && studyState.parts && studyState.parts.length > 0 && <div className="divider" />}

        {/* 4. Chat Session History */}
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
              No chats in this subject yet.
            </div>
          )}
        </div>

        {/* Rename Session Action */}
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

        {/* 5. Navigation */}
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

        {/* 6. Personalization */}
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

        {/* 7. System Status */}
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
