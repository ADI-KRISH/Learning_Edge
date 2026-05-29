import { useState, useEffect, useCallback } from 'react'
import Sidebar from './components/Sidebar.jsx'
import ChatPage from './components/ChatPage.jsx'
import DashboardPage from './components/DashboardPage.jsx'
import QuizPage from './components/QuizPage.jsx'
import GraphPage from './components/GraphPage.jsx'
import AgentPage from './components/AgentPage.jsx'
import { getSessions, createSession, getSubjects } from './api.js'

export default function App() {
  const [page, setPage] = useState('learn')
  const [sessions, setSessions] = useState([])
  const [activeSessionId, setActiveSessionId] = useState('default_session')
  const [activeQuiz, setActiveQuiz] = useState(null)  // { topic, questions, sessionId }
  const [quizHasNew, setQuizHasNew] = useState(false)
  const [subjects, setSubjects] = useState([])
  const [activeSubjectId, setActiveSubjectId] = useState('default_subject')
  const [chatRefreshKey, setChatRefreshKey] = useState(0) // bump to force ChatPage remount

  const refreshSubjects = useCallback(async () => {
    const data = await getSubjects()
    setSubjects(data)
  }, [])

  const refreshSessions = useCallback(async (selectId = null) => {
    const data = await getSessions(activeSubjectId)
    setSessions(data)
    if (selectId) {
      setActiveSessionId(selectId)
    } else if (data.length > 0 && !data.find(s => s.session_id === activeSessionId)) {
      setActiveSessionId(data[0].session_id)
    } else if (data.length === 0) {
      setActiveSessionId('')
    }
  }, [activeSessionId, activeSubjectId])

  useEffect(() => {
    refreshSubjects()
  }, [])

  useEffect(() => {
    refreshSessions()
  }, [activeSubjectId, refreshSessions])

  const handleNewChat = async () => {
    const s = await createSession('New Chat Session', activeSubjectId)
    await refreshSessions()
    setActiveSessionId(s.session_id)
    setPage('learn')
  }

  const handleQuizReady = (quiz) => {
    setActiveQuiz(quiz)
    setQuizHasNew(true)
  }

  const handleGoToQuiz = () => {
    setPage('quiz')
    setQuizHasNew(false)
  }

  const handleQuizPassed = async () => {
    // Refresh sessions (new messages were added by the backend), bump the key to force ChatPage remount
    await refreshSessions()
    setChatRefreshKey(k => k + 1)
    setPage('learn')
  }

  return (
    <div className="app-layout">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={(id) => { setActiveSessionId(id); setPage('learn') }}
        onNewChat={handleNewChat}
        onDeleteSession={async () => await refreshSessions()}
        onRenameSession={async () => await refreshSessions()}
        page={page}
        setPage={(p) => { setPage(p); if (p === 'quiz') setQuizHasNew(false) }}
        quizHasNew={quizHasNew}
        subjects={subjects}
        activeSubjectId={activeSubjectId}
        onSelectSubject={(id) => { setActiveSubjectId(id); setPage('learn') }}
        onRefreshSubjects={refreshSubjects}
      />
      <div className="main-content">
        {page === 'learn' && activeSessionId && (
          <ChatPage
            key={`${activeSessionId}-${chatRefreshKey}`}
            sessionId={activeSessionId}
            onQuizReady={handleQuizReady}
            onGoToQuiz={handleGoToQuiz}
            onSessionRenamed={refreshSessions}
          />
        )}
        {page === 'learn' && !activeSessionId && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-secondary)' }}>
            <h3>No Active Chats</h3>
            <p style={{ marginTop: 8 }}>Click "✚ New Chat" to start learning in this subject.</p>
          </div>
        )}
        {page === 'dashboard' && <DashboardPage activeSubjectId={activeSubjectId} />}
        {page === 'quiz' && (
          <QuizPage
            quiz={activeQuiz}
            sessionId={activeSessionId}
            onDismiss={() => setActiveQuiz(null)}
            onQuizPassed={handleQuizPassed}
          />
        )}
        {page === 'graph' && <GraphPage />}
        {page === 'agent' && <AgentPage />}
      </div>
    </div>
  )
}
