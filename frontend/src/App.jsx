import { useState, useEffect, useCallback } from 'react'
import Sidebar from './components/Sidebar.jsx'
import ChatPage from './components/ChatPage.jsx'
import DashboardPage from './components/DashboardPage.jsx'
import QuizPage from './components/QuizPage.jsx'
import GraphPage from './components/GraphPage.jsx'
import AgentPage from './components/AgentPage.jsx'
import { getSessions, createSession } from './api.js'

export default function App() {
  const [page, setPage] = useState('learn')
  const [sessions, setSessions] = useState([])
  const [activeSessionId, setActiveSessionId] = useState('default_session')
  const [activeQuiz, setActiveQuiz] = useState(null)  // { topic, questions, sessionId }
  const [quizHasNew, setQuizHasNew] = useState(false)

  const refreshSessions = useCallback(async () => {
    const data = await getSessions()
    setSessions(data)
    // If active session was deleted, fall back to first available
    if (data.length > 0 && !data.find(s => s.session_id === activeSessionId)) {
      setActiveSessionId(data[0].session_id)
    }
  }, [activeSessionId])

  useEffect(() => {
    refreshSessions()
  }, [])

  const handleNewChat = async () => {
    const s = await createSession('New Chat Session')
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
      />
      <div className="main-content">
        {page === 'learn' && (
          <ChatPage
            key={activeSessionId}
            sessionId={activeSessionId}
            onQuizReady={handleQuizReady}
            onGoToQuiz={handleGoToQuiz}
            onSessionRenamed={refreshSessions}
          />
        )}
        {page === 'dashboard' && <DashboardPage />}
        {page === 'quiz' && (
          <QuizPage
            quiz={activeQuiz}
            sessionId={activeSessionId}
            onDismiss={() => setActiveQuiz(null)}
          />
        )}
        {page === 'graph' && <GraphPage />}
        {page === 'agent' && <AgentPage />}
      </div>
    </div>
  )
}
