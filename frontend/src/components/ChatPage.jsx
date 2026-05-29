import { useState, useEffect, useRef, useCallback } from 'react'
import { getHistory, streamChat } from '../api.js'
import Markdown from './Markdown.jsx'

const PIPELINE_STEPS = [
  { id: 'supervisor', label: '🚦 Supervisor' },
  { id: 'researcher', label: '📚 Researcher' },
  { id: 'pedagogue', label: '🧠 Pedagogue' },
  { id: 'scribe', label: '✍️ Scribe' },
]

export default function ChatPage({ sessionId, onQuizReady, onGoToQuiz, onSessionRenamed }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [pipelineState, setPipelineState] = useState({ active: null, done: [] })
  const [showPipeline, setShowPipeline] = useState(false)
  const [error, setError] = useState(null)
  const bottomRef = useRef()
  const textareaRef = useRef()
  const cancelRef = useRef(null)

  useEffect(() => {
    const load = async () => {
      const hist = await getHistory(sessionId)
      setMessages(hist)
    }
    load()
  }, [sessionId])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = useCallback(() => {
    const query = input.trim()
    if (!query || isStreaming) return

    setInput('')
    setError(null)
    setIsStreaming(true)
    setShowPipeline(true)
    setPipelineState({ active: null, done: [] })

    // Optimistically add user message
    setMessages(prev => [...prev, { role: 'user', content: query }])

    let assistantContent = ''

    cancelRef.current = streamChat(query, sessionId, {
      onEvent: (event) => {
        if (event.type === 'session_renamed') {
          onSessionRenamed?.()
        } else if (event.type === 'node_complete') {
          setPipelineState(prev => ({
            active: null,
            done: [...prev.done, event.node]
          }))
        } else if (event.type === 'pipeline_start') {
          setPipelineState({ active: 'supervisor', done: [] })
        } else if (event.type === 'tutor_response') {
          assistantContent = event.content
          setMessages(prev => {
            // Replace streaming placeholder if any, else append
            const last = prev[prev.length - 1]
            if (last?.role === 'assistant' && last?._streaming) {
              return [...prev.slice(0, -1), { role: 'assistant', content: event.content }]
            }
            return [...prev, { role: 'assistant', content: event.content }]
          })
        } else if (event.type === 'quiz_ready') {
          onQuizReady?.({
            topic: event.topic,
            questions: event.questions,
            sessionId
          })
          const noticeMsg = `✅ Quiz ready on **${event.topic}**! Click **Active Quiz** in the sidebar to start.`
          setMessages(prev => [...prev, { role: 'assistant', content: noticeMsg }])
        } else if (event.type === 'error') {
          setError(event.content)
          setIsStreaming(false)
          setShowPipeline(false)
        }
      },
      onDone: () => {
        setIsStreaming(false)
        setPipelineState(prev => ({ ...prev, active: null }))
        setTimeout(() => setShowPipeline(false), 2000)
      },
      onError: (err) => {
        setError('Connection error. Is the backend running?')
        setIsStreaming(false)
        setShowPipeline(false)
      }
    })
  }, [input, isStreaming, sessionId, onQuizReady, onSessionRenamed])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleInput = (e) => {
    setInput(e.target.value)
    // Auto-resize textarea
    const ta = textareaRef.current
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 140) + 'px'
  }

  return (
    <div className="chat-container">
      <div className="page-header">
        <div className="page-title">📚 Tutor Agent Workspace</div>
        <div className="page-subtitle">Ask questions to get personalized explanations. Add <em>"quiz me"</em> to get a quiz!</div>
      </div>

      {/* Pipeline Visualizer */}
      {showPipeline && (
        <div className="pipeline-bar">
          {PIPELINE_STEPS.map((step, i) => {
            const isDone = pipelineState.done.includes(step.id)
            const isActive = pipelineState.active === step.id ||
              (!isDone && pipelineState.done.length === i && isStreaming)
            return (
              <span key={step.id} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span className={`pipeline-step ${isDone ? 'done' : isActive ? 'active' : ''}`}>
                  {isDone ? '✓ ' : isActive ? <span className="spinner" style={{ width: 10, height: 10 }} /> : null}
                  {step.label}
                </span>
                {i < PIPELINE_STEPS.length - 1 && <span className="pipeline-arrow">→</span>}
              </span>
            )
          })}
        </div>
      )}

      {/* Messages */}
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-state-icon">🤖</div>
            <div className="empty-state-text">
              Ask me anything about your study materials. I'm powered by Ollama + LangGraph and running 100% offline.
              <br /><br />
              <strong>Tips:</strong> Add "quiz me" at the end to take a quiz on any topic!
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-avatar">
              {msg.role === 'user' ? '👤' : '🤖'}
            </div>
            <div className="message-bubble">
              {msg.role === 'assistant'
                ? <Markdown>{msg.content}</Markdown>
                : msg.content
              }
            </div>
          </div>
        ))}

        {isStreaming && messages[messages.length - 1]?.role === 'user' && (
          <div className="message assistant">
            <div className="message-avatar">🤖</div>
            <div className="message-bubble" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div className="spinner" />
              <span style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>Thinking...</span>
            </div>
          </div>
        )}

        {error && (
          <div className="alert alert-error">
            ⚠️ {error}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="chat-input-area">
        <div className="chat-input-row">
          <textarea
            ref={textareaRef}
            className="chat-input-field"
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Ask your tutor a question... (Shift+Enter for new line)"
            rows={1}
            disabled={isStreaming}
            id="chat-input"
          />
          <button
            className="chat-send-btn"
            onClick={handleSend}
            disabled={!input.trim() || isStreaming}
            id="chat-send-btn"
            title="Send message"
          >
            {isStreaming ? <div className="spinner" style={{ borderTopColor: 'white' }} /> : '➤'}
          </button>
        </div>
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 6, paddingLeft: 4 }}>
          Enter to send • Shift+Enter for new line
        </div>
      </div>
    </div>
  )
}
