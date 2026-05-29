import { useState } from 'react'
import { submitQuiz } from '../api.js'

export default function QuizPage({ quiz, sessionId, onDismiss, onQuizPassed }) {
  const [answers, setAnswers] = useState({})
  const [result, setResult] = useState(null)   // { score, total, passed }
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async () => {
    if (!quiz) return
    const userAnswers = quiz.questions.map((q, i) => answers[i] ?? null)
    setSubmitting(true)
    const res = await submitQuiz({
      session_id: sessionId,
      topic: quiz.topic,
      questions: quiz.questions,
      user_answers: userAnswers
    })
    setResult(res)
    setSubmitting(false)
    // If the quiz was passed, notify the parent so it can navigate to chat + refresh
    if (res.passed) {
      onQuizPassed?.()
    }
  }

  const handleRetake = () => {
    setAnswers({})
    setResult(null)
  }

  if (!quiz) {
    return (
      <>
        <div className="page-header">
          <div className="page-title">📝 Active Quiz Workspace</div>
          <div className="page-subtitle">Objective mastery challenges based on your study materials</div>
        </div>
        <div className="empty-state">
          <div className="empty-state-icon">📝</div>
          <div className="empty-state-text">
            No active quiz. Go to <strong>Learn</strong> and say
            <em>"quiz me on [topic]"</em> to generate an interactive challenge.
          </div>
        </div>
      </>
    )
  }

  const allAnswered = quiz.questions.every((_, i) => answers[i] !== undefined)

  return (
    <>
      <div className="page-header">
        <div className="page-title">📝 Quiz: {quiz.topic}</div>
        <div className="page-subtitle">
          {result
            ? `Scored ${result.score}/${result.total} — ${result.passed ? 'PASSED 🎉' : 'FAILED — keep studying!'}`
            : `${quiz.questions.length} questions • Select your answers below`
          }
        </div>
      </div>

      <div className="quiz-page">
        {/* Result Banner */}
        {result && (
          <div className={`quiz-result-banner ${result.passed ? 'passed' : 'failed'}`}>
            {result.passed
              ? `🎉 Verified Mastery! Score: ${result.score}/${result.total} (${Math.round(result.score/result.total*100)}%)`
              : `📖 Score too low: ${result.score}/${result.total} — ${quiz.topic} flagged as a focus area`
            }
          </div>
        )}

        {/* Questions */}
        {quiz.questions.map((q, qi) => {
          const correctIdx = typeof q.correct === 'number' && q.correct >= 0 && q.correct < q.options.length
            ? q.correct : 0
          const correctAnswer = q.options[correctIdx]
          const userAnswer = answers[qi]

          return (
            <div key={qi} className="quiz-card">
              <div className="quiz-question">Q{qi + 1}: {q.question}</div>

              {q.options.map((opt, oi) => {
                let cls = 'quiz-option'
                if (result) {
                  if (opt === correctAnswer) cls += ' correct'
                  else if (opt === userAnswer && opt !== correctAnswer) cls += ' wrong'
                } else if (userAnswer === opt) {
                  cls += ' selected'
                }

                return (
                  <label key={oi} className={cls} style={{ cursor: result ? 'default' : 'pointer' }}>
                    <input
                      type="radio"
                      name={`q-${qi}`}
                      value={opt}
                      checked={userAnswer === opt}
                      disabled={!!result}
                      onChange={() => !result && setAnswers(a => ({ ...a, [qi]: opt }))}
                    />
                    {opt}
                  </label>
                )
              })}

              {/* Show explanation after submit */}
              {result && q.explanation && (
                <div className="quiz-explanation">
                  💡 {q.explanation}
                </div>
              )}
            </div>
          )
        })}

        {/* Action Buttons */}
        <div className="btn-row" style={{ marginTop: 8 }}>
          {!result ? (
            <button
              className="btn btn-primary"
              style={{ flex: 1 }}
              onClick={handleSubmit}
              disabled={!allAnswered || submitting}
              id="quiz-submit-btn"
            >
              {submitting ? '⏳ Submitting...' : '✅ Submit Answers'}
            </button>
          ) : (
            <>
              {!result.passed && (
                <button
                  className="btn btn-secondary"
                  style={{ flex: 1 }}
                  onClick={handleRetake}
                  id="quiz-retake-btn"
                >
                  🔄 Retake Quiz
                </button>
              )}
              <button
                className="btn btn-danger"
                style={{ flex: 1 }}
                onClick={onDismiss}
                id="quiz-dismiss-btn"
              >
                Dismiss Quiz
              </button>
            </>
          )}
        </div>

        {!allAnswered && !result && (
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 8, textAlign: 'center' }}>
            Please answer all {quiz.questions.length} questions before submitting
          </div>
        )}
      </div>
    </>
  )
}
