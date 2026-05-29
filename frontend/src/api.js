const BASE = '/api'

export async function getSessions() {
  const r = await fetch(`${BASE}/sessions`)
  return r.json()
}

export async function createSession(title = 'New Chat Session') {
  const r = await fetch(`${BASE}/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title })
  })
  return r.json()
}

export async function deleteSession(id) {
  await fetch(`${BASE}/sessions/${id}`, { method: 'DELETE' })
}

export async function renameSession(id, title) {
  await fetch(`${BASE}/sessions/${id}/title`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title })
  })
}

export async function getHistory(sessionId) {
  const r = await fetch(`${BASE}/sessions/${sessionId}/history`)
  return r.json()
}

export async function getGraphTree(sessionId) {
  const r = await fetch(`${BASE}/sessions/${sessionId}/graph_tree`)
  return r.json()
}

export async function getProfile() {
  const r = await fetch(`${BASE}/profile`)
  return r.json()
}

export async function updateProfile(data) {
  const r = await fetch(`${BASE}/profile`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  })
  return r.json()
}

export async function getStatus() {
  const r = await fetch(`${BASE}/status`)
  return r.json()
}

export async function startOllama() {
  const r = await fetch(`${BASE}/status/start_ollama`, { method: 'POST' })
  return r.json()
}

export async function getQuizHistory() {
  const r = await fetch(`${BASE}/quiz/history`)
  return r.json()
}

export async function submitQuiz(data) {
  const r = await fetch(`${BASE}/quiz/submit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  })
  return r.json()
}

export async function uploadDocuments(files) {
  const form = new FormData()
  for (const f of files) form.append('files', f)
  const r = await fetch(`${BASE}/documents/upload`, { method: 'POST', body: form })
  return r.json()
}

export async function ingestDocuments() {
  const r = await fetch(`${BASE}/documents/ingest`, { method: 'POST' })
  return r.json()
}

/**
 * Opens a streaming POST to /api/chat and calls callbacks for each SSE event.
 * Returns a cancel function.
 */
export function streamChat(query, sessionId, callbacks) {
  const controller = new AbortController()

  fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, session_id: sessionId }),
    signal: controller.signal
  }).then(async (res) => {
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      const parts = buffer.split('\n\n')
      buffer = parts.pop()

      for (const part of parts) {
        if (part.startsWith('data: ')) {
          try {
            const event = JSON.parse(part.slice(6))
            callbacks.onEvent?.(event)
          } catch (_) { /* ignore parse errors */ }
        }
      }
    }
    callbacks.onDone?.()
  }).catch(err => {
    if (err.name !== 'AbortError') callbacks.onError?.(err)
  })

  return () => controller.abort()
}
