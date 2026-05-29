export default function AgentPage() {
  return (
    <div className="iframe-page">
      <div className="page-header">
        <div className="page-title">🤖 Agentic Architecture</div>
        <div className="page-subtitle">The LangGraph state machine orchestrating the AI Tutor pipeline</div>
      </div>

      <iframe
        className="full-iframe"
        src="/api/agent_graph"
        title="Agent Architecture"
        id="agent-graph-iframe"
        style={{ flex: '0 0 560px' }}
      />

      <div style={{ padding: '20px 28px', overflowY: 'auto', flex: 1 }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Node</th>
              <th>Role</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>🚦 <strong>Supervisor</strong></td>
              <td>Non-linear Git-style graph memory. Embeds query, decides Commit / Branch / Merge / Checkout operation, and routes to Researcher or Pedagogue.</td>
            </tr>
            <tr>
              <td>📚 <strong>Researcher</strong></td>
              <td>Hybrid RAG Retrieval using Vector Search + BM25 keyword search over your uploaded documents in ChromaDB.</td>
            </tr>
            <tr>
              <td>🧠 <strong>Pedagogue</strong></td>
              <td>Generates personalized explanations using Ollama (llama3.2) conditioned on your profile, history, and RAG context.</td>
            </tr>
            <tr>
              <td>✍️ <strong>Scribe</strong></td>
              <td>Distils the interaction into a minified JSON state, labels the graph node, and persists the DiGraph to SQLite.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}
