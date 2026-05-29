export default function GraphPage() {
  return (
    <div className="iframe-page">
      <div className="page-header">
        <div className="page-title">🕸️ Concept Knowledge Graph</div>
        <div className="page-subtitle">
          Interactive prerequisite map of your study topics. Green = Mastered ✅
        </div>
      </div>
      <iframe
        className="full-iframe"
        src="/api/graph"
        title="Knowledge Graph"
        id="knowledge-graph-iframe"
      />
    </div>
  )
}
