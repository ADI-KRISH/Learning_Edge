/**
 * Lightweight Markdown renderer — no external dependencies.
 * Supports: **bold**, *italic*, `code`, ```code blocks```,
 * # headings, - lists, > blockquotes, blank-line paragraphs.
 */
function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

function renderInline(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
}

export default function Markdown({ children }) {
  if (!children) return null

  const lines = children.split('\n')
  const elements = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]

    // Fenced code block
    if (line.startsWith('```')) {
      const lang = line.slice(3).trim()
      const codeLines = []
      i++
      while (i < lines.length && !lines[i].startsWith('```')) {
        codeLines.push(escapeHtml(lines[i]))
        i++
      }
      elements.push(
        <pre key={i} style={{ background: 'rgba(0,0,0,0.4)', padding: '12px 14px', borderRadius: 8, overflowX: 'auto', margin: '8px 0', fontSize: '0.83em' }}>
          <code dangerouslySetInnerHTML={{ __html: codeLines.join('\n') }} />
        </pre>
      )
      i++ // skip closing ```
      continue
    }

    // Custom RAG container block
    if (line.startsWith(':::rag')) {
      const title = line.slice(6).trim() || 'RAG Context'
      const ragLines = []
      i++
      while (i < lines.length && !lines[i].startsWith(':::')) {
        ragLines.push(escapeHtml(lines[i]))
        i++
      }
      elements.push(
        <details key={i} style={{ margin: '12px 0', padding: '10px', background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.2)', borderRadius: '8px' }}>
          <summary style={{ cursor: 'pointer', fontWeight: 'bold', color: 'var(--accent)' }}>{title}</summary>
          <pre style={{ marginTop: '10px', whiteSpace: 'pre-wrap', fontSize: '0.85em', color: 'var(--text-secondary)' }}>
            <code dangerouslySetInnerHTML={{ __html: ragLines.join('\n') }} />
          </pre>
        </details>
      )
      i++ // skip closing :::
      continue
    }

    // Headings
    if (/^#{1,4} /.test(line)) {
      const level = line.match(/^(#+)/)[1].length
      const text = line.replace(/^#+\s*/, '')
      const Tag = `h${Math.min(level + 1, 6)}`
      elements.push(
        <Tag key={i} style={{ margin: '10px 0 4px', fontWeight: 700 }}
          dangerouslySetInnerHTML={{ __html: renderInline(text) }} />
      )
      i++
      continue
    }

    // Horizontal rule
    if (/^---+$/.test(line.trim())) {
      elements.push(<hr key={i} style={{ border: 'none', borderTop: '1px solid var(--border)', margin: '10px 0' }} />)
      i++
      continue
    }

    // Blockquote
    if (line.startsWith('> ')) {
      const text = line.slice(2)
      elements.push(
        <blockquote key={i} style={{ borderLeft: '3px solid var(--accent)', paddingLeft: 12, margin: '4px 0', color: 'var(--text-secondary)', fontStyle: 'italic' }}
          dangerouslySetInnerHTML={{ __html: renderInline(text) }} />
      )
      i++
      continue
    }

    // Unordered list — collect consecutive items
    if (/^[-*] /.test(line)) {
      const items = []
      while (i < lines.length && /^[-*] /.test(lines[i])) {
        items.push(<li key={i} dangerouslySetInnerHTML={{ __html: renderInline(lines[i].replace(/^[-*] /, '')) }} />)
        i++
      }
      elements.push(<ul key={`ul-${i}`} style={{ paddingLeft: 20, margin: '4px 0' }}>{items}</ul>)
      continue
    }

    // Ordered list — collect consecutive items
    if (/^\d+\. /.test(line)) {
      const items = []
      while (i < lines.length && /^\d+\. /.test(lines[i])) {
        items.push(<li key={i} dangerouslySetInnerHTML={{ __html: renderInline(lines[i].replace(/^\d+\. /, '')) }} />)
        i++
      }
      elements.push(<ol key={`ol-${i}`} style={{ paddingLeft: 20, margin: '4px 0' }}>{items}</ol>)
      continue
    }

    // Empty line — paragraph break (skip)
    if (line.trim() === '') {
      elements.push(<div key={i} style={{ height: 6 }} />)
      i++
      continue
    }

    // Regular paragraph
    elements.push(
      <p key={i} style={{ margin: '2px 0' }}
        dangerouslySetInnerHTML={{ __html: renderInline(line) }} />
    )
    i++
  }

  return <div className="md-content">{elements}</div>
}
