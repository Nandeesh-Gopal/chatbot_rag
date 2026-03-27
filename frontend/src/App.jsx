import { useState, useRef, useEffect } from 'react'
import './index.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const API_URL = `${API_BASE.replace(/\/$/, '')}/chat`

function TypingIndicator() {
  return (
    <div className="message system">
      <div className="avatar system-avatar">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="16" height="16"><rect x="2" y="3" width="20" height="14" rx="2" /><path d="M8 21h8M12 17v4" /><circle cx="12" cy="10" r="3" /></svg>
      </div>
      <div className="bubble">
        <div className="typing-indicator">
          <span /><span /><span />
        </div>
      </div>
    </div>
  )
}

function Message({ content, sender }) {
  return (
    <div className={`message ${sender}`}>
      {sender === 'system' && (
        <div className="avatar system-avatar">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="16" height="16"><rect x="2" y="3" width="20" height="14" rx="2" /><path d="M8 21h8M12 17v4" /><circle cx="12" cy="10" r="3" /></svg>
        </div>
      )}
      <div className="bubble">{content}</div>
      {sender === 'user' && (
        <div className="avatar user-avatar">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="16" height="16"><circle cx="12" cy="8" r="4" /><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" /></svg>
        </div>
      )}
    </div>
  )
}

export default function App() {
  const [messages, setMessages] = useState([
    { id: 0, sender: 'system', content: "Hello! I'm connected to your document index. Ask me anything about the data." }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const chatEndRef = useRef(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleSubmit = async (e) => {
    e.preventDefault()
    const query = input.trim()
    if (!query || loading) return

    const userMsg = { id: Date.now(), sender: 'user', content: query }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      })
      const data = await res.json()
      const answer = res.ok ? data.answer : (data?.detail || 'Sorry, an error occurred talking to the API.')
      const sources = Array.isArray(data?.sources) ? data.sources : []
      const sourceText = sources.length ? `\n\nSources:\n- ${sources.join('\n- ')}` : ''
      setMessages(prev => [...prev, { id: Date.now() + 1, sender: 'system', content: `${answer}${sourceText}` }])
    } catch {
      setMessages(prev => [...prev, {
        id: Date.now() + 1, sender: 'system',
        content: 'Network error. Make sure the FastAPI backend is running on port 8000.'
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <div className="glow top-left" />
      <div className="glow bottom-right" />

      <div className="chat-panel">
        {/* Header */}
        <header className="chat-header">
          <div className="header-left">
            <div className="header-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="22" height="22"><rect x="2" y="3" width="20" height="14" rx="2" /><path d="M8 21h8M12 17v4" /><circle cx="12" cy="10" r="3" /></svg>
            </div>
            <div>
              <h1>Nova AI</h1>
              <p className="status"><span className="dot" />RAG System Active</p>
            </div>
          </div>
        </header>

        {/* Messages */}
        <div className="messages">
          {messages.map(m => (
            <Message key={m.id} content={m.content} sender={m.sender} />
          ))}
          {loading && <TypingIndicator />}
          <div ref={chatEndRef} />
        </div>

        {/* Input */}
        <form className="input-form" onSubmit={handleSubmit}>
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask a question about your documents..."
            disabled={loading}
          />
          <button type="submit" className="send-btn" disabled={loading || !input.trim()}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="18" height="18"><path d="m22 2-7 20-4-9-9-4Z" /><path d="M22 2 11 13" /></svg>
          </button>
        </form>
      </div>
    </div>
  )
}
