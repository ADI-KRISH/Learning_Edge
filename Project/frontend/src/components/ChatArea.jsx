import React, { useState, useRef, useEffect } from 'react';
import { Send, User, Sparkles, ShieldCheck, Loader2 } from 'lucide-react';

const ChatArea = () => {
  const [messages, setMessages] = useState([
    { id: 1, role: 'tutor', content: 'Hello! I am your AI-Powered Tutor. Upload your study resources in the sidebar, and ask me anything about the material. I will provide grounded, hallucination-free explanations.', grounded: false }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { id: Date.now(), role: 'user', content: input, grounded: false };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Simulate API call to FastAPI backend
    setTimeout(() => {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'tutor',
        content: `Based on your loaded materials, here is an explanation for your question. The mechanism underlying this phenomenon largely dictates its trajectory. Let me break it down further...`,
        grounded: true, // Grounding badge flag
        documentRef: 'Lecture_Notes_Ch4.pdf'
      }]);
      setIsLoading(false);
    }, 1800);
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-50/50">
      {/* Header */}
      <header className="h-16 border-b border-border bg-white flex flex-col justify-center px-6 shadow-sm z-10 flex-shrink-0">
        <h2 className="font-semibold text-text-main flex items-center gap-2">
          The Tutor <Sparkles size={16} className="text-primary-500" />
        </h2>
        <p className="text-xs text-text-muted">Teacher-style explanations grounded in your material.</p>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 scrollbar-thin">
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.map((msg) => (
            <div key={msg.id} className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
              {/* Avatar */}
              <div className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0
                ${msg.role === 'tutor' ? 'bg-primary-100 text-primary-600' : 'bg-slate-200 text-slate-600'}`}>
                {msg.role === 'tutor' ? <Sparkles size={18} /> : <User size={18} />}
              </div>
              
              {/* Content Bubble */}
              <div className={`max-w-[75%] rounded-2xl px-5 py-4 shadow-sm border
                ${msg.role === 'tutor' ? 'bg-white border-border' : 'bg-primary-600 text-white border-primary-600'}`}>
                <p className={`text-sm leading-relaxed ${msg.role === 'user' ? 'text-white' : 'text-text-main'}`}>
                  {msg.content}
                </p>
                
                {/* Grounding Indicator */}
                {msg.role === 'tutor' && msg.grounded && (
                  <div className="mt-3 pt-3 border-t border-slate-100 flex items-center gap-1.5 text-emerald-600 font-medium text-xs">
                    <ShieldCheck size={14} />
                    <span>Source Verified Grounding</span>
                    <span className="text-slate-400 font-normal ml-1">• {msg.documentRef}</span>
                  </div>
                )}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex gap-4">
               <div className="w-10 h-10 rounded-full bg-primary-100 text-primary-600 flex items-center justify-center flex-shrink-0">
                <Sparkles size={18} />
              </div>
              <div className="bg-white border border-border rounded-2xl px-5 py-4 shadow-sm flex items-center gap-2 text-sm text-text-muted">
                <Loader2 size={16} className="animate-spin text-primary-500" />
                Synthesizing answer from vector DB...
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="p-4 bg-white border-t border-border flex-shrink-0">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto relative flex items-end gap-2 bg-background border border-border rounded-2xl p-2 focus-within:ring-2 focus-within:ring-primary-500/20 focus-within:border-primary-500 transition-all">
          <textarea 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask your tutor a question..."
            className="flex-1 bg-transparent resize-none outline-none max-h-32 min-h-[44px] py-3 px-3 text-sm text-text-main disabled:opacity-50"
            rows={1}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
            disabled={isLoading}
          />
          <button 
            type="submit" 
            disabled={!input.trim() || isLoading}
            className="w-11 h-11 flex-shrink-0 bg-primary-600 hover:bg-primary-700 disabled:bg-primary-300 text-white rounded-xl flex items-center justify-center transition-colors mb-0.5"
          >
            <Send size={18} className="translate-x-px translate-y-px" />
          </button>
        </form>
        <p className="text-center text-[10px] text-text-muted mt-3">
          AI generated responses may contain inaccuracies. Verify with source documents.
        </p>
      </div>
    </div>
  );
};

export default ChatArea;
