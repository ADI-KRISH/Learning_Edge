import React, { useState } from 'react';
import { UploadCloud, BookOpen, Settings2, FileText, X, CheckCircle } from 'lucide-react';

const Sidebar = ({ setFiles, files, models, selectedModel, setSelectedModel }) => {
  const [isDragActive, setIsDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      // Add fake processing states
      const newFile = {
        id: Date.now(),
        name: e.dataTransfer.files[0].name,
        status: 'extracting', 
      };
      setFiles(prev => [...prev, newFile]);
      
      // Simulate backend processing
      setTimeout(() => {
        setFiles(prev => prev.map(f => f.id === newFile.id ? { ...f, status: 'ready' } : f));
      }, 2500);
    }
  };

  const removeFile = (id) => {
    setFiles(files.filter(f => f.id !== id));
  };

  return (
    <aside className="w-80 bg-surface border-r border-border h-full flex flex-col pt-6 pb-4 px-4 shadow-sm z-10 flex-shrink-0">
      <div className="flex items-center gap-2 mb-8 px-2">
        <div className="bg-primary-500 rounded p-2 text-white">
          <BookOpen size={20} />
        </div>
        <h1 className="font-semibold text-lg text-text-main">AI Tutor Portal</h1>
      </div>

      {/* Upload Zone */}
      <div className="mb-6">
        <h2 className="text-sm font-medium text-text-muted mb-3 px-2 uppercase tracking-wider">Add Knowledge</h2>
        <div 
          className={`relative rounded-xl border-2 border-dashed transition-all p-6 text-center cursor-pointer
            ${isDragActive ? 'border-primary-500 bg-primary-50' : 'border-border hover:border-primary-300 hover:bg-slate-50'}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <UploadCloud className="mx-auto text-primary-500 mb-2" size={24} />
          <p className="text-sm font-medium text-text-main mb-1">Drag & drop resources</p>
          <p className="text-xs text-text-muted">PDFs, Lecture Notes, Chapters</p>
        </div>
      </div>

      {/* Document Library */}
      <div className="flex-1 overflow-y-auto mb-6 scrollbar-thin">
        <h2 className="text-sm font-medium text-text-muted mb-3 px-2 uppercase tracking-wider">Library</h2>
        <div className="space-y-2">
          {files.length === 0 ? (
            <p className="text-xs text-text-muted px-2 italic">No active documents for session.</p>
          ) : (
            files.map(file => (
              <div key={file.id} className="group flex items-center justify-between p-3 rounded-lg border border-border bg-white shadow-sm">
                <div className="flex items-center gap-3 overflow-hidden">
                  <FileText size={16} className="text-primary-600 flex-shrink-0" />
                  <div className="overflow-hidden">
                    <p className="text-sm text-text-main truncate">{file.name}</p>
                    {file.status === 'extracting' ? (
                      <p className="text-xs text-primary-500 flex items-center gap-1 mt-0.5">
                        <span className="animate-pulse h-1.5 w-1.5 rounded-full bg-primary-500 inline-block"></span>
                        Extracting & Segmenting...
                      </p>
                    ) : (
                      <p className="text-xs text-emerald-600 flex items-center gap-1 mt-0.5"><CheckCircle size={10} /> Ready for RAG</p>
                    )}
                  </div>
                </div>
                <button onClick={() => removeFile(file.id)} className="text-slate-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity">
                  <X size={16} />
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Setup / Model Selection */}
      <div className="mt-auto border-t border-border pt-4">
        <h2 className="text-sm font-medium text-text-muted mb-3 px-2 flex items-center gap-2">
          <Settings2 size={14} /> System Settings
        </h2>
        <div className="px-2">
          <label className="block text-xs text-text-muted mb-1.5">RAG Language Model</label>
          <select 
            value={selectedModel} 
            onChange={(e) => setSelectedModel(e.target.value)}
            className="w-full bg-white border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/20 focus:border-primary-500 transition-all text-text-main"
          >
            {models.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
          <p className="text-[10px] text-text-muted mt-2">Used for Benchmarking Study.</p>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
