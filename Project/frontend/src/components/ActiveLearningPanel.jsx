import React, { useState } from 'react';
import { Layers, GraduationCap, LayoutList, RefreshCcw, ShieldCheck } from 'lucide-react';

const ActiveLearningPanel = () => {
  const [activeTab, setActiveTab] = useState('summary');
  const [isFlipped, setIsFlipped] = useState(false);
  const [selectedAnswer, setSelectedAnswer] = useState(null);

  const tabs = [
    { id: 'summary', label: 'Summaries', icon: LayoutList },
    { id: 'flashcards', label: 'Flashcards', icon: Layers },
    { id: 'quiz', label: 'Quiz', icon: GraduationCap },
  ];

  return (
    <aside className="w-[360px] bg-white border-l border-border h-full flex flex-col flex-shrink-0 z-10 shadow-sm">
      {/* Header Tabs */}
      <header className="h-16 border-b border-border flex items-end px-4 gap-2 flex-shrink-0">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button 
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1.5 pb-3 px-3 text-sm font-medium border-b-2 transition-colors
                ${isActive ? 'border-primary-500 text-primary-600' : 'border-transparent text-text-muted hover:text-text-main'}`}
            >
              <Icon size={16} /> 
              {tab.label}
            </button>
          )
        })}
      </header>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto p-5 scrollbar-thin bg-slate-50/30">
        {activeTab === 'summary' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-text-main">Revision Summaries</h3>
              <button className="text-xs text-primary-600 font-medium">Generate New</button>
            </div>
            <div className="bg-white border text-sm text-text-main border-border rounded-xl p-4 shadow-sm prose prose-sm">
              <h4 className="font-medium text-primary-700 mb-2">Key Concepts: Chapter 4</h4>
              <ul className="space-y-2 list-disc pl-4 text-slate-600">
                <li>Retrieval-Augmented Generation bridges the gap between semantic understanding and parametric memory.</li>
                <li>Embeddings capture contextual context rather than semantic similarity mapping alone.</li>
                <li>Quantized Low-Rank Adaptation (QLoRA) parameter updates reduce memory dramatically.</li>
              </ul>
            </div>
          </div>
        )}

        {activeTab === 'flashcards' && (
          <div className="flex flex-col h-full">
            <h3 className="font-semibold text-text-main mb-4">Active Recall</h3>
            
            <div 
              className="flex-1 relative cursor-pointer min-h-[250px] perspective-1000 group mb-4"
              onClick={() => setIsFlipped(!isFlipped)}
            >
              <div className={`w-full h-full absolute top-0 left-0 transition-all duration-500 transform-style-preserve-3d flex items-center justify-center p-6 text-center border rounded-2xl shadow-sm bg-white border-border hover:border-primary-300
                ${isFlipped ? '[transform:rotateY(180deg)]' : ''}`}>
                
                <div className="absolute top-4 right-4 text-slate-300">
                  <RefreshCcw size={16} />
                </div>
                
                {/* Front */}
                <div className={`absolute inset-0 backface-hidden flex items-center justify-center p-6 text-base font-medium text-text-main
                  ${isFlipped ? 'invisible' : 'visible'}`}>
                  What is the primary benefit of using FAISS in a local RAG stack?
                </div>
                
                {/* Back */}
                <div className={`absolute inset-0 backface-hidden [transform:rotateY(180deg)] flex items-center justify-center p-6 text-base text-primary-700 bg-primary-50 rounded-2xl
                  ${isFlipped ? 'visible' : 'invisible'}`}>
                  It allows for highly optimized, low-latency semantic search and clustering of dense vectors without relying on cloud infrastructure.
                </div>
              </div>
            </div>
            
            <div className="flex justify-between items-center text-sm font-medium">
              <button className="px-4 py-2 bg-white border border-border rounded-lg shadow-sm hover:bg-slate-50">Previous</button>
              <span className="text-text-muted">1 / 15</span>
              <button className="px-4 py-2 bg-primary-600 text-white rounded-lg shadow-sm hover:bg-primary-700">Next Card</button>
            </div>
          </div>
        )}

        {activeTab === 'quiz' && (
          <div className="space-y-4">
             <div className="flex justify-between items-end mb-2">
                <h3 className="font-semibold text-text-main">Quiz Dashboard</h3>
                <span className="text-xs px-2 py-1 bg-yellow-100 text-yellow-800 rounded font-medium">Hard Mode</span>
             </div>
             <p className="text-sm font-medium text-slate-700 mb-4 bg-white p-4 border border-border shadow-sm rounded-xl">
               Which of the following best describes the role of "Instruction Tuning" vs "Standard Fine-Tuning" in LLMs?
             </p>
             <div className="space-y-3">
               {[
                 { id: 'a', text: 'Standard tuning relies on few-shot prompting, while instruction tuning updates weights.' },
                 { id: 'b', text: 'Instruction tuning optimizes for alignment to human directives, whereas standard fine-tuning typically continues next-token prediction over domain data.' },
                 { id: 'c', text: 'Instruction tuning uses LoRA arrays to compress model context windows.', distractor: true },
                 { id: 'd', text: 'There is no mathematical difference; both update the same loss function.', distractor: true }
               ].map((option) => {
                 const isSelected = selectedAnswer === option.id;
                 const isCorrect = option.id === 'b';
                 const showResult = selectedAnswer !== null;

                 let itemClass = "w-full text-left p-4 rounded-xl border text-sm transition-all focus:outline-none focus:ring-2 focus:ring-primary-500/30 ";
                 
                 if (!showResult) {
                   itemClass += "bg-white border-border hover:border-primary-300";
                 } else if (isCorrect) {
                   itemClass += "bg-emerald-50 border-emerald-300 text-emerald-800 shadow-sm";
                 } else if (isSelected && !isCorrect) {
                   itemClass += "bg-red-50 border-red-300 text-red-800 shadow-sm";
                 } else {
                   itemClass += "bg-slate-50 border-slate-200 text-slate-400";
                 }

                 return (
                   <button 
                     key={option.id}
                     onClick={() => !showResult && setSelectedAnswer(option.id)}
                     disabled={showResult}
                     className={itemClass}
                   >
                     <div className="flex gap-3">
                       <span className="font-semibold uppercase opacity-70 mt-0.5">{option.id}</span>
                       <span className="leading-relaxed">
                         {option.text}
                         {showResult && option.distractor && isSelected && (
                           <div className="block mt-2 text-xs font-semibold text-red-600 bg-red-100/50 p-2 rounded">
                             * This is a plausibe distractor. It mixes concepts from attention mechanisms with efficiency methods.
                           </div>
                         )}
                       </span>
                     </div>
                   </button>
                 )
               })}
             </div>
             {selectedAnswer !== null && (
                <button 
                  onClick={() => setSelectedAnswer(null)}
                  className="w-full mt-6 py-2.5 bg-slate-900 text-white rounded-lg text-sm font-medium hover:bg-slate-800 mb-4"
                >
                  Next Question
                </button>
             )}
          </div>
        )}
      </div>
      
      {/* Privacy Note Footer Component - Placed in Active Learning side but visually anchors right side */}
      <div className="p-3 border-t border-border bg-slate-50 text-[10px] text-center text-text-muted flex items-center justify-center gap-1.5 flex-shrink-0">
        <ShieldCheck size={12} className="text-emerald-600" />
        <span>Unified, Privacy-Preserving Local Architecture—No Data Leaves This Device.</span>
      </div>
    </aside>
  );
};

export default ActiveLearningPanel;
