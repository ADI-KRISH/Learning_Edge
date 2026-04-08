import { useState } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import ActiveLearningPanel from './components/ActiveLearningPanel';

function App() {
  const [files, setFiles] = useState([]);
  const [selectedModel, setSelectedModel] = useState('3b');
  
  const models = [
    { id: '1b', label: 'Llama-3-1B (Fastest)' },
    { id: '3b', label: 'Llama-3-3B (Balanced)' },
    { id: '7b', label: 'Llama-3-7B (Most Accurate)' },
  ];

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* 1. Sidebar - Document Upload & settings */}
      <Sidebar 
        files={files} 
        setFiles={setFiles}
        models={models}
        selectedModel={selectedModel}
        setSelectedModel={setSelectedModel}
      />
      
      {/* 2. Main Area - Tutor Chat Interface */}
      <ChatArea />
      
      {/* 3. Active Learning Panel - Summaries, Flashcards, Quizzes */}
      <ActiveLearningPanel />
    </div>
  )
}

export default App;
