import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { ContentArea } from './components/ContentArea';
import { ChatBot } from './components/ChatBot';
import { INITIAL_TOPICS } from './constants';
import { Topic, ExplanationContent } from './types';
import { ThemeProvider } from './contexts/ThemeContext';

const App: React.FC = () => {
  const [selectedTopic, setSelectedTopic] = useState<Topic | null>(INITIAL_TOPICS[0]);
  const [currentContent, setCurrentContent] = useState<ExplanationContent | null>(null);

  return (
    <ThemeProvider>
      <div className="flex h-screen bg-background text-text-main font-sans selection:bg-accent/30 transition-colors duration-300">
        <Sidebar 
          topics={INITIAL_TOPICS} 
          selectedTopic={selectedTopic} 
          onSelectTopic={setSelectedTopic} 
        />
        <ContentArea 
          topic={selectedTopic} 
          onContentLoaded={setCurrentContent}
        />
        <ChatBot 
          currentTopic={selectedTopic} 
          contextContent={currentContent}
        />
      </div>
    </ThemeProvider>
  );
};

export default App;