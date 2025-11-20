import React from 'react';
import { Topic, TopicCategory } from '../types';
import { BookOpen, Code, BarChart, Database, Activity, CloudCog, Brain, Server, Network } from 'lucide-react';

interface SidebarProps {
  topics: Topic[];
  selectedTopic: Topic | null;
  onSelectTopic: (topic: Topic) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ topics, selectedTopic, onSelectTopic }) => {
  const categories = Object.values(TopicCategory);

  const getIcon = (cat: TopicCategory) => {
    switch (cat) {
      case TopicCategory.PYTHON: return <Code size={16} className="text-yellow-500" />;
      case TopicCategory.ML: return <Activity size={16} className="text-blue-500" />;
      case TopicCategory.STATS: return <BarChart size={16} className="text-green-500" />;
      case TopicCategory.GENAI: return <Brain size={16} className="text-purple-400" />;
      case TopicCategory.DE: return <Server size={16} className="text-cyan-500" />;
      case TopicCategory.DATA: return <Database size={16} className="text-indigo-500" />;
      case TopicCategory.MLOPS: return <CloudCog size={16} className="text-orange-500" />;
      default: return <BookOpen size={16} />;
    }
  };

  return (
    <div className="w-72 bg-surface border-r border-border flex flex-col h-full transition-colors duration-300">
      <div className="p-5 border-b border-border">
        <h1 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-accent to-purple-400 flex items-center gap-2">
          <BookOpen size={24} className="text-accent"/>
          DataSci Master
        </h1>
        <p className="text-xs text-text-muted mt-1">Senior Technical Interview Prep</p>
      </div>

      <div className="overflow-y-auto flex-grow p-3 space-y-6 custom-scrollbar">
        {categories.map(category => {
          const catTopics = topics.filter(t => t.category === category);
          if (catTopics.length === 0) return null;

          return (
            <div key={category}>
              <h2 className="text-xs font-bold text-text-muted uppercase tracking-wider px-3 mb-2 flex items-center gap-2">
                {getIcon(category)} {category}
              </h2>
              <div className="space-y-1">
                {catTopics.map(topic => (
                  <button
                    key={topic.id}
                    onClick={() => onSelectTopic(topic)}
                    className={`w-full text-left px-3 py-2 rounded-md text-sm transition-colors duration-200 ${
                      selectedTopic?.id === topic.id 
                        ? 'bg-surfaceHighlight text-accent font-medium border border-border' 
                        : 'text-text-muted hover:bg-surfaceHighlight/50 hover:text-text-main'
                    }`}
                  >
                    {topic.title}
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};