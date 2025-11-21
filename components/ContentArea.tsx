import React, { useEffect, useState, useRef } from 'react';
import { Topic, VisualizationType, ExplanationContent } from '../types';
import { fetchTopicExplanation } from '../services/geminiService';
import { STATIC_CONTENT } from '../data/staticContent';
import { MarkdownView } from './MarkdownView';
import { CodeBlock } from './CodeBlock';
import { CodePlayground } from './CodePlayground';
import { QuizModule } from './QuizModule';
import { DeepDiveView } from './DeepDiveView';

// Visualizations
import { LinearRegressionVizFixed } from './visualizations/LinearRegressionViz';
import { KMeansViz } from './visualizations/KMeansViz';
import { NormalDistViz } from './visualizations/NormalDistViz';
import { SortingViz } from './visualizations/SortingViz';
import { LogisticRegressionViz } from './visualizations/LogisticRegressionViz';
import { TreeEnsembleViz } from './visualizations/TreeEnsembleViz';
import { GradientDescentViz } from './visualizations/GradientDescentViz';
import { NeuralNetViz } from './visualizations/NeuralNetViz';
import { RegularizationViz } from './visualizations/RegularizationViz';
import { PCAViz } from './visualizations/PCAViz';
import { SVMViz } from './visualizations/SVMViz';
import { BayesianViz } from './visualizations/BayesianViz';
import { BiasVarianceViz } from './visualizations/BiasVarianceViz';
import { ProbabilityDistViz } from './visualizations/ProbabilityDistViz';

import { Loader2, BrainCircuit, CheckCircle2, AlertCircle, Code, BookOpen, GraduationCap, Terminal, Layers, RotateCcw, Save, Zap } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { ThemeSwitcher } from './ThemeSwitcher';

interface ContentAreaProps {
  topic: Topic | null;
  onContentLoaded?: (content: ExplanationContent | null) => void;
}

type Tab = 'guide' | 'deep_dive' | 'code' | 'quiz';

const STORAGE_KEY = 'dsc_masterclass_cache_v1';

export const ContentArea: React.FC<ContentAreaProps> = ({ topic, onContentLoaded }) => {
  const [content, setContent] = useState<ExplanationContent | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<Tab>('guide');
  const [lastSaved, setLastSaved] = useState<string | null>(null);
  const [source, setSource] = useState<'static' | 'cache' | 'ai'>('ai');
  
  // Ref to track the currently mounted/selected topic ID to prevent race conditions
  const activeTopicIdRef = useRef<string | null>(null);

  // Propagate content changes to parent
  useEffect(() => {
    if (onContentLoaded) {
      onContentLoaded(content);
    }
  }, [content, onContentLoaded]);

  // Load cache on mount
  const loadFromCache = (topicId: string) => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const cache = JSON.parse(raw);
        return cache[topicId] || null;
      }
    } catch (e) {
      console.error("Cache read error", e);
    }
    return null;
  };

  const saveToCache = (topicId: string, data: ExplanationContent) => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      const cache = raw ? JSON.parse(raw) : {};
      cache[topicId] = { ...data, timestamp: Date.now() };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(cache));
    } catch (e) {
      console.error("Cache write error", e);
    }
  };

  const loadContent = async (forceRefresh = false) => {
    if (!topic) return;
    
    const currentRequestId = topic.id;
    
    // 1. Check for Pre-Generated Static Content (Instant Load)
    if (!forceRefresh && STATIC_CONTENT[topic.id]) {
        if (activeTopicIdRef.current !== currentRequestId) return;
        setContent(STATIC_CONTENT[topic.id]);
        setSource('static');
        setLastSaved("Pre-generated (Goldmine)");
        setLoading(false);
        setActiveTab('guide');
        return;
    }

    // 2. Check Browser Cache
    if (!forceRefresh) {
      const cachedData = loadFromCache(topic.id);
      if (cachedData) {
        if (activeTopicIdRef.current !== currentRequestId) return;
        setContent(cachedData);
        setSource('cache');
        setLastSaved(new Date(cachedData.timestamp || Date.now()).toLocaleTimeString());
        setLoading(false);
        setActiveTab('guide');
        return;
      }
    }

    // 3. Fetch from API
    setLoading(true);
    setActiveTab('guide');
    
    try {
      const data = await fetchTopicExplanation(topic.title);
      
      if (activeTopicIdRef.current !== currentRequestId) {
        console.log(`Ignored stale response for ${currentRequestId} (Current: ${activeTopicIdRef.current})`);
        return;
      }

      setContent(data);
      setSource('ai');
      saveToCache(topic.id, data);
      setLastSaved(new Date().toLocaleTimeString());
    } catch (e) {
      console.error(e);
    } finally {
      if (activeTopicIdRef.current === currentRequestId) {
        setLoading(false);
      }
    }
  };

  useEffect(() => {
    activeTopicIdRef.current = topic?.id || null;
    setContent(null);
    loadContent(false);
  }, [topic]);

  const renderVisualizer = () => {
    if (!topic) return null;
    switch (topic.visualizationType) {
      case VisualizationType.LINEAR_REGRESSION: return <LinearRegressionVizFixed />;
      case VisualizationType.LOGISTIC_REGRESSION: return <LogisticRegressionViz />;
      case VisualizationType.K_MEANS: return <KMeansViz />;
      case VisualizationType.TREE_ENSEMBLES: return <TreeEnsembleViz />;
      case VisualizationType.GRADIENT_DESCENT: return <GradientDescentViz />;
      case VisualizationType.BACKPROPAGATION: return <NeuralNetViz />;
      case VisualizationType.REGULARIZATION: return <RegularizationViz />;
      case VisualizationType.PCA: return <PCAViz />;
      case VisualizationType.SVM: return <SVMViz />;
      case VisualizationType.BAYESIAN: return <BayesianViz />;
      case VisualizationType.BIAS_VARIANCE: return <BiasVarianceViz />;
      case VisualizationType.PROB_DISTRIBUTIONS: return <ProbabilityDistViz />;
      case VisualizationType.NORMAL_DISTRIBUTION: return <NormalDistViz />;
      case VisualizationType.SORTING: return <SortingViz />;
      default:
        return (
          <div className="h-full flex flex-col items-center justify-center text-text-muted bg-surface/50 rounded-lg border border-border border-dashed">
            <BrainCircuit size={48} className="mb-4 opacity-50" />
            <p>Conceptual visualization available via Text Explanation below.</p>
          </div>
        );
    }
  };

  if (!topic) {
    return (
      <div className="flex-grow flex items-center justify-center text-text-muted">
        <div className="text-center">
          <BrainCircuit size={64} className="mx-auto mb-4 opacity-20" />
          <h2 className="text-2xl font-semibold text-text-main">Select a Topic to Begin</h2>
          <p className="max-w-md mt-2 text-text-muted">Choose from Python, Machine Learning, or Statistics to start your deep dive.</p>
        </div>
      </div>
    );
  }

  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: 'guide', label: 'Guide & Visuals', icon: <BookOpen size={16}/> },
    { id: 'deep_dive', label: 'Deep Dive', icon: <Layers size={16}/> },
    { id: 'code', label: 'Code Lab', icon: <Terminal size={16}/> },
    { id: 'quiz', label: 'Assessment', icon: <GraduationCap size={16}/> },
  ];

  return (
    <div className="flex-grow flex flex-col h-full overflow-hidden relative bg-background transition-colors duration-300">
      {/* Top Bar */}
      <div className="bg-surface/80 border-b border-border p-4 backdrop-blur-sm z-10 transition-colors duration-300">
        <div className="flex justify-between items-start mb-4">
            <div>
                <h2 className="text-2xl font-bold text-text-main flex items-center gap-2">{topic.title}</h2>
                <p className="text-text-muted text-sm">{topic.shortDesc}</p>
            </div>
            <div className="flex flex-col items-end gap-1">
               <div className="flex items-center gap-2 mb-1">
                 <span className="px-2 py-1 rounded bg-surfaceHighlight text-xs text-text-muted border border-border">{topic.category}</span>
                 <ThemeSwitcher />
               </div>
               <div className="flex items-center gap-2">
                 {source === 'static' && <span className="text-[10px] text-yellow-500 flex items-center gap-1 font-bold"><Zap size={10}/> Expert Verified</span>}
                 {lastSaved && !loading && <span className="text-[10px] text-text-muted flex items-center gap-1"><Save size={10}/> Saved {lastSaved}</span>}
               </div>
            </div>
        </div>
        
        <div className="flex justify-between items-end">
            <div className="flex gap-2 border-b border-border/50">
                {tabs.map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`px-4 py-2 text-sm font-medium flex items-center gap-2 transition-all relative ${
                            activeTab === tab.id 
                            ? 'text-accent' 
                            : 'text-text-muted hover:text-text-main'
                        }`}
                    >
                        {tab.icon}
                        {tab.label}
                        {activeTab === tab.id && (
                            <motion.div layoutId="activeTab" className="absolute bottom-0 left-0 right-0 h-0.5 bg-accent" />
                        )}
                    </button>
                ))}
            </div>

            <button 
                onClick={() => loadContent(true)} 
                disabled={loading}
                className="flex items-center gap-2 text-xs text-text-muted hover:text-text-main hover:bg-surfaceHighlight px-3 py-1 rounded transition-colors"
            >
                <RotateCcw size={12} /> {loading ? 'Generating...' : 'Regenerate with AI'}
            </button>
        </div>
      </div>

      <div className="flex-grow overflow-y-auto p-6 custom-scrollbar">
        {loading ? (
           <div className="flex flex-col items-center justify-center h-full space-y-4">
                <Loader2 className="animate-spin text-accent" size={48} />
                <p className="text-text-muted animate-pulse">Consulting the AI Master...</p>
                <p className="text-xs text-text-muted">Generating LaTeX & Deep Dive Content</p>
           </div>
        ) : content ? (
            <div className="max-w-6xl mx-auto h-full pb-10">
                <AnimatePresence mode="wait">
                    {activeTab === 'guide' && (
                        <motion.div 
                            key="guide"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="grid grid-cols-1 xl:grid-cols-2 gap-8"
                        >
                            <div className="space-y-6">
                                <div className="h-[400px] w-full">
                                    {renderVisualizer()}
                                </div>
                                <div className="bg-surface/50 p-6 rounded-xl border border-border">
                                    <h3 className="text-lg font-semibold text-accent mb-3 flex items-center gap-2">
                                        <BrainCircuit size={18}/> Technical Overview
                                    </h3>
                                    <MarkdownView text={content.overview} />
                                </div>
                            </div>
                            <div className="space-y-6">
                                <div className="bg-surface/50 p-6 rounded-xl border border-border">
                                    <h3 className="text-lg font-semibold text-purple-400 mb-3">Mathematical Intuition</h3>
                                    <div className="text-text-main leading-relaxed">
                                        <MarkdownView text={content.mathematicalIntuition} />
                                    </div>
                                </div>
                                <div>
                                    <div className="flex justify-between items-end mb-2">
                                        <h3 className="text-lg font-semibold text-yellow-400 flex items-center gap-2"><Code size={18}/> Implementation</h3>
                                        <button onClick={() => setActiveTab('code')} className="text-xs text-accent hover:text-accent-hover hover:underline">
                                            Run in Code Lab &rarr;
                                        </button>
                                    </div>
                                    <CodeBlock code={content.codeSnippet} />
                                </div>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div className="bg-green-900/10 p-4 rounded-lg border border-green-900/30">
                                        <h4 className="text-green-400 font-medium mb-3 flex items-center gap-2"><CheckCircle2 size={16}/> Strengths</h4>
                                        <ul className="space-y-2">
                                            {(content.prosCons?.pros || []).map((pro, i) => (
                                                <li key={i} className="text-sm text-text-main flex items-start gap-2">
                                                    <span className="mt-1.5 w-1 h-1 rounded-full bg-green-500 shrink-0"></span>
                                                    <div className="flex-1"><MarkdownView bare text={pro} /></div>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                    <div className="bg-red-900/10 p-4 rounded-lg border border-red-900/30">
                                        <h4 className="text-red-400 font-medium mb-3 flex items-center gap-2"><AlertCircle size={16}/> Limitations</h4>
                                        <ul className="space-y-2">
                                            {(content.prosCons?.cons || []).map((con, i) => (
                                                <li key={i} className="text-sm text-text-main flex items-start gap-2">
                                                    <span className="mt-1.5 w-1 h-1 rounded-full bg-red-500 shrink-0"></span>
                                                    <div className="flex-1"><MarkdownView bare text={con} /></div>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                </div>
                                <div className="bg-surfaceHighlight/50 p-4 rounded-lg border border-border">
                                    <h4 className="text-text-main font-medium mb-3">Industry Use Cases</h4>
                                    <div className="flex flex-wrap gap-2">
                                        {(content.useCases || []).map((uc, i) => (
                                            <div key={i} className="px-3 py-1 bg-surfaceHighlight text-text-muted rounded-full text-xs border border-border">
                                                <MarkdownView bare as="span" text={uc} />
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    )}

                    {activeTab === 'deep_dive' && (
                        <motion.div
                            key="deep_dive"
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                        >
                            <DeepDiveView content={content.deepDive} />
                        </motion.div>
                    )}

                    {activeTab === 'code' && (
                        <motion.div
                            key="code"
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                            className="h-full max-w-5xl mx-auto"
                        >
                            <div className="mb-4">
                                <h3 className="text-xl font-semibold text-text-main mb-2">Interactive Python Lab</h3>
                                <p className="text-text-muted text-sm">Experiment with the code below. The environment includes numpy, pandas, scikit-learn, scipy, matplotlib, and seaborn.</p>
                            </div>
                            <CodePlayground initialCode={content.codeSnippet} />
                        </motion.div>
                    )}

                    {activeTab === 'quiz' && (
                        <motion.div
                            key="quiz"
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                        >
                            <QuizModule questions={content.quiz} onRetry={() => setActiveTab('guide')} />
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        ) : null}
      </div>
    </div>
  );
};