import React, { useState, useEffect } from 'react';
import { QuizQuestion } from '../types';
import { CheckCircle, XCircle, ArrowRight, Award, RefreshCw, HelpCircle } from 'lucide-react';
import { motion } from 'framer-motion';
import { MarkdownView } from './MarkdownView';

interface QuizModuleProps {
  questions: QuizQuestion[];
  onRetry: () => void;
}

export const QuizModule: React.FC<QuizModuleProps> = ({ questions, onRetry }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedOption, setSelectedOption] = useState<number | null>(null);
  const [showFeedback, setShowFeedback] = useState(false);
  const [score, setScore] = useState(0);
  const [completed, setCompleted] = useState(false);

  useEffect(() => {
    resetQuiz();
  }, [questions]);

  const resetQuiz = () => {
    setCurrentIndex(0);
    setSelectedOption(null);
    setShowFeedback(false);
    setScore(0);
    setCompleted(false);
  };

  const handleOptionClick = (index: number) => {
    if (showFeedback) return;
    setSelectedOption(index);
  };

  const handleSubmit = () => {
    if (selectedOption === null) return;
    
    const isCorrect = selectedOption === questions[currentIndex].correctIndex;
    if (isCorrect) setScore(s => s + 1);
    setShowFeedback(true);
  };

  const handleNext = () => {
    if (currentIndex < questions.length - 1) {
      setCurrentIndex(c => c + 1);
      setSelectedOption(null);
      setShowFeedback(false);
    } else {
      setCompleted(true);
    }
  };

  if (!questions || questions.length === 0) {
    return <div className="text-slate-500 p-8 text-center">No quiz available for this topic yet.</div>;
  }

  if (completed) {
    const percentage = Math.round((score / questions.length) * 100);
    return (
      <div className="flex flex-col items-center justify-center h-[500px] text-center space-y-6">
        <motion.div 
          initial={{ scale: 0 }} animate={{ scale: 1 }} 
          className={`p-6 rounded-full ${percentage >= 70 ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}`}
        >
          <Award size={64} />
        </motion.div>
        <div>
          <h2 className="text-3xl font-bold text-white mb-2">Assessment Complete</h2>
          <p className="text-slate-400">You scored <span className="text-white font-bold">{score}/{questions.length}</span> ({percentage}%)</p>
        </div>
        <div className="flex gap-4">
            <button onClick={resetQuiz} className="px-6 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-white transition flex items-center gap-2">
                <RefreshCw size={18} /> Retry Quiz
            </button>
             {percentage < 100 && (
                <button onClick={onRetry} className="px-6 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-white transition">
                    Review Topic
                </button>
             )}
        </div>
      </div>
    );
  }

  const question = questions[currentIndex];
  if (!question || !question.options) {
      return <div className="text-red-400 p-4">Error loading question data.</div>;
  }

  return (
    <div className="max-w-2xl mx-auto py-8">
      {/* Progress Bar */}
      <div className="mb-8">
        <div className="flex justify-between text-xs text-slate-400 mb-2">
          <span>Question {currentIndex + 1} of {questions.length}</span>
          <span>Score: {score}</span>
        </div>
        <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
          <div 
            className="h-full bg-blue-500 transition-all duration-500"
            style={{ width: `${((currentIndex + 1) / questions.length) * 100}%` }}
          />
        </div>
      </div>

      <motion.div
        key={currentIndex}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        className="space-y-6"
      >
        <div className="text-xl font-semibold text-white leading-relaxed">
            <MarkdownView text={question.question} bare />
        </div>

        <div className="space-y-3">
          {(question.options || []).map((option, idx) => {
            let statusClass = "border-slate-700 bg-slate-800/50 hover:bg-slate-800";
            
            if (showFeedback) {
               if (idx === question.correctIndex) statusClass = "border-green-500/50 bg-green-500/10 text-green-300";
               else if (idx === selectedOption) statusClass = "border-red-500/50 bg-red-500/10 text-red-300 opacity-50";
               else statusClass = "border-slate-800 opacity-30";
            } else if (selectedOption === idx) {
               statusClass = "border-blue-500 bg-blue-500/20 text-blue-300";
            }

            return (
              <button
                key={idx}
                onClick={() => handleOptionClick(idx)}
                disabled={showFeedback}
                className={`w-full p-4 rounded-lg border-2 text-left transition-all duration-200 flex justify-between items-center group ${statusClass}`}
              >
                <div className="text-sm">
                    <MarkdownView text={option} bare as="span" />
                </div>
                {showFeedback && idx === question.correctIndex && <CheckCircle size={18} className="text-green-400 shrink-0 ml-2" />}
                {showFeedback && idx === selectedOption && idx !== question.correctIndex && <XCircle size={18} className="text-red-400 shrink-0 ml-2" />}
              </button>
            );
          })}
        </div>

        {showFeedback && (
            <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-slate-800/50 p-4 rounded-lg border border-slate-700"
            >
                <div className="flex items-start gap-3">
                    <HelpCircle className="shrink-0 text-blue-400 mt-1" size={20} />
                    <div>
                        <h4 className="font-medium text-blue-200 mb-1">Explanation</h4>
                        <div className="text-sm text-slate-300 leading-relaxed">
                            <MarkdownView text={question.explanation} bare />
                        </div>
                    </div>
                </div>
            </motion.div>
        )}

        <div className="flex justify-end mt-8">
            {!showFeedback ? (
                 <button 
                    onClick={handleSubmit}
                    disabled={selectedOption === null}
                    className="px-6 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white font-medium transition-colors"
                 >
                    Submit Answer
                 </button>
            ) : (
                <button 
                    onClick={handleNext}
                    className="px-6 py-2 bg-slate-100 hover:bg-white text-slate-900 rounded-lg font-medium flex items-center gap-2 transition-colors"
                >
                    {currentIndex < questions.length - 1 ? 'Next Question' : 'Finish Quiz'} <ArrowRight size={18}/>
                </button>
            )}
        </div>
      </motion.div>
    </div>
  );
};