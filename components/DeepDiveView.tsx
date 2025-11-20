import React from 'react';
import { DeepDiveContent } from '../types';
import { MarkdownView } from './MarkdownView';
import { BookOpen, Link as LinkIcon, Sigma, ArrowUpRight } from 'lucide-react';

export const DeepDiveView: React.FC<{ content: DeepDiveContent }> = ({ content }) => {
  if (!content) {
      return <div className="text-slate-500 text-center p-10">No advanced content available for this topic.</div>;
  }

  return (
    <div className="space-y-8 max-w-4xl mx-auto py-6">
      
      {/* Advanced Theory Section */}
      <div className="bg-slate-900/50 p-8 rounded-xl border border-slate-800 shadow-lg">
        <h3 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400 mb-6 flex items-center gap-3">
           <BookOpen className="text-purple-400"/> Advanced Theory
        </h3>
        <MarkdownView text={content.advancedTheory || 'Advanced theory is currently unavailable.'} />
      </div>

      {/* Key Formulas Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
         <div className="bg-slate-900/50 p-6 rounded-xl border border-slate-800">
            <h3 className="text-lg font-semibold text-yellow-400 mb-4 flex items-center gap-2">
                <Sigma size={20} /> Key Mathematical Formulations
            </h3>
            <div className="space-y-4">
                {(content.keyFormulas || []).length > 0 ? (
                    (content.keyFormulas || []).map((formula, idx) => {
                        // Ensure formula is wrapped in display delimiters if not already
                        const cleanFormula = formula.replace(/^\$\$|\$\$/g, '').trim();
                        const displayMath = `$$ ${cleanFormula} $$`;
                        
                        return (
                            <div key={idx} className="bg-slate-950 px-4 py-3 rounded border-l-4 border-yellow-500/50 overflow-x-auto">
                                <MarkdownView text={displayMath} />
                            </div>
                        );
                    })
                ) : (
                    <div className="text-slate-500 text-sm italic">No specific formulas provided.</div>
                )}
            </div>
         </div>

         {/* Seminal Papers Section */}
         <div className="bg-slate-900/50 p-6 rounded-xl border border-slate-800">
            <h3 className="text-lg font-semibold text-blue-400 mb-4 flex items-center gap-2">
                <LinkIcon size={20} /> Seminal Papers & Resources
            </h3>
            <div className="space-y-3">
                {(content.seminalPapers || []).length > 0 ? (
                    (content.seminalPapers || []).map((paper, idx) => (
                        <a 
                            key={idx} 
                            href={paper.url || '#'} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="block p-4 bg-slate-800/50 hover:bg-slate-800 border border-slate-700 hover:border-blue-500/50 rounded-lg transition-all group"
                        >
                            <div className="font-medium text-slate-200 group-hover:text-blue-300 transition-colors mb-1">
                                {paper.title}
                            </div>
                            <div className="text-xs text-slate-500 flex items-center gap-1">
                                External Resource <ArrowUpRight size={10} />
                            </div>
                        </a>
                    ))
                ) : (
                    <div className="text-slate-500 text-sm italic">No references available.</div>
                )}
            </div>
         </div>
      </div>
    </div>
  );
};