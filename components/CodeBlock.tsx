import React from 'react';
import { Copy } from 'lucide-react';

export const CodeBlock: React.FC<{ code: string; language?: string }> = ({ code, language = 'python' }) => {
  return (
    <div className="relative bg-slate-950 rounded-lg border border-slate-800 overflow-hidden my-4 group">
      <div className="flex justify-between items-center px-4 py-2 bg-slate-900 border-b border-slate-800">
        <span className="text-xs text-slate-400 font-mono uppercase">{language}</span>
        <button 
          onClick={() => navigator.clipboard.writeText(code)}
          className="text-slate-500 hover:text-slate-200 transition-colors"
        >
          <Copy size={14} />
        </button>
      </div>
      <div className="p-4 overflow-x-auto">
        <pre className="text-sm font-mono text-slate-300 leading-relaxed">
          <code>{code}</code>
        </pre>
      </div>
    </div>
  );
};
