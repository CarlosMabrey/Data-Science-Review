import React, { useState, useEffect, useRef } from 'react';
import { Play, RefreshCw, Terminal, Loader2, Trash2, Sparkles } from 'lucide-react';
import { fixPythonCode } from '../services/geminiService';

declare global {
  interface Window {
    loadPyodide: (config: any) => Promise<any>;
  }
}

export const CodePlayground: React.FC<{ initialCode: string }> = ({ initialCode }) => {
  const [code, setCode] = useState(initialCode);
  const [output, setOutput] = useState<string[]>([]);
  const [pyodide, setPyodide] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [isFixing, setIsFixing] = useState(false);

  useEffect(() => {
    setCode(initialCode);
  }, [initialCode]);

  useEffect(() => {
    const initPyodide = async () => {
      try {
        if (!window.loadPyodide) {
          console.error("Pyodide script not loaded");
          return;
        }
        const py = await window.loadPyodide({
          indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.0/full/"
        });
        
        // Load micropip to install packages that might not be in the core bundle or to ensure resolution
        await py.loadPackage("micropip");
        const micropip = py.pyimport("micropip");

        // Load core packages
        await py.loadPackage(['numpy', 'pandas', 'scikit-learn', 'scipy', 'matplotlib']);
        
        // Install seaborn using micropip to avoid "No known package" errors in some environments
        await micropip.install('seaborn');
        
        setPyodide(py);
        setOutput(['Python 3.11 (Pyodide) Environment Ready.', 'Loaded: numpy, pandas, sklearn, scipy, matplotlib, seaborn']);
      } catch (err) {
        console.error("Failed to load Pyodide:", err);
        setOutput(['Error loading Python environment. Check internet connection or reload page.']);
      } finally {
        setIsLoading(false);
      }
    };
    initPyodide();
  }, []);

  const runCode = async () => {
    if (!pyodide) return;
    setIsRunning(true);
    
    // Reset stdout capture
    const logs: string[] = [];
    pyodide.setStdout({ batched: (msg: string) => logs.push(msg) });
    pyodide.setStderr({ batched: (msg: string) => logs.push(`Error: ${msg}`) });

    try {
      // Matplotlib cleanup hack for Pyodide
      await pyodide.runPythonAsync(`
import matplotlib.pyplot as plt
plt.clf()
      `);
      await pyodide.runPythonAsync(code);
      setOutput(prev => [...prev, '>>> Run Complete', ...logs]);
    } catch (err: any) {
      setOutput(prev => [...prev, `Traceback: ${err.message}`]);
    } finally {
      setIsRunning(false);
    }
  };

  const handleFixWithAI = async () => {
    const lastOutput = output.slice(-5).join('\n');
    setIsFixing(true);
    try {
        const result = await fixPythonCode(code, lastOutput);
        setCode(result.fixedCode);
        setOutput(prev => [...prev, '--------------------------------', `>>> AI FIX: ${result.explanation}`, '--------------------------------']);
    } catch (e) {
        setOutput(prev => [...prev, `>>> AI Fix Failed: Could not generate fix.`]);
    } finally {
        setIsFixing(false);
    }
  };

  return (
    <div className="flex flex-col h-[600px] bg-slate-950 rounded-lg border border-slate-800 overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center justify-between p-3 bg-slate-900 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Terminal size={16} className="text-green-400" />
          <span className="text-sm font-semibold text-slate-300">Interactive Python Lab</span>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleFixWithAI}
            disabled={isFixing || isLoading}
            className="px-3 py-1.5 text-xs flex items-center gap-2 text-purple-300 hover:text-purple-100 hover:bg-purple-900/50 rounded border border-purple-500/30 disabled:opacity-50 transition-all"
          >
            {isFixing ? <Loader2 size={14} className="animate-spin"/> : <Sparkles size={14} />}
            {isFixing ? 'Consulting Professor...' : 'Fix with AI'}
          </button>
          <div className="w-px bg-slate-700 mx-1"></div>
          <button
            onClick={() => setCode(initialCode)}
            className="px-3 py-1.5 text-xs flex items-center gap-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded"
          >
            <RefreshCw size={14} /> Reset
          </button>
          <button
            onClick={() => setOutput([])}
            className="px-3 py-1.5 text-xs flex items-center gap-2 text-slate-400 hover:text-red-400 hover:bg-slate-800 rounded"
          >
            <Trash2 size={14} /> Clear
          </button>
          <button
            onClick={runCode}
            disabled={isLoading || isRunning}
            className={`px-4 py-1.5 text-xs flex items-center gap-2 rounded font-semibold transition-colors ${
              isLoading || isRunning 
                ? 'bg-slate-700 text-slate-400 cursor-wait' 
                : 'bg-green-600 hover:bg-green-500 text-white'
            }`}
          >
            {isRunning ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
            {isRunning ? 'Running...' : 'Run Code'}
          </button>
        </div>
      </div>

      {/* Editor & Output Split */}
      <div className="flex flex-col md:flex-row flex-grow overflow-hidden">
        <div className="w-full md:w-1/2 h-full border-r border-slate-800 flex flex-col">
            <textarea
                value={code}
                onChange={(e) => setCode(e.target.value)}
                className="w-full h-full bg-[#0d1117] text-slate-300 font-mono text-sm p-4 resize-none focus:outline-none leading-relaxed"
                spellCheck={false}
                disabled={isLoading}
            />
        </div>
        <div className="w-full md:w-1/2 h-full bg-[#0f172a] flex flex-col">
            <div className="bg-slate-900 px-4 py-1 text-xs text-slate-500 font-mono border-b border-slate-800 flex justify-between">
                <span>Output Console</span>
                {isLoading && <span className="text-yellow-500">Installing Libraries (seaborn)...</span>}
            </div>
            <div className="flex-grow p-4 font-mono text-sm overflow-y-auto custom-scrollbar">
                {isLoading ? (
                    <div className="flex flex-col items-center justify-center h-full text-slate-500 gap-2">
                        <Loader2 size={24} className="animate-spin" /> 
                        <p>Initializing Data Science Environment...</p>
                        <p className="text-xs text-slate-600">Loading pandas, numpy, scipy, sklearn</p>
                        <p className="text-xs text-slate-600">Installing seaborn via micropip...</p>
                    </div>
                ) : output.length === 0 ? (
                    <span className="text-slate-600 italic">Ready. Output will appear here. Plots render in the memory buffer (text representation only in this view).</span>
                ) : (
                    output.map((line, i) => (
                        <div key={i} className={`break-words ${line.startsWith('Error') || line.startsWith('Traceback') ? 'text-red-400' : line.startsWith('>>>') ? 'text-green-400 mt-2 mb-1 font-bold' : 'text-slate-300'}`}>
                            {line}
                        </div>
                    ))
                )}
            </div>
        </div>
      </div>
    </div>
  );
};