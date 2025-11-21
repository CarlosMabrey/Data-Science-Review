import React, { useState, useMemo, useEffect } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip, BarChart, Bar } from 'recharts';
import { RefreshCw, Play } from 'lucide-react';

export const NormalDistViz: React.FC = () => {
  const [mode, setMode] = useState<'PDF' | 'CLT'>('PDF');
  
  // PDF State
  const [mean, setMean] = useState(0);
  const [stdDev, setStdDev] = useState(1);

  // CLT State
  const [samples, setSamples] = useState<number[]>([]);
  const [cltRunning, setCltRunning] = useState(false);

  const pdfData = useMemo(() => {
    const result = [];
    const minX = -5;
    const maxX = 5;
    const step = 0.1;
    for (let x = minX; x <= maxX; x += step) {
      const exponent = -0.5 * Math.pow((x - mean) / stdDev, 2);
      const y = (1 / (stdDev * Math.sqrt(2 * Math.PI))) * Math.exp(exponent);
      result.push({ x: parseFloat(x.toFixed(2)), y });
    }
    return result;
  }, [mean, stdDev]);

  // CLT Logic
  useEffect(() => {
      if (cltRunning && samples.length < 500) {
          const interval = setInterval(() => {
              // Take sample of 30 from Uniform Dist [0, 10]
              let sum = 0;
              for(let i=0; i<30; i++) sum += Math.random() * 10;
              setSamples(prev => [...prev, sum/30]);
          }, 50);
          return () => clearInterval(interval);
      } else {
          setCltRunning(false);
      }
  }, [cltRunning, samples]);

  const cltHistogram = useMemo(() => {
      if (samples.length === 0) return [];
      const bins = new Array(20).fill(0);
      // Uniform [0,10], mean is 5. Range likely 3-7
      samples.forEach(s => {
          const binIdx = Math.min(19, Math.max(0, Math.floor((s - 2) * 2.5))); 
          bins[binIdx]++;
      });
      return bins.map((count, i) => ({ bin: (2 + i/2.5).toFixed(1), count }));
  }, [samples]);

  return (
    <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 h-full flex flex-col">
      <div className="mb-4 flex justify-between items-center">
        <div className="flex gap-2 bg-slate-800 p-1 rounded">
            <button onClick={() => setMode('PDF')} className={`px-3 py-1 rounded text-xs transition ${mode === 'PDF' ? 'bg-slate-600 text-white' : 'text-slate-400'}`}>PDF</button>
            <button onClick={() => setMode('CLT')} className={`px-3 py-1 rounded text-xs transition ${mode === 'CLT' ? 'bg-slate-600 text-white' : 'text-slate-400'}`}>Central Limit Theorem</button>
        </div>
      </div>

      {mode === 'PDF' ? (
          <>
            <div className="flex-grow w-full min-h-[250px]">
                <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={pdfData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="x" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f1f5f9' }} itemStyle={{ color: '#38bdf8' }}/>
                    <Area type="monotone" dataKey="y" stroke="#38bdf8" fill="#38bdf8" fillOpacity={0.3} />
                </AreaChart>
                </ResponsiveContainer>
            </div>
            <div className="mt-6 space-y-4 p-4 bg-slate-950 rounded border border-slate-800">
                <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1"><span>Mean (μ): {mean}</span></div>
                <input type="range" min="-3" max="3" step="0.1" value={mean} onChange={(e) => setMean(parseFloat(e.target.value))} className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"/>
                </div>
                <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1"><span>Standard Deviation (σ): {stdDev}</span></div>
                <input type="range" min="0.2" max="3" step="0.1" value={stdDev} onChange={(e) => setStdDev(parseFloat(e.target.value))} className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"/>
                </div>
            </div>
          </>
      ) : (
          <div className="flex flex-col h-full">
              <div className="flex-grow w-full min-h-[200px]">
                <ResponsiveContainer width="100%" height="100%">
                <BarChart data={cltHistogram}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="bin" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Bar dataKey="count" fill="#22c55e" />
                </BarChart>
                </ResponsiveContainer>
            </div>
            <div className="mt-4 p-4 bg-slate-950 rounded border border-slate-800 text-center">
                <p className="text-xs text-slate-400 mb-3">
                    Sampling from a <strong>Uniform Distribution</strong>. <br/>
                    The distribution of <strong>Sample Means</strong> converges to Normal.
                </p>
                <div className="flex justify-center gap-4">
                    <button onClick={() => setCltRunning(!cltRunning)} className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-500 rounded text-white text-xs">
                        <Play size={14}/> {cltRunning ? 'Pause Sampling' : 'Start Sampling'}
                    </button>
                    <button onClick={() => {setSamples([]); setCltRunning(false)}} className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded text-white text-xs">
                        <RefreshCw size={14}/> Reset
                    </button>
                </div>
                <div className="mt-2 text-xs text-slate-500">Samples: {samples.length}</div>
            </div>
          </div>
      )}
    </div>
  );
};