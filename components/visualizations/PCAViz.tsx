import React, { useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, ResponsiveContainer, ReferenceLine } from 'recharts';

const generateCorrelatedData = () => {
  const data = [];
  for (let i = 0; i < 50; i++) {
    const x = Math.random() * 100;
    const y = x * 0.8 + Math.random() * 20; // Strong positive correlation
    data.push({ x, y });
  }
  return data;
};

export const PCAViz: React.FC = () => {
  const [data] = useState(generateCorrelatedData());
  const [showPC, setShowPC] = useState(false);
  const [project, setProject] = useState(false);

  const projectedData = project 
    ? data.map(p => {
        // Project onto line y = 0.8x roughly (PC1)
        const avg = (p.x + p.y)/2; 
        return { x: avg, y: avg * 0.8 }; 
      })
    : data;

  return (
    <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 h-full flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-blue-400">PCA Projection</h3>
        <div className="flex gap-2">
          <button onClick={() => setShowPC(!showPC)} className={`px-3 py-1 rounded text-xs border ${showPC ? 'bg-purple-500/20 border-purple-500 text-purple-300' : 'border-slate-600 text-slate-400'}`}>
            {showPC ? 'Hide Components' : 'Show Principal Components'}
          </button>
          <button onClick={() => setProject(!project)} className={`px-3 py-1 rounded text-xs border ${project ? 'bg-blue-500/20 border-blue-500 text-blue-300' : 'border-slate-600 text-slate-400'}`}>
            {project ? 'Reset View' : 'Project to 1D'}
          </button>
        </div>
      </div>

      <div className="flex-grow w-full min-h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis type="number" dataKey="x" stroke="#94a3b8" domain={[0, 100]} />
            <YAxis type="number" dataKey="y" stroke="#94a3b8" domain={[0, 100]} />
            <Scatter name="Data" data={projectedData} fill="#38bdf8" />
            
            {showPC && (
                <>
                    {/* PC1 (Variance Max) */}
                    <ReferenceLine segment={[{x: 10, y: 8}, {x: 90, y: 72}]} stroke="#d946ef" strokeWidth={3} label="PC1" />
                    {/* PC2 (Orthogonal) */}
                    <ReferenceLine segment={[{x: 50, y: 40}, {x: 40, y: 52}]} stroke="#22c55e" strokeWidth={2} label="PC2" />
                </>
            )}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 text-xs text-slate-500 text-center">
          PCA finds the axes of maximum variance (PC1) and projects data onto them to reduce dimensions while preserving information.
      </div>
    </div>
  );
};