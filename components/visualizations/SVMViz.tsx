import React, { useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, ResponsiveContainer, ReferenceLine, Cell } from 'recharts';

const data = [
  { x: 1, y: 2, c: 0 }, { x: 2, y: 1, c: 0 }, { x: 3, y: 3, c: 0 },
  { x: 6, y: 8, c: 1 }, { x: 7, y: 7, c: 1 }, { x: 8, y: 9, c: 1 },
];

export const SVMViz: React.FC = () => {
  const [C, setC] = useState(1); // Regularization param
  
  // Simulated margin width based on C
  // High C = Hard Margin (Narrow)
  // Low C = Soft Margin (Wide)
  const margin = 2 / Math.sqrt(C); 

  return (
    <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 h-full flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-blue-400">SVM Hyperplane & Margins</h3>
      </div>

      <div className="flex-grow w-full min-h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis type="number" dataKey="x" stroke="#94a3b8" domain={[0, 10]} />
            <YAxis type="number" dataKey="y" stroke="#94a3b8" domain={[0, 10]} />
            <Scatter name="Points" data={data}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.c === 0 ? '#ef4444' : '#3b82f6'} />
              ))}
            </Scatter>
            
            {/* Hyperplane y = x + 1 */}
            <ReferenceLine segment={[{x: 0, y: 1}, {x: 9, y: 10}]} stroke="#fff" strokeWidth={2} />
            
            {/* Margins */}
            <ReferenceLine segment={[{x: 0, y: 1 + margin}, {x: 9, y: 10 + margin}]} stroke="#64748b" strokeDasharray="5 5" />
            <ReferenceLine segment={[{x: 0, y: 1 - margin}, {x: 9, y: 10 - margin}]} stroke="#64748b" strokeDasharray="5 5" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 p-4 bg-slate-950 rounded border border-slate-800">
         <div className="flex justify-between text-xs text-slate-400 mb-2">
            <span>Parameter C: {C.toFixed(2)}</span>
            <span>{C < 0.5 ? 'Soft Margin (Allows misclassifications)' : 'Hard Margin (Strict)'}</span>
        </div>
        <input type="range" min="0.1" max="5" step="0.1" value={C} onChange={e => setC(Number(e.target.value))} className="w-full accent-blue-500"/>
      </div>
    </div>
  );
};