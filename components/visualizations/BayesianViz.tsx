import React, { useState } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip } from 'recharts';

// Beta distribution PDF
function betaPDF(x: number, alpha: number, beta: number) {
    return Math.pow(x, alpha - 1) * Math.pow(1 - x, beta - 1);
}

export const BayesianViz: React.FC = () => {
  const [heads, setHeads] = useState(0);
  const [tails, setTails] = useState(0);
  
  // Prior: Alpha=2, Beta=2 (Weakly centered around 0.5)
  const priorAlpha = 2;
  const priorBeta = 2;

  const data = [];
  for (let x = 0; x <= 1; x += 0.02) {
    const posterior = betaPDF(x, priorAlpha + heads, priorBeta + tails);
    // Scale for viz roughly
    data.push({ x: parseFloat(x.toFixed(2)), p: posterior });
  }

  return (
    <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 h-full flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-blue-400">Bayesian Updating (Coin Flip)</h3>
        <div className="flex gap-2">
            <button onClick={() => setHeads(h => h + 1)} className="px-3 py-1 bg-green-600 hover:bg-green-500 rounded text-white text-xs">Add Head</button>
            <button onClick={() => setTails(t => t + 1)} className="px-3 py-1 bg-red-600 hover:bg-red-500 rounded text-white text-xs">Add Tail</button>
            <button onClick={() => {setHeads(0); setTails(0)}} className="px-3 py-1 bg-slate-700 rounded text-white text-xs">Reset</button>
        </div>
      </div>

      <div className="flex-grow w-full min-h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="x" stroke="#94a3b8" label={{ value: 'Probability of Heads (θ)', position: 'insideBottom', offset: -5 }} />
            <YAxis hide />
            <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
            <Area type="monotone" dataKey="p" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.3} name="Posterior Density" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      
      <div className="mt-2 text-center text-slate-400 text-sm">
          Data: <span className="text-green-400">{heads} Heads</span>, <span className="text-red-400">{tails} Tails</span>
          <br/>
          <span className="text-xs opacity-70">Posterior becomes narrower (more confident) as data increases.</span>
      </div>
    </div>
  );
};