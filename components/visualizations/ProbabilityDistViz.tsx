import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip } from 'recharts';

type DistType = 'BINOMIAL' | 'POISSON' | 'GEOMETRIC';

export const ProbabilityDistViz: React.FC = () => {
  const [type, setType] = useState<DistType>('BINOMIAL');
  const [param1, setParam1] = useState(0.5); // p or lambda
  const [param2, setParam2] = useState(10);  // n (for binomial)

  const factorial = (n: number): number => (n <= 1 ? 1 : n * factorial(n - 1));
  
  const getData = () => {
      const data = [];
      if (type === 'BINOMIAL') {
          // P(k) = (n choose k) * p^k * (1-p)^(n-k)
          const n = Math.round(param2);
          const p = param1;
          for(let k=0; k<=n; k++) {
             const nCk = factorial(n) / (factorial(k) * factorial(n-k));
             const prob = nCk * Math.pow(p, k) * Math.pow(1-p, n-k);
             data.push({ k, prob });
          }
      } else if (type === 'POISSON') {
          // P(k) = lambda^k * e^-lambda / k!
          const lambda = param1 * 10; // scale for slider
          for(let k=0; k<=20; k++) {
             const prob = (Math.pow(lambda, k) * Math.exp(-lambda)) / factorial(k);
             data.push({ k, prob });
          }
      } else if (type === 'GEOMETRIC') {
          // P(k) = (1-p)^(k-1) * p
          const p = param1;
          for(let k=1; k<=15; k++) {
              const prob = Math.pow(1-p, k-1) * p;
              data.push({ k, prob });
          }
      }
      return data;
  };

  const data = getData();

  return (
    <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 h-full flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-blue-400">Discrete Distributions</h3>
        <select 
            value={type} 
            onChange={(e) => setType(e.target.value as DistType)}
            className="bg-slate-800 text-xs text-white border border-slate-700 rounded px-2 py-1"
        >
            <option value="BINOMIAL">Binomial</option>
            <option value="POISSON">Poisson</option>
            <option value="GEOMETRIC">Geometric</option>
        </select>
      </div>

      <div className="flex-grow w-full min-h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="k" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
            <Bar dataKey="prob" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 p-4 bg-slate-950 rounded border border-slate-800 space-y-4">
          {type === 'BINOMIAL' && (
              <>
                <div>
                    <div className="text-xs text-slate-400 flex justify-between mb-1"><span>Probability (p): {param1.toFixed(2)}</span></div>
                    <input type="range" min="0.01" max="0.99" step="0.01" value={param1} onChange={e => setParam1(Number(e.target.value))} className="w-full accent-blue-500"/>
                </div>
                <div>
                    <div className="text-xs text-slate-400 flex justify-between mb-1"><span>Trials (n): {Math.round(param2)}</span></div>
                    <input type="range" min="5" max="20" step="1" value={param2} onChange={e => setParam2(Number(e.target.value))} className="w-full accent-blue-500"/>
                </div>
              </>
          )}
           {type === 'POISSON' && (
                <div>
                    <div className="text-xs text-slate-400 flex justify-between mb-1"><span>Lambda (λ): {(param1*10).toFixed(1)}</span></div>
                    <input type="range" min="0.1" max="1.5" step="0.1" value={param1} onChange={e => setParam1(Number(e.target.value))} className="w-full accent-blue-500"/>
                </div>
          )}
           {type === 'GEOMETRIC' && (
                <div>
                    <div className="text-xs text-slate-400 flex justify-between mb-1"><span>Probability (p): {param1.toFixed(2)}</span></div>
                    <input type="range" min="0.1" max="0.9" step="0.1" value={param1} onChange={e => setParam1(Number(e.target.value))} className="w-full accent-blue-500"/>
                </div>
          )}
      </div>
    </div>
  );
};