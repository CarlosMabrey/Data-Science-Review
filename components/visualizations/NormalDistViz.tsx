import React, { useState, useMemo } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip } from 'recharts';

export const NormalDistViz: React.FC = () => {
  const [mean, setMean] = useState(0);
  const [stdDev, setStdDev] = useState(1);
  const [sampleSize, setSampleSize] = useState(1000);

  const data = useMemo(() => {
    const result = [];
    // Plot from -5 to 5
    const minX = -5;
    const maxX = 5;
    const step = 0.1;

    for (let x = minX; x <= maxX; x += step) {
      // PDF formula: (1 / (sigma * sqrt(2*pi))) * e^(-0.5 * ((x-mu)/sigma)^2)
      const exponent = -0.5 * Math.pow((x - mean) / stdDev, 2);
      const y = (1 / (stdDev * Math.sqrt(2 * Math.PI))) * Math.exp(exponent);
      result.push({ x: parseFloat(x.toFixed(2)), y });
    }
    return result;
  }, [mean, stdDev]);

  return (
    <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 h-full flex flex-col">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-blue-400">Normal Distribution (Gaussian)</h3>
        <p className="text-slate-400 text-xs mt-1">
            Probability Density Function. Visualize how Mean (μ) and Standard Deviation (σ) affect the curve.
        </p>
      </div>

      <div className="flex-grow w-full min-h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="x" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f1f5f9' }} 
                itemStyle={{ color: '#38bdf8' }}
            />
            <Area type="monotone" dataKey="y" stroke="#38bdf8" fill="#38bdf8" fillOpacity={0.3} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-6 space-y-4 p-4 bg-slate-950 rounded border border-slate-800">
        <div>
          <div className="flex justify-between text-xs text-slate-400 mb-1">
            <span>Mean (μ): {mean}</span>
          </div>
          <input 
            type="range" min="-3" max="3" step="0.1" 
            value={mean} onChange={(e) => setMean(parseFloat(e.target.value))}
            className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
          />
        </div>
        <div>
          <div className="flex justify-between text-xs text-slate-400 mb-1">
            <span>Standard Deviation (σ): {stdDev}</span>
          </div>
          <input 
            type="range" min="0.2" max="3" step="0.1" 
            value={stdDev} onChange={(e) => setStdDev(parseFloat(e.target.value))}
            className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
          />
        </div>
      </div>
    </div>
  );
};
