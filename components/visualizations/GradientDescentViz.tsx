import React, { useState, useEffect } from 'react';
import { Play, RefreshCw } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, ReferenceDot } from 'recharts';

export const GradientDescentViz: React.FC = () => {
  const [x, setX] = useState(-9); // Start far left
  const [isRunning, setIsRunning] = useState(false);
  const [history, setHistory] = useState<{x: number, y: number}[]>([]);

  const data = [];
  for (let i = -10; i <= 10; i += 0.5) {
    data.push({ val: i, cost: i * i }); // J(w) = w^2
  }

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isRunning) {
      interval = setInterval(() => {
        setX(prev => {
            // Gradient of w^2 is 2w
            // w_new = w - alpha * 2w
            const alpha = 0.1;
            const grad = 2 * prev;
            const next = prev - alpha * grad;
            
            if (Math.abs(grad) < 0.01) {
                setIsRunning(false);
                return prev;
            }
            
            setHistory(h => [...h, { x: next, y: next*next }]);
            return next;
        });
      }, 100);
    }
    return () => clearInterval(interval);
  }, [isRunning]);

  const reset = () => {
      setX(Math.random() > 0.5 ? -9 : 9);
      setIsRunning(false);
      setHistory([]);
  };

  return (
    <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 h-full flex flex-col">
       <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-blue-400">Gradient Descent Optimization</h3>
        <div className="flex gap-2">
          <button onClick={() => setIsRunning(!isRunning)} className="p-2 bg-blue-600 rounded text-white text-xs flex items-center gap-1">
            <Play size={14} /> {isRunning ? 'Pause' : 'Descend'}
          </button>
          <button onClick={reset} className="p-2 bg-slate-700 rounded text-white text-xs flex items-center gap-1">
            <RefreshCw size={14} /> Reset
          </button>
        </div>
      </div>

      <div className="flex-grow w-full min-h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
            <XAxis dataKey="val" hide />
            <YAxis hide />
            <Area type="monotone" dataKey="cost" stroke="#64748b" fill="#1e293b" />
            <ReferenceDot x={x} y={x*x} r={6} fill="#ef4444" stroke="white" />
            {history.map((pt, i) => (
                <ReferenceDot key={i} x={pt.x} y={pt.y} r={2} fill="#ef4444" fillOpacity={0.3} />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 p-3 bg-slate-950 rounded font-mono text-xs text-slate-300">
          <div className="flex justify-between">
            <span>Parameter (w): <span className="text-red-400">{x.toFixed(3)}</span></span>
            <span>Gradient: <span className="text-yellow-400">{(2*x).toFixed(3)}</span></span>
            <span>Cost J(w): <span className="text-blue-400">{(x*x).toFixed(3)}</span></span>
          </div>
          <div className="mt-2 text-slate-500">
              Minimizing $J(w) = w^2$. The ball rolls down the slope (gradient) towards the minimum.
          </div>
      </div>
    </div>
  );
};