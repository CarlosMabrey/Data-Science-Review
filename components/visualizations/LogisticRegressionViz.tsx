import React, { useState, useEffect, useCallback } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Cell } from 'recharts';
import { Play, RefreshCw } from 'lucide-react';

export const LogisticRegressionViz: React.FC = () => {
  const [data, setData] = useState<{ x: number; y: number; class: number }[]>([]);
  const [weights, setWeights] = useState({ w: 0, b: 0 });
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);

  const generateData = useCallback(() => {
    const newData = [];
    for (let i = 0; i < 30; i++) {
      const x = Math.random() * 10 - 5; // -5 to 5
      // True boundary roughly at x = 0
      const prob = 1 / (1 + Math.exp(-(2 * x + 0.5))); 
      const cls = Math.random() < prob ? 1 : 0;
      newData.push({ x, y: cls, class: cls });
    }
    setData(newData);
    setWeights({ w: 0, b: 0 });
    setEpoch(0);
    setIsTraining(false);
  }, []);

  useEffect(() => {
    generateData();
  }, [generateData]);

  const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

  const trainStep = useCallback(() => {
    const lr = 0.1;
    let dw = 0;
    let db = 0;
    const n = data.length;

    data.forEach(pt => {
      const z = weights.w * pt.x + weights.b;
      const pred = sigmoid(z);
      const error = pred - pt.class;
      dw += error * pt.x;
      db += error;
    });

    setWeights(prev => ({
      w: prev.w - (lr * dw) / n,
      b: prev.b - (lr * db) / n
    }));
    setEpoch(e => e + 1);
  }, [data, weights]);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isTraining && epoch < 200) {
      interval = setInterval(trainStep, 20);
    } else {
      setIsTraining(false);
    }
    return () => clearInterval(interval);
  }, [isTraining, epoch, trainStep]);

  // Generate sigmoid curve points for visualization
  const curveData = [];
  for (let i = -5; i <= 5; i += 0.2) {
    curveData.push({ x: i, y: sigmoid(weights.w * i + weights.b) });
  }

  return (
    <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 h-full flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-blue-400">Logistic Regression (Sigmoid Fit)</h3>
        <div className="flex gap-2">
          <button onClick={() => setIsTraining(!isTraining)} className="p-2 bg-blue-600 rounded text-white text-xs flex items-center gap-1">
            <Play size={14} /> {isTraining ? 'Pause' : 'Train'}
          </button>
          <button onClick={generateData} className="p-2 bg-slate-700 rounded text-white text-xs flex items-center gap-1">
            <RefreshCw size={14} /> Reset
          </button>
        </div>
      </div>

      <div className="flex-grow w-full min-h-[300px] relative">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis type="number" dataKey="x" stroke="#94a3b8" domain={[-6, 6]} />
            <YAxis type="number" dataKey="y" stroke="#94a3b8" domain={[-0.1, 1.1]} />
            <Scatter name="Data" data={data} shape="circle">
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.class === 1 ? '#22c55e' : '#ef4444'} />
              ))}
            </Scatter>
            <Scatter name="Curve" data={curveData} line={{ stroke: '#3b82f6', strokeWidth: 2 }} shape={() => null} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 text-center text-xs text-slate-400 font-mono">
        Epoch: {epoch} | Weight: {weights.w.toFixed(3)} | Bias: {weights.b.toFixed(3)}
      </div>
    </div>
  );
};