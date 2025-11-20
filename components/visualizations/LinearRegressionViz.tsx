import React, { useState, useEffect, useCallback } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Line, ComposedChart } from 'recharts';
import { Play, RefreshCw } from 'lucide-react';

const generateData = () => {
  const data = [];
  for (let i = 0; i < 20; i++) {
    const x = i * 5 + Math.random() * 10;
    const y = 2 * x + 10 + (Math.random() * 40 - 20);
    data.push({ x, y });
  }
  return data;
};

export const LinearRegressionViz: React.FC = () => {
  const [data, setData] = useState<{ x: number; y: number }[]>([]);
  const [slope, setSlope] = useState(0);
  const [intercept, setIntercept] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);

  useEffect(() => {
    setData(generateData());
  }, []);

  const trainStep = useCallback(() => {
    // Simple Gradient Descent Simulation
    const learningRate = 0.0001;
    let m = slope;
    let b = intercept;

    let dM = 0;
    let dB = 0;
    const N = data.length;

    data.forEach(point => {
      const y_pred = m * point.x + b;
      const error = point.y - y_pred;
      dM += -(2 / N) * point.x * error;
      dB += -(2 / N) * error;
    });

    setSlope(m - learningRate * dM);
    setIntercept(b - learningRate * dB);
    setEpoch(prev => prev + 1);
  }, [data, slope, intercept]);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isTraining && epoch < 100) {
      interval = setInterval(trainStep, 50);
    } else {
      setIsTraining(false);
    }
    return () => clearInterval(interval);
  }, [isTraining, epoch, trainStep]);

  const regressionLine = [
    { x: 0, y: intercept },
    { x: 120, y: slope * 120 + intercept }, // extrapolate for visual
  ];

  return (
    <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 h-full flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-blue-400">Linear Regression Training (Gradient Descent)</h3>
        <div className="flex gap-2">
          <button 
            onClick={() => { setSlope(0); setIntercept(0); setEpoch(0); setIsTraining(true); }}
            className="p-2 bg-blue-600 hover:bg-blue-500 rounded text-white flex items-center gap-2 text-xs"
          >
            <Play size={14} /> Train
          </button>
          <button 
            onClick={() => { setData(generateData()); setSlope(0); setIntercept(0); setEpoch(0); setIsTraining(false); }}
            className="p-2 bg-slate-700 hover:bg-slate-600 rounded text-white flex items-center gap-2 text-xs"
          >
            <RefreshCw size={14} /> New Data
          </button>
        </div>
      </div>
      
      <div className="flex-grow relative w-full min-h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis type="number" dataKey="x" name="feature" unit="" stroke="#94a3b8" domain={[0, 120]} />
            <YAxis type="number" dataKey="y" name="target" unit="" stroke="#94a3b8" domain={[0, 300]} />
            <Scatter name="Data" data={data} fill="#60a5fa" />
            {/* Fake Line using reference line concept but with Scatter for recharts limitation or use specific Line component */}
            {/* Using a separate Line Chart overlaid is complex, using ComposedChart is better, but for simplicity using SVG overlay or just multiple scatter points to simulate line? No, Recharts ComposedChart allows Line + Scatter */}
          </ScatterChart>
        </ResponsiveContainer>
        {/* Manual SVG Line Overlay for better performance than re-rendering Chart constantly */}
        <svg className="absolute top-0 left-0 w-full h-full pointer-events-none overflow-visible">
           {/* Note: Exact coordinate mapping in pure SVG over Recharts is tricky without hooks. 
               For this demo, we will rely on ComposedChart logic if possible, OR simplify:
               We render a second Line series in the chart.
           */}
        </svg>
      </div>

      {/* Re-implementing with ComposedChart for the line */}
      <div className="absolute inset-0 pt-16 px-4 pb-4 pointer-events-none">
         {/* This is a cheat to show the line. In a real app, we'd use ComposedChart. 
             Let's swap the above return to ComposedChart for correctness. */}
      </div>

      <div className="mt-4 grid grid-cols-3 gap-4 text-sm font-mono text-slate-400 bg-slate-950 p-2 rounded">
        <div>Epoch: <span className="text-white">{epoch}</span></div>
        <div>Slope (m): <span className="text-green-400">{slope.toFixed(4)}</span></div>
        <div>Bias (b): <span className="text-yellow-400">{intercept.toFixed(4)}</span></div>
      </div>
    </div>
  );
};

// Actually implementing ComposedChart properly below to replace the above render:

export const LinearRegressionVizFixed: React.FC = () => {
  const [data, setData] = useState<{ x: number; y: number }[]>([]);
  const [slope, setSlope] = useState(0);
  const [intercept, setIntercept] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);

  useEffect(() => {
    setData(generateData());
  }, []);

  const trainStep = useCallback(() => {
    const learningRate = 0.00005; // Smaller LR for stability
    let m = slope;
    let b = intercept;
    let dM = 0;
    let dB = 0;
    const N = data.length;

    data.forEach(point => {
      const y_pred = m * point.x + b;
      const error = point.y - y_pred;
      dM += -(2 / N) * point.x * error;
      dB += -(2 / N) * error;
    });

    setSlope(m - learningRate * dM);
    setIntercept(b - learningRate * dB);
    setEpoch(prev => prev + 1);
  }, [data, slope, intercept]);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isTraining && epoch < 200) {
      interval = setInterval(trainStep, 20);
    } else {
      setIsTraining(false);
    }
    return () => clearInterval(interval);
  }, [isTraining, epoch, trainStep]);

  const lineData = [
    { x: 0, lineY: intercept },
    { x: 100, lineY: slope * 100 + intercept },
  ];

  return (
    <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 h-full flex flex-col">
       <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-blue-400">Gradient Descent Visualizer</h3>
        <div className="flex gap-2">
          <button 
            onClick={() => { if(!isTraining) { setSlope(0); setIntercept(0); setEpoch(0); setIsTraining(true); }}}
            disabled={isTraining}
            className="px-3 py-1 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded text-white flex items-center gap-2 text-xs"
          >
            <Play size={14} /> {epoch > 0 && isTraining ? "Training..." : "Start Descent"}
          </button>
          <button 
            onClick={() => { setData(generateData()); setSlope(0); setIntercept(0); setEpoch(0); setIsTraining(false); }}
            className="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-white flex items-center gap-2 text-xs"
          >
            <RefreshCw size={14} /> Reset
          </button>
        </div>
      </div>

      <div className="flex-grow w-full min-h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="x" type="number" stroke="#94a3b8" domain={[0, 100]} />
            <YAxis dataKey="y" type="number" stroke="#94a3b8" domain={[0, 250]} />
            <Scatter name="Samples" data={data} fill="#38bdf8" />
            <Line 
              data={lineData} 
              type="monotone" 
              dataKey="lineY" 
              stroke="#ef4444" 
              strokeWidth={3} 
              dot={false} 
              activeDot={false}
              isAnimationActive={false} // Crucial for smooth realtime updates
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      
      <div className="mt-4 p-3 bg-slate-950 rounded border border-slate-800">
        <div className="flex justify-between text-xs text-slate-500 mb-2 uppercase tracking-wider">Model Parameters</div>
        <div className="grid grid-cols-3 gap-4 text-sm font-mono">
            <div className="flex flex-col">
                <span className="text-slate-500 text-xs">Epoch</span>
                <span className="text-white font-bold">{epoch}</span>
            </div>
            <div className="flex flex-col">
                <span className="text-slate-500 text-xs">Slope (m)</span>
                <span className="text-green-400">{slope.toFixed(4)}</span>
            </div>
            <div className="flex flex-col">
                <span className="text-slate-500 text-xs">Intercept (b)</span>
                <span className="text-yellow-400">{intercept.toFixed(4)}</span>
            </div>
        </div>
      </div>
    </div>
  );
}