import React, { useState, useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Line, ComposedChart } from 'recharts';

// Generate noisy sine wave data
const generateData = () => {
  const data = [];
  for (let i = 0; i <= 10; i += 0.5) {
    data.push({ x: i, y: Math.sin(i) + (Math.random() - 0.5) * 0.5 });
  }
  return data;
};

export const RegularizationViz: React.FC = () => {
  const [degree, setDegree] = useState(3);
  const [alpha, setAlpha] = useState(0); // Regularization strength
  const [data] = useState(generateData());

  // Simulate polynomial fit curve
  const curveData = useMemo(() => {
    const pts = [];
    for (let x = 0; x <= 10; x += 0.2) {
       // Fake polynomial logic for visualization
       // If degree is high and alpha is low -> Overfit (pass through all points roughly)
       // If alpha is high -> smooth out (dampen higher order terms)
       
       let y = Math.sin(x); // True function
       
       if (degree > 5 && alpha < 2) {
           // Add high frequency noise to simulate overfitting
           y += Math.sin(x * degree) * 0.1 * (10 - alpha);
       } else if (alpha > 5) {
           // Underfit / Dampened
           y = y * (1 - alpha/20); 
       }
       
       pts.push({ x, y });
    }
    return pts;
  }, [degree, alpha]);

  return (
    <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 h-full flex flex-col">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-blue-400">Overfitting vs Regularization</h3>
      </div>

      <div className="flex-grow w-full min-h-[250px]">
         <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={curveData} margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="x" type="number" domain={[0, 10]} stroke="#94a3b8"/>
            <YAxis domain={[-2, 2]} stroke="#94a3b8"/>
            <Scatter data={data} fill="#ef4444" />
            <Line type="monotone" dataKey="y" stroke="#38bdf8" strokeWidth={2} dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-6 p-4 bg-slate-950 rounded border border-slate-800">
          <div>
              <div className="flex justify-between text-xs text-slate-400 mb-2">
                  <span>Polynomial Degree: {degree}</span>
                  <span>{degree < 3 ? 'Underfit' : degree > 8 ? 'Overfit' : 'Good'}</span>
              </div>
              <input type="range" min="1" max="15" value={degree} onChange={e => setDegree(Number(e.target.value))} className="w-full accent-blue-500"/>
          </div>
          <div>
               <div className="flex justify-between text-xs text-slate-400 mb-2">
                  <span>Regularization (L2): {alpha}</span>
                  <span>{alpha > 0 ? 'Constraining weights' : 'None'}</span>
              </div>
              <input type="range" min="0" max="10" value={alpha} onChange={e => setAlpha(Number(e.target.value))} className="w-full accent-green-500"/>
          </div>
      </div>
    </div>
  );
};