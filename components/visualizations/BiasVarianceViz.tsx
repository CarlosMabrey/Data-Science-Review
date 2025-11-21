import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts';

export const BiasVarianceViz: React.FC = () => {
  const [complexity, setComplexity] = useState(1); // 0 = High Bias, 2 = High Variance

  const generateModels = () => {
    const models = [];
    const xVals = [0, 1, 2, 3, 4, 5, 6];
    
    // True Function: Sin wave
    const trueData = xVals.map(x => ({ x, y: Math.sin(x) * 5 + 5 }));

    // Generate 5 hypothetical model fits based on complexity
    for(let m=0; m<5; m++) {
        const modelLine = xVals.map(x => {
            if (complexity === 0) {
                // Underfitting (High Bias): Straight lines ignoring curve
                return { x, y: 5 + (Math.random() - 0.5) * 2 }; 
            } else if (complexity === 1) {
                // Good fit: Close to sine
                return { x, y: Math.sin(x) * 5 + 5 + (Math.random() - 0.5) * 2 };
            } else {
                // Overfitting (High Variance): Wildly swinging lines
                return { x, y: Math.sin(x) * 5 + 5 + (Math.random() - 0.5) * 10 };
            }
        });
        models.push(modelLine);
    }
    return { trueData, models };
  };

  const { trueData, models } = generateModels();

  return (
    <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 h-full flex flex-col">
      <h3 className="text-lg font-semibold text-blue-400 mb-4">Bias-Variance Tradeoff</h3>
      
      <div className="flex-grow w-full min-h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
             <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
             <XAxis type="number" dataKey="x" hide domain={[0, 6]} />
             <YAxis hide domain={[0, 15]} />
             
             {/* True Function */}
             <Line data={trueData} dataKey="y" stroke="#fff" strokeWidth={3} dot={false} />
             
             {/* Model Fits */}
             {models.map((m, i) => (
                 <Line key={i} data={m} dataKey="y" stroke={complexity === 0 ? '#ef4444' : complexity === 1 ? '#22c55e' : '#eab308'} strokeWidth={1} strokeOpacity={0.6} dot={false} />
             ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 p-4 bg-slate-950 rounded border border-slate-800">
         <div className="flex justify-between text-xs text-slate-400 mb-3">
            <span className={`${complexity === 0 ? 'text-white font-bold' : ''}`}>High Bias (Underfit)</span>
            <span className={`${complexity === 1 ? 'text-white font-bold' : ''}`}>Balanced</span>
            <span className={`${complexity === 2 ? 'text-white font-bold' : ''}`}>High Variance (Overfit)</span>
        </div>
        <input type="range" min="0" max="2" step="1" value={complexity} onChange={e => setComplexity(Number(e.target.value))} className="w-full accent-blue-500"/>
        <p className="text-xs text-center mt-2 text-slate-500">
            {complexity === 0 && "Models represent the data poorly (Straight lines on curved data). Consistent but wrong."}
            {complexity === 1 && "Models capture the trend well with moderate variance."}
            {complexity === 2 && "Models fluctuate wildly based on noise. High variance between runs."}
        </p>
      </div>
    </div>
  );
};