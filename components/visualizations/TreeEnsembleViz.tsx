import React, { useState, useEffect } from 'react';
import { Play, RefreshCw, Layers } from 'lucide-react';

// Simulating a 2D classification problem (e.g., concentric circles or moons)
const generateData = () => {
  const points = [];
  for (let i = 0; i < 100; i++) {
    const x = Math.random() * 100;
    const y = Math.random() * 100;
    // Circle class logic: center (50,50), radius 30
    const dist = Math.sqrt((x - 50) ** 2 + (y - 50) ** 2);
    const label = dist < 30 ? 1 : 0;
    // Add some noise
    const noisyLabel = Math.random() > 0.95 ? 1 - label : label;
    points.push({ x, y, label: noisyLabel });
  }
  return points;
};

export const TreeEnsembleViz: React.FC = () => {
  const [points, setPoints] = useState(generateData());
  const [trees, setTrees] = useState<number>(0);
  const [isTraining, setIsTraining] = useState(false);
  
  // Grid for decision boundary visualization
  const gridSize = 20;
  const [grid, setGrid] = useState<number[][]>([]);

  const updateBoundary = (treeCount: number) => {
    const newGrid = [];
    for (let i = 0; i < 100; i += gridSize) {
      const row = [];
      for (let j = 0; j < 100; j += gridSize) {
        // Simulate model prediction based on trees
        // 0 trees: random, 10 trees: crude box, 50 trees: circular approx
        const cx = i + gridSize/2;
        const cy = j + gridSize/2;
        const dist = Math.sqrt((cx - 50)**2 + (cy - 50)**2);
        
        let prob = 0;
        if (treeCount === 0) prob = 0.5;
        else {
           // Slowly converge to the true circle as trees increase
           // Noise/Error decreases as trees increase
           const error = 30 * Math.exp(-treeCount / 10); 
           const decisionDist = 30 + (Math.random() - 0.5) * error;
           prob = dist < decisionDist ? 0.9 : 0.1;
        }
        row.push(prob);
      }
      newGrid.push(row);
    }
    setGrid(newGrid);
  };

  useEffect(() => {
    updateBoundary(trees);
  }, [trees]);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isTraining && trees < 50) {
      interval = setInterval(() => setTrees(t => t + 1), 100);
    } else {
      setIsTraining(false);
    }
    return () => clearInterval(interval);
  }, [isTraining, trees]);

  return (
    <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 h-full flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-blue-400">Random Forest (Voting)</h3>
        <div className="flex gap-2">
           <button onClick={() => setIsTraining(!isTraining)} className="p-2 bg-blue-600 rounded text-white text-xs flex items-center gap-1">
            <Play size={14} /> {isTraining ? 'Pause' : 'Add Trees'}
          </button>
          <button onClick={() => { setTrees(0); setPoints(generateData()); }} className="p-2 bg-slate-700 rounded text-white text-xs flex items-center gap-1">
            <RefreshCw size={14} /> Reset
          </button>
        </div>
      </div>

      <div className="relative flex-grow bg-slate-950 border border-slate-800 rounded overflow-hidden">
         {/* Decision Boundary Grid */}
         <div className="absolute inset-0 grid" style={{ gridTemplateColumns: `repeat(${100/gridSize}, 1fr)`, gridTemplateRows: `repeat(${100/gridSize}, 1fr)` }}>
            {grid.map((row, ri) => (
                row.map((prob, ci) => (
                    <div key={`${ri}-${ci}`} style={{ backgroundColor: `rgba(34, 197, 94, ${prob})`, opacity: 0.3 }} />
                ))
            ))}
         </div>

         {/* Points */}
         {points.map((p, i) => (
             <div 
                key={i} 
                className={`absolute w-2 h-2 rounded-full ${p.label === 1 ? 'bg-green-500' : 'bg-red-500'}`}
                style={{ left: `${p.x}%`, top: `${p.y}%`, transform: 'translate(-50%, -50%)' }}
             />
         ))}
      </div>
      
      <div className="mt-4 flex justify-center items-center gap-4 text-sm text-slate-300">
          <div className="flex items-center gap-2">
              <Layers size={18} className="text-purple-400" />
              <span className="font-bold text-xl">{trees}</span>
              <span className="text-slate-500">Trees in Ensemble</span>
          </div>
          <div className="text-xs text-slate-500">
              More trees = Smoother, more robust boundary (Lower Variance)
          </div>
      </div>
    </div>
  );
};