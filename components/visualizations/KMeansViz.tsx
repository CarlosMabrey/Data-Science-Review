import React, { useState, useEffect } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Cell } from 'recharts';
import { Play, RefreshCw, Plus, Minus } from 'lucide-react';

interface Point { x: number; y: number; cluster: number }
interface Centroid { x: number; y: number; color: string }

const COLORS = ['#ef4444', '#22c55e', '#3b82f6', '#eab308', '#a855f7'];

const generatePoints = () => {
  const points: Point[] = [];
  for (let i = 0; i < 100; i++) {
    points.push({
      x: Math.random() * 100,
      y: Math.random() * 100,
      cluster: -1,
    });
  }
  return points;
};

export const KMeansViz: React.FC = () => {
  const [k, setK] = useState(3);
  const [points, setPoints] = useState<Point[]>([]);
  const [centroids, setCentroids] = useState<Centroid[]>([]);
  const [step, setStep] = useState(0);
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    setPoints(generatePoints());
    initializeCentroids(3);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const initializeCentroids = (count: number) => {
    const newCentroids: Centroid[] = [];
    for (let i = 0; i < count; i++) {
      newCentroids.push({
        x: Math.random() * 100,
        y: Math.random() * 100,
        color: COLORS[i % COLORS.length],
      });
    }
    setCentroids(newCentroids);
    setStep(0);
    
    // Reset points
    setPoints(prev => prev.map(p => ({ ...p, cluster: -1 })));
  };

  const assignClusters = () => {
    let changed = false;
    const newPoints = points.map(p => {
      let minDist = Infinity;
      let clusterIndex = -1;
      
      centroids.forEach((c, idx) => {
        const dist = Math.sqrt(Math.pow(p.x - c.x, 2) + Math.pow(p.y - c.y, 2));
        if (dist < minDist) {
          minDist = dist;
          clusterIndex = idx;
        }
      });

      if (p.cluster !== clusterIndex) changed = true;
      return { ...p, cluster: clusterIndex };
    });
    setPoints(newPoints);
    return changed;
  };

  const updateCentroids = () => {
    const newCentroids = centroids.map((c, idx) => {
      const clusterPoints = points.filter(p => p.cluster === idx);
      if (clusterPoints.length === 0) return c;
      
      const avgX = clusterPoints.reduce((sum, p) => sum + p.x, 0) / clusterPoints.length;
      const avgY = clusterPoints.reduce((sum, p) => sum + p.y, 0) / clusterPoints.length;
      return { ...c, x: avgX, y: avgY };
    });
    setCentroids(newCentroids);
  };

  const runStep = () => {
    if (step % 2 === 0) {
      // Assignment Step
      assignClusters();
    } else {
      // Update Step
      updateCentroids();
    }
    setStep(s => s + 1);
  };

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isRunning) {
      interval = setInterval(() => {
        runStep();
        // Stop condition check could be here, but for viz we just let it run a bit or user stops
        if(step > 20) setIsRunning(false);
      }, 800);
    }
    return () => clearInterval(interval);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isRunning, step, points, centroids]);

  return (
    <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 h-full flex flex-col">
       <div className="flex flex-wrap justify-between items-center mb-4 gap-2">
        <h3 className="text-lg font-semibold text-blue-400">K-Means Clustering</h3>
        <div className="flex gap-2 items-center">
            <div className="flex items-center bg-slate-800 rounded px-2 mr-2">
                <span className="text-xs text-slate-400 mr-2">K={k}</span>
                <button onClick={() => { if(k<5) { setK(k+1); initializeCentroids(k+1); }}} className="p-1 hover:text-white text-slate-400"><Plus size={12}/></button>
                <button onClick={() => { if(k>1) { setK(k-1); initializeCentroids(k-1); }}} className="p-1 hover:text-white text-slate-400"><Minus size={12}/></button>
            </div>
          <button 
            onClick={() => setIsRunning(!isRunning)}
            className={`px-3 py-1 rounded text-white flex items-center gap-2 text-xs ${isRunning ? 'bg-red-600 hover:bg-red-500' : 'bg-blue-600 hover:bg-blue-500'}`}
          >
            <Play size={14} /> {isRunning ? "Pause" : "Run"}
          </button>
          <button 
            onClick={() => { setPoints(generatePoints()); initializeCentroids(k); }}
            className="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-white flex items-center gap-2 text-xs"
          >
            <RefreshCw size={14} /> Reset
          </button>
        </div>
      </div>

      <div className="flex-grow w-full min-h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis type="number" dataKey="x" name="x" stroke="#94a3b8" domain={[0, 100]} />
            <YAxis type="number" dataKey="y" name="y" stroke="#94a3b8" domain={[0, 100]} />
            <Scatter name="Points" data={points} shape="circle">
                {points.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.cluster === -1 ? '#64748b' : COLORS[entry.cluster]} />
                ))}
            </Scatter>
            <Scatter name="Centroids" data={centroids} shape="cross">
                 {centroids.map((entry, index) => (
                    <Cell key={`centroid-${index}`} fill={entry.color} strokeWidth={4} />
                ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

       <div className="mt-4 text-center text-slate-400 text-sm">
        Step: <span className="text-white font-bold">{Math.floor(step / 2)}</span> 
        <span className="mx-2">|</span> 
        State: <span className="text-blue-400">{step % 2 === 0 ? 'Assign Clusters' : 'Update Centroids'}</span>
      </div>
    </div>
  );
}