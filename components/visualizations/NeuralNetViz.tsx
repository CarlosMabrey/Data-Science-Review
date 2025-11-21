import React, { useState } from 'react';
import { Play, RefreshCw, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';

export const NeuralNetViz: React.FC = () => {
  const [input, setInput] = useState(0.5);
  const [target, setTarget] = useState(1.0);
  const [weights, setWeights] = useState({ w1: 0.5, w2: -0.5, w3: 0.8, w4: 0.2 }); // w1/w2 to hidden, w3/w4 to output
  const [gradients, setGradients] = useState<{ [key: string]: number }>({});
  const [step, setStep] = useState<'idle' | 'forward' | 'backward'>('idle');

  // Simplified Network: 1 Input -> 2 Hidden -> 1 Output
  // Forward
  const h1 = input * weights.w1;
  const h2 = input * weights.w2;
  const h1_act = Math.max(0, h1); // ReLU
  const h2_act = Math.max(0, h2); // ReLU
  const out = h1_act * weights.w3 + h2_act * weights.w4;
  const loss = 0.5 * (target - out) ** 2;

  const doBackprop = () => {
    setStep('backward');
    const lr = 0.1;
    
    // dLoss/dOut
    const dLoss_dOut = -(target - out);
    
    // dOut/dW3 = h1_act
    const dLoss_dW3 = dLoss_dOut * h1_act;
    const dLoss_dW4 = dLoss_dOut * h2_act;

    // dOut/dH1_act = w3
    const dLoss_dH1_act = dLoss_dOut * weights.w3;
    const dLoss_dH2_act = dLoss_dOut * weights.w4;

    // dH1_act/dH1 (ReLU derivative)
    const dH1_act_dH1 = h1 > 0 ? 1 : 0;
    const dH2_act_dH2 = h2 > 0 ? 1 : 0;

    const dLoss_dW1 = dLoss_dH1_act * dH1_act_dH1 * input;
    const dLoss_dW2 = dLoss_dH2_act * dH2_act_dH2 * input;

    setGradients({ w1: dLoss_dW1, w2: dLoss_dW2, w3: dLoss_dW3, w4: dLoss_dW4 });

    setTimeout(() => {
        setWeights({
            w1: weights.w1 - lr * dLoss_dW1,
            w2: weights.w2 - lr * dLoss_dW2,
            w3: weights.w3 - lr * dLoss_dW3,
            w4: weights.w4 - lr * dLoss_dW4,
        });
        setStep('idle');
        setGradients({});
    }, 2000);
  };

  return (
    <div className="bg-slate-900 p-6 rounded-lg border border-slate-800 h-full flex flex-col relative overflow-hidden">
      <div className="flex justify-between items-center mb-8 z-10">
        <h3 className="text-lg font-semibold text-purple-400">Backpropagation Visualization</h3>
        <button onClick={doBackprop} disabled={step !== 'idle'} className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded text-white text-xs flex items-center gap-2">
           {step === 'backward' ? 'Updating Weights...' : 'Step Backprop'}
        </button>
      </div>

      {/* Network Graph SVG */}
      <svg className="w-full h-[300px] z-0 pointer-events-none">
          {/* Edges */}
          <line x1="10%" y1="50%" x2="50%" y2="30%" stroke="#475569" strokeWidth="2" />
          <line x1="10%" y1="50%" x2="50%" y2="70%" stroke="#475569" strokeWidth="2" />
          <line x1="50%" y1="30%" x2="90%" y2="50%" stroke="#475569" strokeWidth="2" />
          <line x1="50%" y1="70%" x2="90%" y2="50%" stroke="#475569" strokeWidth="2" />
          
          {/* Gradients (Animated Flow Backward) */}
          {step === 'backward' && (
              <>
                <circle r="4" fill="red"><animateMotion dur="1s" repeatCount="1" path="M 90% 50% L 50% 30%" /></circle>
                <circle r="4" fill="red"><animateMotion dur="1s" repeatCount="1" path="M 90% 50% L 50% 70%" /></circle>
                <circle r="4" fill="red" begin="0.5s"><animateMotion dur="1s" repeatCount="1" path="M 50% 30% L 10% 50%" /></circle>
                <circle r="4" fill="red" begin="0.5s"><animateMotion dur="1s" repeatCount="1" path="M 50% 70% L 10% 50%" /></circle>
              </>
          )}
      </svg>

      {/* Nodes */}
      <div className="absolute inset-0 flex justify-between items-center px-12 pointer-events-none">
          {/* Input Layer */}
          <div className="flex flex-col items-center">
              <div className="w-12 h-12 rounded-full bg-blue-500 flex items-center justify-center text-white font-bold text-xs shadow-lg shadow-blue-500/20">X</div>
              <span className="mt-2 text-xs text-slate-400">Input: {input}</span>
          </div>

          {/* Hidden Layer */}
          <div className="flex flex-col gap-20">
              <div className="flex flex-col items-center">
                  <div className="w-12 h-12 rounded-full bg-slate-700 flex items-center justify-center text-white font-bold text-xs border border-slate-600">H1</div>
                  <span className="text-[10px] text-slate-500">w1: {weights.w1.toFixed(2)}</span>
                  {gradients.w1 && <span className="text-[10px] text-red-400 animate-pulse">dL/dw: {gradients.w1.toFixed(3)}</span>}
              </div>
               <div className="flex flex-col items-center">
                  <div className="w-12 h-12 rounded-full bg-slate-700 flex items-center justify-center text-white font-bold text-xs border border-slate-600">H2</div>
                  <span className="text-[10px] text-slate-500">w2: {weights.w2.toFixed(2)}</span>
                  {gradients.w2 && <span className="text-[10px] text-red-400 animate-pulse">dL/dw: {gradients.w2.toFixed(3)}</span>}
              </div>
          </div>

          {/* Output Layer */}
          <div className="flex flex-col items-center">
              <div className="w-12 h-12 rounded-full bg-green-600 flex items-center justify-center text-white font-bold text-xs shadow-lg shadow-green-500/20">Y</div>
              <span className="mt-2 text-xs text-slate-400">Out: {out.toFixed(2)}</span>
              <span className="text-xs text-red-400">Loss: {loss.toFixed(3)}</span>
          </div>
      </div>
    </div>
  );
};