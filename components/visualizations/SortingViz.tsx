import React, { useState, useEffect, useRef } from 'react';
import { BarChart, Bar, Cell, ResponsiveContainer } from 'recharts';
import { Play, RefreshCw, Info } from 'lucide-react';

type Algorithm = 'BUBBLE' | 'SELECTION' | 'INSERTION' | 'QUICK' | 'MERGE';

const ALGO_INFO: Record<Algorithm, { name: string; complexity: string; desc: string; detail: string }> = {
  BUBBLE: { 
    name: 'Bubble Sort', 
    complexity: 'O(n²)', 
    desc: 'Repeatedly swaps adjacent elements if they are in the wrong order.',
    detail: 'The algorithm steps through the list, compares adjacent elements and swaps them if they are in the wrong order. The pass through the list is repeated until the list is sorted. With each pass, the largest unsorted element "bubbles" to its correct position at the end of the array.'
  },
  SELECTION: { 
    name: 'Selection Sort', 
    complexity: 'O(n²)', 
    desc: 'Repeatedly finds the minimum element and moves it to the beginning.',
    detail: 'The list is divided into two parts: the sublist of items already sorted (left) and the sublist of items remaining to be sorted (right). In each iteration, it finds the smallest element (highlighted purple) in the unsorted sublist and swaps it with the leftmost unsorted element.'
  },
  INSERTION: { 
    name: 'Insertion Sort', 
    complexity: 'O(n²)', 
    desc: 'Builds the sorted array one item at a time by shifting elements.',
    detail: 'It works like sorting playing cards in your hands. The array is virtually split into a sorted and an unsorted part. Values from the unsorted part are picked and placed at the correct position in the sorted part by shifting larger elements to the right.'
  },
  QUICK: { 
    name: 'Quick Sort', 
    complexity: 'O(n log n)', 
    desc: 'Divides array into partitions around a pivot element.',
    detail: 'It picks a "pivot" element (highlighted purple) and partitions the array so all smaller elements come before the pivot and all larger elements come after. The pivot is then in its final sorted position. The process is recursively applied to the sub-arrays.'
  },
  MERGE: { 
    name: 'Merge Sort', 
    complexity: 'O(n log n)', 
    desc: 'Recursively halves the array and merges sorted halves.',
    detail: 'A divide-and-conquer algorithm that divides the input array into two halves, calls itself for the two halves, and then merges the two sorted halves. The visualization shows the merging process overwriting the original array values at the write index (red).'
  },
};

export const SortingViz: React.FC = () => {
  const [array, setArray] = useState<number[]>([]);
  const [algorithm, setAlgorithm] = useState<Algorithm>('BUBBLE');
  const [sorting, setSorting] = useState(false);
  const [compareIndices, setCompareIndices] = useState<number[]>([]);
  const [swapIndices, setSwapIndices] = useState<number[]>([]);
  const [sortedIndices, setSortedIndices] = useState<number[]>([]);
  const [specialIndices, setSpecialIndices] = useState<number[]>([]); // For Pivot or Min
  
  // Refs for async control
  const sortingRef = useRef(false);
  const stopRef = useRef(false);

  useEffect(() => {
    resetArray();
    return () => { stopRef.current = true; };
  }, []);

  const resetArray = () => {
    stopRef.current = true;
    setTimeout(() => {
        const arr = Array.from({ length: 30 }, () => Math.floor(Math.random() * 90) + 10);
        setArray(arr);
        setSorting(false);
        setCompareIndices([]);
        setSwapIndices([]);
        setSortedIndices([]);
        setSpecialIndices([]);
        stopRef.current = false;
        sortingRef.current = false;
    }, 100);
  };

  const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

  const handleSort = async () => {
    if (sorting) return;
    setSorting(true);
    sortingRef.current = true;
    stopRef.current = false;
    setSortedIndices([]);
    setSpecialIndices([]);

    const arr = [...array];

    try {
        switch (algorithm) {
            case 'BUBBLE': await bubbleSort(arr); break;
            case 'SELECTION': await selectionSort(arr); break;
            case 'INSERTION': await insertionSort(arr); break;
            case 'QUICK': await quickSort(arr, 0, arr.length - 1); break;
            case 'MERGE': await mergeSort(arr, 0, arr.length - 1); break;
        }
        if (!stopRef.current) {
             setSortedIndices(arr.map((_, i) => i));
             setSpecialIndices([]);
        }
    } catch (e) {
        // Stopped
    }

    setSorting(false);
    sortingRef.current = false;
    setCompareIndices([]);
    setSwapIndices([]);
    setSpecialIndices([]);
  };

  // --- Algorithms ---

  const bubbleSort = async (arr: number[]) => {
    const n = arr.length;
    for (let i = 0; i < n - 1; i++) {
      if (stopRef.current) return;
      for (let j = 0; j < n - i - 1; j++) {
        if (stopRef.current) return;
        setCompareIndices([j, j + 1]);
        await delay(30);
        
        if (arr[j] > arr[j + 1]) {
          setSwapIndices([j, j + 1]);
          [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
          setArray([...arr]);
          await delay(30);
        }
        setSwapIndices([]);
      }
      setSortedIndices(prev => [...prev, n - 1 - i]);
    }
    setSortedIndices(prev => [...prev, 0]);
  };

  const selectionSort = async (arr: number[]) => {
      const n = arr.length;
      for (let i = 0; i < n; i++) {
          if (stopRef.current) return;
          let minIdx = i;
          setSpecialIndices([minIdx]); // Highlight current min candidate
          
          for (let j = i + 1; j < n; j++) {
              if (stopRef.current) return;
              setCompareIndices([j]); // Scan
              await delay(20);
              
              if (arr[j] < arr[minIdx]) {
                  minIdx = j;
                  setSpecialIndices([minIdx]); // New min found
                  await delay(20);
              }
          }
          
          if (minIdx !== i) {
              setSwapIndices([i, minIdx]);
              [arr[i], arr[minIdx]] = [arr[minIdx], arr[i]];
              setArray([...arr]);
              await delay(50);
              setSwapIndices([]);
          }
          setSortedIndices(prev => [...prev, i]);
      }
  };

  const insertionSort = async (arr: number[]) => {
      const n = arr.length;
      setSortedIndices([0]); 
      for (let i = 1; i < n; i++) {
          if (stopRef.current) return;
          let key = arr[i];
          let j = i - 1;
          
          setSpecialIndices([i]); // Element being inserted
          
          while (j >= 0 && arr[j] > key) {
              if (stopRef.current) return;
              setCompareIndices([j]);
              await delay(20);

              setSwapIndices([j, j + 1]); // Shift visual
              arr[j + 1] = arr[j];
              setArray([...arr]);
              await delay(30);
              j = j - 1;
          }
          arr[j + 1] = key;
          setArray([...arr]);
          setSwapIndices([j + 1]); // Place key
          await delay(30);
          
          setSwapIndices([]);
          setCompareIndices([]);
          setSpecialIndices([]);
          setSortedIndices(Array.from({length: i + 1}, (_, k) => k));
      }
  };

  const quickSort = async (arr: number[], low: number, high: number) => {
      if (low < high) {
          if (stopRef.current) return;
          const pi = await partition(arr, low, high);
          
          setSortedIndices(prev => [...prev, pi]); // Pivot is sorted

          await quickSort(arr, low, pi - 1);
          await quickSort(arr, pi + 1, high);
      } else if (low === high) {
          setSortedIndices(prev => [...prev, low]);
      }
  };

  const partition = async (arr: number[], low: number, high: number) => {
      const pivot = arr[high];
      setSpecialIndices([high]); // Highlight Pivot
      let i = (low - 1);
      
      for (let j = low; j <= high - 1; j++) {
          if (stopRef.current) return -1;
          setCompareIndices([j]);
          await delay(20);
          
          if (arr[j] < pivot) {
              i++;
              setSwapIndices([i, j]);
              [arr[i], arr[j]] = [arr[j], arr[i]];
              setArray([...arr]);
              await delay(30);
              setSwapIndices([]);
          }
      }
      setSwapIndices([i + 1, high]);
      [arr[i + 1], arr[high]] = [arr[high], arr[i + 1]];
      setArray([...arr]);
      await delay(30);
      setSwapIndices([]);
      setSpecialIndices([]); // Clear pivot highlight for this level
      return (i + 1);
  };

  const mergeSort = async (arr: number[], l: number, r: number) => {
    if (l >= r) return;
    if (stopRef.current) return;

    const m = l + Math.floor((r - l) / 2);
    await mergeSort(arr, l, m);
    await mergeSort(arr, m + 1, r);
    await merge(arr, l, m, r);
  };

  const merge = async (arr: number[], l: number, m: number, r: number) => {
      const n1 = m - l + 1;
      const n2 = r - m;
      const L = new Array(n1);
      const R = new Array(n2);

      for (let i = 0; i < n1; i++) L[i] = arr[l + i];
      for (let j = 0; j < n2; j++) R[j] = arr[m + 1 + j];

      let i = 0, j = 0, k = l;

      setSpecialIndices([]);

      while (i < n1 && j < n2) {
          if (stopRef.current) return;
          
          setCompareIndices([k]); // Highlight position being determined
          await delay(30);

          if (L[i] <= R[j]) {
              arr[k] = L[i];
              i++;
          } else {
              arr[k] = R[j];
              j++;
          }
          setArray([...arr]); 
          setSwapIndices([k]); // Show write
          await delay(20);
          k++;
      }

      while (i < n1) {
          if (stopRef.current) return;
          arr[k] = L[i];
          setArray([...arr]);
          setSwapIndices([k]);
          i++;
          k++;
          await delay(20);
      }
      while (j < n2) {
          if (stopRef.current) return;
          arr[k] = R[j];
          setArray([...arr]);
          setSwapIndices([k]);
          j++;
          k++;
          await delay(20);
      }
      setSwapIndices([]);
      setCompareIndices([]);
  };

  // --- Render ---

  const chartData = array.map((val, idx) => ({ index: idx, value: val }));

  const getBarColor = (index: number) => {
      if (sortedIndices.includes(index)) return '#22c55e'; // Green
      if (specialIndices.includes(index)) return '#a855f7'; // Purple (Pivot/Min)
      if (swapIndices.includes(index)) return '#ef4444'; // Red
      if (compareIndices.includes(index)) return '#eab308'; // Yellow
      return '#60a5fa'; // Blue
  };

  return (
    <div className="bg-slate-900 p-4 rounded-lg border border-slate-800 h-full flex flex-col">
      <div className="flex flex-wrap justify-between items-center mb-4 gap-3">
        <div className="flex items-center gap-3">
            <h3 className="text-lg font-semibold text-blue-400">Sorting Visualizer</h3>
            <select 
                value={algorithm}
                onChange={(e) => { resetArray(); setAlgorithm(e.target.value as Algorithm); }}
                disabled={sorting}
                className="bg-slate-800 text-slate-200 text-xs border border-slate-700 rounded px-2 py-1 outline-none focus:border-blue-500"
            >
                {Object.entries(ALGO_INFO).map(([key, val]) => (
                    <option key={key} value={key}>{val.name}</option>
                ))}
            </select>
        </div>
        
        <div className="flex gap-2">
           <button 
            onClick={handleSort}
            disabled={sorting}
            className="px-3 py-1 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed rounded text-white flex items-center gap-2 text-xs transition-colors"
          >
            <Play size={14} /> Start
          </button>
          <button 
            onClick={resetArray}
            className="px-3 py-1 bg-slate-700 hover:bg-slate-600 rounded text-white flex items-center gap-2 text-xs transition-colors"
          >
            <RefreshCw size={14} /> Reset
          </button>
        </div>
      </div>

      <div className="bg-slate-950 p-3 rounded border border-slate-800 mb-4 text-xs text-slate-400 flex gap-3">
          <div className="shrink-0 pt-0.5">
              <Info size={16} className="text-blue-400"/>
          </div>
          <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="font-bold text-slate-200 uppercase tracking-wider">{ALGO_INFO[algorithm].name}</span>
                <span className="font-mono text-yellow-500 bg-yellow-500/10 px-2 py-0.5 rounded text-[10px]">{ALGO_INFO[algorithm].complexity}</span>
              </div>
              <p className="leading-relaxed">{ALGO_INFO[algorithm].detail}</p>
          </div>
      </div>

      <div className="flex-grow w-full min-h-[200px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} barCategoryGap={1}>
            <Bar dataKey="value" animationDuration={0}>
              {chartData.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={getBarColor(index)} 
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      
       <div className="mt-3 flex justify-center gap-4 text-[10px] uppercase tracking-wider text-slate-500 font-mono">
          <span className="flex items-center gap-1"><div className="w-2 h-2 bg-blue-400 rounded-full"></div> Unsorted</span>
          <span className="flex items-center gap-1"><div className="w-2 h-2 bg-purple-500 rounded-full" style={{backgroundColor: '#a855f7'}}></div> Pivot/Min</span>
          <span className="flex items-center gap-1"><div className="w-2 h-2 bg-yellow-500 rounded-full" style={{backgroundColor: '#eab308'}}></div> Compare</span>
          <span className="flex items-center gap-1"><div className="w-2 h-2 bg-red-500 rounded-full"></div> Swap/Write</span>
          <span className="flex items-center gap-1"><div className="w-2 h-2 bg-green-500 rounded-full"></div> Sorted</span>
      </div>
    </div>
  );
};
