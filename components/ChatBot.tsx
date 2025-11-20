import React, { useState, useEffect, useRef } from 'react';
import { createChatSession, connectLiveSession } from '../services/geminiService';
import { Chat, GenerateContentResponse, LiveServerMessage, Modality } from "@google/genai";
import { MessageSquare, X, Send, Loader2, User, Bot, Mic, StopCircle, RotateCw, CheckCircle, AlertCircle, AudioWaveform } from 'lucide-react';
import { Topic, ExplanationContent, QuizQuestion } from '../types';
import { MarkdownView } from './MarkdownView';
import { motion, AnimatePresence } from 'framer-motion';

interface ChatBotProps {
  currentTopic: Topic | null;
  contextContent: ExplanationContent | null;
}

interface Message {
  id: string;
  role: 'user' | 'model';
  text: string;
  timestamp: number;
}

// --- Audio Helpers ---

function base64ToBytes(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

function createPcmBlob(data: Float32Array): { data: string; mimeType: string } {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  
  let binary = '';
  const bytes = new Uint8Array(int16.buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  
  return {
    data: btoa(binary),
    mimeType: 'audio/pcm;rate=16000',
  };
}

// --- Helper Components for Interactive Elements ---

const FlashcardComponent: React.FC<{ front: string; back: string }> = ({ front, back }) => {
  const [flipped, setFlipped] = useState(false);
  
  // FIXED: Using 'backface-hidden' from updated index.html styles and proper 3D transform logic
  return (
    <div 
        className="my-4 w-full h-56 cursor-pointer group perspective-1000" 
        onClick={() => setFlipped(!flipped)}
    >
        <div 
            className={`relative w-full h-full transition-transform duration-700 transform-style-3d`}
            style={{ transform: flipped ? 'rotateY(180deg)' : 'rotateY(0deg)' }}
        >
            {/* Front */}
            <div className="absolute inset-0 backface-hidden bg-surface border border-border rounded-xl p-6 flex flex-col items-center justify-center text-center shadow-lg group-hover:border-accent transition-colors">
                <h4 className="text-xs text-accent uppercase tracking-wider mb-2 font-bold">Flashcard</h4>
                <div className="text-lg text-text-main font-medium overflow-y-auto max-h-[70%] custom-scrollbar">
                    <MarkdownView text={front} bare />
                </div>
                <div className="mt-auto text-[10px] text-text-muted flex items-center gap-1">
                    <RotateCw size={10} /> Click to flip
                </div>
            </div>
            {/* Back */}
            <div 
                className="absolute inset-0 backface-hidden bg-accent-dim border border-accent rounded-xl p-6 flex flex-col items-center justify-center text-center shadow-lg" 
                style={{ transform: 'rotateY(180deg)' }}
            >
                <h4 className="text-xs text-text-main uppercase tracking-wider mb-2 font-bold">Answer</h4>
                <div className="text-sm text-text-main overflow-y-auto max-h-[70%] custom-scrollbar">
                    <MarkdownView text={back} bare />
                </div>
                <div className="mt-auto text-[10px] text-text-muted flex items-center gap-1">
                    <RotateCw size={10} /> Click to flip back
                </div>
            </div>
        </div>
    </div>
  );
};

const InlineQuizComponent: React.FC<{ data: QuizQuestion }> = ({ data }) => {
    const [selected, setSelected] = useState<number | null>(null);
    const [showResult, setShowResult] = useState(false);

    const handleSelect = (idx: number) => {
        if(showResult) return;
        setSelected(idx);
        setShowResult(true);
    };

    return (
        <div className="my-4 bg-surface/50 rounded-xl border border-border p-4 overflow-hidden">
            <div className="flex items-center gap-2 mb-3">
                <div className="bg-accent/20 text-accent p-1 rounded"><Bot size={14}/></div>
                <span className="text-xs font-bold text-text-muted uppercase">Quick Check</span>
            </div>
            <div className="text-sm font-medium text-text-main mb-4">
                 <MarkdownView text={data.question} bare />
            </div>
            <div className="space-y-2">
                {data.options.map((opt, idx) => {
                    let style = "bg-surfaceHighlight border-border hover:bg-surfaceHighlight/80";
                    let icon = null;

                    if (showResult) {
                        if (idx === data.correctIndex) {
                            style = "bg-green-500/20 border-green-500 text-green-100";
                            icon = <CheckCircle size={14} className="text-green-400" />;
                        } else if (idx === selected) {
                            style = "bg-red-500/20 border-red-500 text-red-100";
                            icon = <AlertCircle size={14} className="text-red-400" />;
                        } else {
                            style = "bg-surfaceHighlight border-border opacity-50";
                        }
                    }

                    return (
                        <button
                            key={idx}
                            onClick={() => handleSelect(idx)}
                            disabled={showResult}
                            className={`w-full text-left p-3 rounded text-xs border transition-all flex justify-between items-center ${style}`}
                        >
                            <span className="flex-1"><MarkdownView bare text={opt} as="span"/></span>
                            {icon}
                        </button>
                    );
                })}
            </div>
            {showResult && (
                <motion.div 
                    initial={{ opacity: 0, height: 0 }} 
                    animate={{ opacity: 1, height: 'auto' }}
                    className="mt-3 p-3 bg-surfaceHighlight rounded border border-border text-xs text-text-muted"
                >
                    <span className="font-bold text-text-main">Explanation: </span>
                    <MarkdownView bare text={data.explanation} as="span"/>
                </motion.div>
            )}
        </div>
    );
};

// --- Main Component ---

export const ChatBot: React.FC<ChatBotProps> = ({ currentTopic, contextContent }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'init',
      role: 'model',
      text: "Hello! I'm your Data Science Interview Coach. Ask me to \"Interview you\" or \"Show me flashcards\" to get started.",
      timestamp: Date.now()
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  // Voice Mode State
  const [isVoiceMode, setIsVoiceMode] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
  const [isSpeaking, setIsSpeaking] = useState(false);

  const chatRef = useRef<Chat | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Audio Refs
  const audioContextsRef = useRef<{ input?: AudioContext, output?: AudioContext }>({});
  const sessionRef = useRef<Promise<any> | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const sourceNodesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  // Initialize chat session once
  useEffect(() => {
    chatRef.current = createChatSession();
    return () => {
      stopVoiceSession();
    };
  }, []);

  // Inject context when it changes (Text Chat)
  useEffect(() => {
    if (chatRef.current && currentTopic && contextContent) {
      const contextMessage = `
        [SYSTEM UPDATE]
        User is viewing topic: "${currentTopic.title}".
        Overview: ${contextContent.overview.substring(0, 200)}...
        
        Available Assessment Questions:
        ${JSON.stringify(contextContent.quiz)}
      `;
      
      chatRef.current.sendMessageStream({ message: contextMessage }).then(async (result) => {
           for await (const _ of result) {}
      }).catch(e => console.error("Failed to update context", e));
    }
  }, [currentTopic, contextContent]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if(!isVoiceMode) scrollToBottom();
  }, [messages, isOpen, isVoiceMode]);

  const startVoiceSession = async () => {
      setIsVoiceMode(true);
      setVoiceStatus('connecting');
      
      try {
          // Initialize Audio Contexts
          const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
          const inputCtx = new AudioContextClass({ sampleRate: 16000 });
          const outputCtx = new AudioContextClass({ sampleRate: 24000 });
          audioContextsRef.current = { input: inputCtx, output: outputCtx };

          // Get Microphone Stream
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          
          const config = {
              responseModalities: [Modality.AUDIO],
              speechConfig: {
                  voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } },
              },
              systemInstruction: `You are a conversational Data Science Tutor named "DataSci Bot". 
              The user is currently studying "${currentTopic?.title || 'General Data Science'}". 
              Keep your responses concise, encouraging, and spoken naturally.`,
          };

          const callbacks = {
              onopen: () => {
                  setVoiceStatus('connected');
                  console.log("Voice Session Connected");
                  
                  // Start Input Stream
                  const source = inputCtx.createMediaStreamSource(stream);
                  const processor = inputCtx.createScriptProcessor(4096, 1, 1);
                  
                  processor.onaudioprocess = (e) => {
                      const inputData = e.inputBuffer.getChannelData(0);
                      const pcmBlob = createPcmBlob(inputData);
                      
                      if (sessionRef.current) {
                          sessionRef.current.then((session: any) => {
                              session.sendRealtimeInput({ media: pcmBlob });
                          });
                      }
                  };
                  
                  source.connect(processor);
                  processor.connect(inputCtx.destination);
              },
              onmessage: async (msg: LiveServerMessage) => {
                  // Handle Audio Output
                  const data = msg.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
                  if (data) {
                      const outputCtx = audioContextsRef.current.output;
                      if (outputCtx) {
                          const bytes = base64ToBytes(data);
                          const audioBuffer = await decodeAudioData(bytes, outputCtx, 24000, 1);
                          
                          const source = outputCtx.createBufferSource();
                          source.buffer = audioBuffer;
                          source.connect(outputCtx.destination);
                          
                          source.addEventListener('ended', () => {
                              sourceNodesRef.current.delete(source);
                              if (sourceNodesRef.current.size === 0) setIsSpeaking(false);
                          });

                          // Schedule playback
                          const currentTime = outputCtx.currentTime;
                          const startTime = Math.max(nextStartTimeRef.current, currentTime);
                          source.start(startTime);
                          nextStartTimeRef.current = startTime + audioBuffer.duration;
                          
                          sourceNodesRef.current.add(source);
                          setIsSpeaking(true);
                      }
                  }
                  
                  // Handle Interruption
                  if (msg.serverContent?.interrupted) {
                      console.log("Interrupted");
                      sourceNodesRef.current.forEach(node => {
                          try { node.stop(); } catch (e) {}
                      });
                      sourceNodesRef.current.clear();
                      nextStartTimeRef.current = 0;
                      setIsSpeaking(false);
                  }
                  
                  // Turn Complete
                  if (msg.serverContent?.turnComplete) {
                      // Optional: logic when turn finishes
                  }
              },
              onclose: () => {
                  setVoiceStatus('disconnected');
                  setIsVoiceMode(false);
              },
              onerror: (err: any) => {
                  console.error("Live API Error", err);
                  setVoiceStatus('error');
              }
          };

          sessionRef.current = connectLiveSession(callbacks, config);

      } catch (e) {
          console.error("Failed to start voice session", e);
          setVoiceStatus('error');
          setIsVoiceMode(false);
      }
  };

  const stopVoiceSession = () => {
      if (sessionRef.current) {
          sessionRef.current.then((s: any) => s.close());
          sessionRef.current = null;
      }
      
      audioContextsRef.current.input?.close();
      audioContextsRef.current.output?.close();
      sourceNodesRef.current.forEach(node => {
          try { node.stop(); } catch(e) {}
      });
      sourceNodesRef.current.clear();
      
      setIsVoiceMode(false);
      setVoiceStatus('disconnected');
      setIsSpeaking(false);
  };

  const handleSend = async () => {
    if (!input.trim() || !chatRef.current) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      text: input,
      timestamp: Date.now()
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const result = await chatRef.current.sendMessageStream({ message: userMsg.text });
      
      const botMsgId = (Date.now() + 1).toString();
      setMessages(prev => [...prev, {
        id: botMsgId,
        role: 'model',
        text: '',
        timestamp: Date.now()
      }]);

      let accumulatedText = '';
      for await (const chunk of result) {
        const chunkText = (chunk as GenerateContentResponse).text || '';
        accumulatedText += chunkText;
        setMessages(prev => prev.map(msg => msg.id === botMsgId ? { ...msg, text: accumulatedText } : msg));
      }
    } catch (error) {
      console.error("Chat Error", error);
      setMessages(prev => [...prev, { id: Date.now().toString(), role: 'model', text: "I'm having trouble connecting.", timestamp: Date.now() }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const renderMessageContent = (text: string) => {
      const parts = text.split(/(:::(?:flashcard|quiz)\s*\{[\s\S]*?\}\s*:::)/g);
      return parts.map((part, i) => {
          if (part.startsWith(':::flashcard')) {
              try {
                  const jsonStr = part.replace(/^:::flashcard\s*/, '').replace(/\s*:::$/, '');
                  const data = JSON.parse(jsonStr);
                  return <FlashcardComponent key={i} front={data.front} back={data.back} />;
              } catch (e) { return null; }
          } else if (part.startsWith(':::quiz')) {
              try {
                  const jsonStr = part.replace(/^:::quiz\s*/, '').replace(/\s*:::$/, '');
                  const data = JSON.parse(jsonStr);
                  return <InlineQuizComponent key={i} data={data} />;
              } catch (e) { return null; }
          } else {
              if (!part.trim()) return null;
              return <div key={i}><MarkdownView text={part} bare /></div>;
          }
      });
  };

  return (
    <>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            className="fixed bottom-20 right-6 w-[400px] md:w-[450px] h-[600px] bg-surface border border-border rounded-2xl shadow-2xl flex flex-col overflow-hidden z-50"
          >
            {/* Header */}
            <div className="bg-surface/80 backdrop-blur-sm p-4 border-b border-border flex justify-between items-center relative z-10">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-accent to-purple-500 flex items-center justify-center">
                    <Bot size={18} className="text-text-inverted" />
                </div>
                <div>
                  <h3 className="font-semibold text-text-main">{isVoiceMode ? 'Live Voice' : 'AI Coach'}</h3>
                  <p className="text-[10px] text-text-muted flex items-center gap-1">
                    {isVoiceMode ? (isSpeaking ? 'Speaking...' : 'Listening...') : (currentTopic ? `Context: ${currentTopic.title}` : 'General')}
                  </p>
                </div>
              </div>
              <div className="flex gap-2">
                {!isVoiceMode && (
                    <button onClick={startVoiceSession} className="p-2 text-text-muted hover:text-text-main hover:bg-surfaceHighlight rounded-full transition-colors" title="Start Voice Mode">
                        <Mic size={18} />
                    </button>
                )}
                <button onClick={() => setIsOpen(false)} className="text-text-muted hover:text-text-main transition-colors">
                    <X size={20} />
                </button>
              </div>
            </div>

            {/* Content Area */}
            <div className="flex-grow overflow-y-auto bg-background/50 relative custom-scrollbar">
                {isVoiceMode ? (
                    <div className="h-full flex flex-col items-center justify-center p-6 space-y-8">
                        <div className="relative">
                             <div className={`w-32 h-32 rounded-full flex items-center justify-center transition-all duration-300 ${isSpeaking ? 'bg-accent shadow-[0_0_60px_rgba(59,130,246,0.6)] scale-110' : 'bg-surface border border-border'}`}>
                                {isSpeaking ? <AudioWaveform size={48} className="text-white animate-pulse"/> : <Mic size={48} className="text-text-muted"/>}
                             </div>
                             {/* Ripple effects */}
                             {isSpeaking && (
                                 <>
                                    <div className="absolute inset-0 rounded-full border-2 border-accent/30 animate-[ping_2s_linear_infinite]"></div>
                                    <div className="absolute inset-0 rounded-full border-2 border-accent/20 animate-[ping_2s_linear_infinite_0.5s]"></div>
                                 </>
                             )}
                        </div>

                        <div className="text-center space-y-2">
                            <h3 className="text-2xl font-bold text-text-main">
                                {voiceStatus === 'connecting' ? 'Connecting...' : isSpeaking ? 'AI Speaking' : 'Listening...'}
                            </h3>
                            <p className="text-text-muted text-sm max-w-[250px] mx-auto">
                                {voiceStatus === 'error' ? 'Connection Failed' : 'Speak naturally to discuss Data Science concepts.'}
                            </p>
                        </div>

                        <button 
                            onClick={stopVoiceSession}
                            className="px-6 py-3 bg-red-600 hover:bg-red-500 text-white rounded-full font-medium flex items-center gap-2 shadow-lg transition-transform hover:scale-105"
                        >
                            <StopCircle size={20}/> End Voice Mode
                        </button>
                    </div>
                ) : (
                    <div className="p-4 space-y-4">
                        {messages.map((msg) => (
                            <div key={msg.id} className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                <div className={`max-w-[90%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ${msg.role === 'user' ? 'bg-accent text-text-inverted rounded-br-none font-medium' : 'bg-surface text-text-main rounded-bl-none border border-border'}`}>
                                    {msg.role === 'user' ? msg.text : renderMessageContent(msg.text)}
                                </div>
                            </div>
                        ))}
                        {isLoading && (
                            <div className="flex justify-start">
                                <div className="bg-surface rounded-2xl rounded-bl-none px-4 py-3 border border-border">
                                    <Loader2 size={16} className="animate-spin text-accent" />
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>
                )}
            </div>

            {/* Input Area (Only visible in Text Mode) */}
            {!isVoiceMode && (
                <div className="p-4 bg-surface border-t border-border">
                    {messages.length < 4 && (
                        <div className="mb-3 flex gap-2 overflow-x-auto no-scrollbar">
                            <button onClick={() => { setInput(`Generate 3 flashcards about ${currentTopic?.title}`); handleSend(); }} className="whitespace-nowrap px-3 py-1 bg-surfaceHighlight hover:bg-surfaceHighlight/80 border border-border rounded-full text-xs text-text-muted transition flex items-center gap-1">
                                <RotateCw size={10} /> Flashcards
                            </button>
                            <button onClick={() => { setInput(`Quiz me on ${currentTopic?.title}`); handleSend(); }} className="whitespace-nowrap px-3 py-1 bg-surfaceHighlight hover:bg-surfaceHighlight/80 border border-border rounded-full text-xs text-text-muted transition flex items-center gap-1">
                                <CheckCircle size={10} /> Quiz Me
                            </button>
                        </div>
                    )}
                    <div className="relative">
                        <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Type a message..."
                        className="w-full bg-background border border-border rounded-xl pl-4 pr-12 py-3 text-sm text-text-main focus:outline-none focus:border-accent resize-none h-[50px] custom-scrollbar"
                        />
                        <button
                        onClick={handleSend}
                        disabled={!input.trim() || isLoading}
                        className="absolute right-2 top-2 p-2 bg-accent hover:bg-accent-hover disabled:bg-surfaceHighlight disabled:text-text-muted text-text-inverted rounded-lg transition-colors"
                        >
                        {isLoading ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
                        </button>
                    </div>
                </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Toggle Button */}
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setIsOpen(!isOpen)}
        className={`fixed bottom-6 right-6 w-14 h-14 rounded-full shadow-lg flex items-center justify-center z-50 transition-colors ${
            isOpen ? 'bg-surfaceHighlight text-text-muted' : 'bg-accent text-text-inverted hover:bg-accent-hover'
        }`}
      >
        {isOpen ? <X size={24} /> : <MessageSquare size={24} />}
      </motion.button>
    </>
  );
};