import React from 'react';
import { useTheme, Theme } from '../contexts/ThemeContext';
import { Palette, Moon, Zap, Droplets } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export const ThemeSwitcher: React.FC = () => {
  const { theme, setTheme } = useTheme();
  const [isOpen, setIsOpen] = React.useState(false);

  const themes: { id: Theme; label: string; icon: React.ReactNode; color: string }[] = [
    { id: 'default', label: 'Deep Space', icon: <Moon size={14} />, color: 'bg-blue-500' },
    { id: 'cyberpunk', label: 'Cyberpunk', icon: <Zap size={14} />, color: 'bg-fuchsia-500' },
    { id: 'ocean', label: 'Ocean', icon: <Droplets size={14} />, color: 'bg-teal-500' },
  ];

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="p-2 rounded-lg text-text-muted hover:text-accent hover:bg-surfaceHighlight transition-colors"
        title="Change Theme"
      >
        <Palette size={18} />
      </button>

      <AnimatePresence>
        {isOpen && (
          <>
            <div 
                className="fixed inset-0 z-40" 
                onClick={() => setIsOpen(false)}
            />
            <motion.div
              initial={{ opacity: 0, y: 10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 10, scale: 0.95 }}
              className="absolute right-0 bottom-12 w-48 bg-surface border border-border rounded-xl shadow-xl overflow-hidden z-50"
            >
              <div className="p-2 space-y-1">
                {themes.map((t) => (
                  <button
                    key={t.id}
                    onClick={() => { setTheme(t.id); setIsOpen(false); }}
                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                      theme === t.id 
                        ? 'bg-accent/20 text-accent' 
                        : 'text-text-muted hover:bg-surfaceHighlight hover:text-text-main'
                    }`}
                  >
                    <span className={`w-2 h-2 rounded-full ${t.color}`} />
                    <span className="flex-1 text-left">{t.label}</span>
                    {t.icon}
                  </button>
                ))}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
};