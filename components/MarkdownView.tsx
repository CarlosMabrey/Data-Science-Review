import React, { useEffect, useRef } from 'react';

declare global {
  interface Window {
    marked: {
      parse: (text: string) => string;
    };
    katex: {
      renderToString: (tex: string, options?: any) => string;
    };
  }
}

interface MarkdownViewProps {
  text: string;
  className?: string;
  as?: 'div' | 'span';
  bare?: boolean;
}

export const MarkdownView: React.FC<MarkdownViewProps> = ({ text, className = '', as = 'div', bare = false }) => {
  const contentRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (!contentRef.current || !text) return;

    // 1. Tokenize Math
    // We use a very specific, non-conflicting random token prefix to avoid markdown parser issues.
    const tokenMap = new Map<string, { math: string; display: boolean }>();
    
    const protectMath = (str: string) => {
      // Capture $$...$$ (Display) and $...$ (Inline)
      // We use a closure to store the mapping
      return str.replace(/(\$\$[\s\S]*?\$\$)|(\$(?:\\.|[^$\\])*\$)/g, (match) => {
        const isDisplay = match.startsWith('$$');
        const cleanMath = isDisplay ? match.slice(2, -2) : match.slice(1, -1);
        
        // Generate a unique ID that is purely alphanumeric to pass through markdown parsers unharmed
        const id = `MATHXZY${Math.random().toString(36).substring(2, 11)}TOKEN`;
        tokenMap.set(id, { math: cleanMath, display: isDisplay });
        return id;
      });
    };

    const protectedText = protectMath(text);

    // 2. Parse Markdown -> HTML
    let html = "";
    if (window.marked) {
      // Configure marked to not escape single quotes if possible, but standard parse is usually fine with alphanumeric
      html = window.marked.parse(protectedText);
    } else {
      html = `<p>${protectedText}</p>`;
    }

    // 3. Restore Math (HTML Replacement)
    // We iterate over the rendered HTML string and replace tokens with KaTeX output
    tokenMap.forEach((value, key) => {
      let rendered = "";
      if (window.katex) {
        try {
          rendered = window.katex.renderToString(value.math, {
            displayMode: value.display,
            throwOnError: false,
            output: 'html',
            trust: true // Allow color commands etc
          });
        } catch (e) {
          console.error("KaTeX error:", e);
          rendered = `<span class="text-red-500 font-mono text-xs">${value.math}</span>`;
        }
      } else {
        rendered = `<code class="bg-slate-800 text-yellow-300 px-1 rounded">${value.math}</code>`;
      }
      // Replace all occurrences of the ID in the HTML
      html = html.split(key).join(rendered);
    });

    contentRef.current.innerHTML = html;
  }, [text]);

  if (!text) return null;

  const baseClasses = bare 
    ? "" 
    : "prose prose-invert prose-sm max-w-none prose-p:leading-relaxed prose-p:text-slate-300 prose-headings:text-blue-400 prose-headings:font-bold prose-strong:text-white prose-strong:font-bold prose-ul:list-disc prose-ul:pl-4 prose-ol:list-decimal prose-ol:pl-4 prose-code:text-yellow-300 prose-code:bg-slate-800/50 prose-code:px-1 prose-code:rounded prose-a:text-blue-400 hover:prose-a:text-blue-300";

  const combinedClasses = `${baseClasses} ${className}`;
  const Tag = as;

  return <Tag ref={contentRef as any} className={combinedClasses} />;
};