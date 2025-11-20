import { GoogleGenAI, Type } from "@google/genai";
import { ExplanationContent } from '../types';

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export const connectLiveSession = (callbacks: any, config: any) => {
  return ai.live.connect({
    model: 'gemini-2.5-flash-native-audio-preview-09-2025',
    callbacks,
    config
  });
};

export const fetchTopicExplanation = async (topicTitle: string): Promise<ExplanationContent> => {
  const prompt = `
    You are a world-renowned **Harvard Professor of Statistics and Machine Learning**. Your task is to provide a rigorous, senior-level technical masterclass on the topic: "${topicTitle}".

    **Persona Guidelines:**
    - **Authoritative & Precise:** Use precise terminology (e.g., "stochastic gradient descent" instead of "learning method").
    - **No Fluff:** Skip basic introductions. Assume the reader is a Senior Engineer.
    - **Mathematical Rigor:** Use strict LaTeX for all formulas. Show derivations where appropriate.

    **Requirements:**
    1. **Visuals & Formatting:** Use strict LaTeX for ALL math.
       - Inline: $x^2$
       - Block: $$ \\sum x_i $$
    2. **Code:** Provide a Python code snippet that is **production-grade**.
       - **MANDATORY:** Use ONLY these libraries: \`numpy\`, \`pandas\`, \`scikit-learn\`, \`scipy\`, \`matplotlib\`, \`seaborn\`.
       - **Visualization:** The code MUST generate a plot using \`matplotlib\` or \`seaborn\`.
       - Ensure the code is complete and runnable in a Pyodide environment.
    3. **Assessment:** Create 10 challenging multiple-choice questions that test deep understanding (e.g., edge cases, assumptions, complexity).

    **Output JSON Structure:**
    - **overview**: High-level technical summary (Markdown).
    - **mathematicalIntuition**: Deep dive into the math (LaTeX).
    - **useCases**: 3-5 specific industry applications.
    - **prosCons**: Professional trade-offs.
    - **codeSnippet**: Runnable Python script.
    - **quiz**: 10 Questions.
    - **deepDive**: Advanced theory, formulas, and papers.

    Generate the response in strict JSON format matching the schema below.
  `;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            overview: { type: Type.STRING },
            mathematicalIntuition: { type: Type.STRING },
            useCases: { type: Type.ARRAY, items: { type: Type.STRING } },
            prosCons: {
              type: Type.OBJECT,
              properties: {
                pros: { type: Type.ARRAY, items: { type: Type.STRING } },
                cons: { type: Type.ARRAY, items: { type: Type.STRING } },
              },
            },
            codeSnippet: { type: Type.STRING },
            quiz: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  question: { type: Type.STRING },
                  options: { type: Type.ARRAY, items: { type: Type.STRING } },
                  correctIndex: { type: Type.INTEGER },
                  explanation: { type: Type.STRING }
                }
              }
            },
            deepDive: {
              type: Type.OBJECT,
              properties: {
                advancedTheory: { type: Type.STRING },
                keyFormulas: { type: Type.ARRAY, items: { type: Type.STRING } },
                seminalPapers: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      title: { type: Type.STRING },
                      url: { type: Type.STRING }
                    }
                  }
                }
              }
            }
          },
          required: ["overview", "mathematicalIntuition", "useCases", "prosCons", "codeSnippet", "quiz", "deepDive"],
        },
      },
    });

    const text = response.text;
    if (!text) throw new Error("No content returned from Gemini");
    return JSON.parse(text) as ExplanationContent;
  } catch (error) {
    console.error("Gemini API Error:", error);
    return {
      overview: "Failed to load content from Gemini. Please try again or check your API key.",
      mathematicalIntuition: "N/A",
      useCases: [],
      prosCons: { pros: [], cons: [] },
      codeSnippet: "# Error loading code",
      quiz: [],
      deepDive: { advancedTheory: "N/A", keyFormulas: [], seminalPapers: [] }
    };
  }
};

export const fixPythonCode = async (code: string, errorMsg?: string): Promise<{ fixedCode: string; explanation: string }> => {
    const prompt = `
      You are an expert Python Data Science debugger.
      Fix the following Python code. 
      **Constraint:** Use ONLY standard libraries: numpy, pandas, scipy, sklearn, matplotlib, seaborn.
      
      **Code:**
      ${code}

      **Error/Context:**
      ${errorMsg || "General logic check"}

      Return the fixed code and a concise explanation of the fix.
    `;

    try {
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        fixedCode: { type: Type.STRING },
                        explanation: { type: Type.STRING }
                    },
                    required: ["fixedCode", "explanation"]
                }
            }
        });
        const text = response.text;
        if(!text) throw new Error("No fix returned");
        return JSON.parse(text);
    } catch (e) {
        console.error(e);
        return { fixedCode: code, explanation: "Failed to generate fix." };
    }
};

export const createChatSession = () => {
  return ai.chats.create({
    model: "gemini-3-pro-preview",
    config: {
      systemInstruction: `You are the "DataSci MasterClass" AI Tutor and Interview Coach.
      
      **Your Goal:**
      Help users master advanced Data Science, Machine Learning, and Python concepts. You must be capable of conducting rigorous technical mock interviews and helping users memorize concepts.

      **Interactive Capabilities:**
      You have the ability to render **Flashcards** and **Interactive Quizzes** directly in the chat.
      
      1. **Flashcards:** To show a flashcard, output a JSON block wrapped in \`:::flashcard\` and \`:::\`.
         Format:
         :::flashcard
         {
           "front": "The concept or question",
           "back": "The definition, formula, or answer"
         }
         :::

      2. **Quizzes:** To ask a multiple choice question, output a JSON block wrapped in \`:::quiz\` and \`:::\`.
         Format:
         :::quiz
         {
           "question": "Question text here?",
           "options": ["Option A", "Option B", "Option C", "Option D"],
           "correctIndex": 1,
           "explanation": "Brief explanation of why the answer is correct."
         }
         :::

      **Context Awareness:**
      The user is navigating a web application. You may receive system updates with the current topic and a list of available assessment questions. 
      - If the user asks to "quiz me" and you have questions in the context, pick one and present it using the \`:::quiz\` format.
      - If the user asks for "flashcards", generate them based on the current topic overview using the \`:::flashcard\` format.

      **Modes:**
      1. **Tutor Mode (Default):** Explain concepts with senior-level depth.
      2. **Interview Mode:** When asked to "interview me", shift persona. Ask ONE hard technical question at a time.

      **Formatting Rules:**
      - **Math:** ALWAYS use LaTeX format for equations (e.g., $E = mc^2$).
      - **Code:** Use standard markdown blocks.
      `,
    }
  });
};