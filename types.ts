export enum TopicCategory {
  PYTHON = 'Python & Engineering',
  ML = 'Machine Learning',
  STATS = 'Statistics & Math',
  GENAI = 'Generative AI & NLP',
  DE = 'Data Engineering & Cloud',
  DATA = 'Data Science Core',
  MLOPS = 'MLOps & Production',
}

export enum VisualizationType {
  NONE = 'NONE',
  LINEAR_REGRESSION = 'LINEAR_REGRESSION',
  LOGISTIC_REGRESSION = 'LOGISTIC_REGRESSION',
  K_MEANS = 'K_MEANS',
  NORMAL_DISTRIBUTION = 'NORMAL_DISTRIBUTION',
  SORTING = 'SORTING',
  TREE_ENSEMBLES = 'TREE_ENSEMBLES',
  GRADIENT_DESCENT = 'GRADIENT_DESCENT',
  BACKPROPAGATION = 'BACKPROPAGATION',
  REGULARIZATION = 'REGULARIZATION',
  PCA = 'PCA',
  SVM = 'SVM',
  BAYESIAN = 'BAYESIAN',
  BIAS_VARIANCE = 'BIAS_VARIANCE',
  PROB_DISTRIBUTIONS = 'PROB_DISTRIBUTIONS',
}

export interface Topic {
  id: string;
  title: string;
  category: TopicCategory;
  shortDesc: string;
  visualizationType: VisualizationType;
}

export interface QuizQuestion {
  question: string;
  options: string[];
  correctIndex: number;
  explanation: string;
}

export interface DeepDiveContent {
  advancedTheory: string;
  keyFormulas: string[];
  seminalPapers: { title: string; url?: string }[];
}

export interface ExplanationContent {
  overview: string;
  mathematicalIntuition: string;
  useCases: string[];
  prosCons: { pros: string[]; cons: string[] };
  codeSnippet: string;
  quiz: QuizQuestion[];
  deepDive: DeepDiveContent;
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
}