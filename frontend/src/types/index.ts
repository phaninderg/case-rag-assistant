export interface SearchResult {
  case_number: string;
  solution?: string;
  similarity_score: number;
  issue?: string;
  root_cause?: string;
  resolution?: string;
  steps_support?: string;
  case_task_number?: string;
  metadata?: {
    root_cause?: string;
    [key: string]: any;
  };
}

export interface SearchResponse {
  query: string;
  total_results: number;
  results?: SearchResult[];  // Present when include_solutions is false
  ai_summary?: string;  // Present when include_solutions is true
  metadata?: {
    model: string;
    include_solutions: boolean;
    min_score: number;
  };
}
