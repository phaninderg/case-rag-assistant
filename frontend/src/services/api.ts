import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Search for cases with optional parameters
export const searchCases = async (
  query: string, 
  k: number = 5, 
  minScore: number = 0.6,
  includeSolutions: boolean = true
) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/api/cases/search`, {
      query,
      k,
      min_score: minScore,
      include_solutions: includeSolutions
    });
    return response.data;
  } catch (error) {
    console.error('Search API error:', error);
    throw error;
  }
};

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
}

export interface ChatResponse {
  message: ChatMessage;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export const chatWithModel = async (request: ChatRequest): Promise<ChatResponse> => {
  try {
    // Directly send the request without wrapping in another object
    const response = await axios.post<ChatResponse>(`${API_BASE_URL}/api/chat`, {
      messages: request.messages,
      temperature: request.temperature ?? 0.7,
      max_tokens: request.max_tokens ?? 1000,
      stream: request.stream ?? false
    });
    return response.data;
  } catch (error) {
    console.error('Chat API error:', error);
    throw error;
  }
};
