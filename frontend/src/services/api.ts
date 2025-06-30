import axios, { AxiosError, AxiosRequestConfig, AxiosResponse } from 'axios';

// Get API URL and timeout from environment variables with fallbacks
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API_TIMEOUT = Number(process.env.REACT_APP_API_TIMEOUT || 30000);

// Create axios instance with default configuration
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for logging and auth tokens if needed
api.interceptors.request.use(
  (config) => {
    // Add any auth tokens here if needed in the future
    // if (localStorage.getItem('token')) {
    //   config.headers.Authorization = `Bearer ${localStorage.getItem('token')}`;
    // }
    return config;
  },
  (error) => Promise.reject(error)
);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean };
    
    // Handle specific error cases
    if (error.response) {
      // Server responded with an error status code
      const { status } = error.response;
      
      if (status === 401) {
        // Handle unauthorized errors
        console.error('Authentication error');
      } else if (status === 429) {
        // Handle rate limiting
        console.warn('Rate limit exceeded, retrying after delay');
        // Wait 2 seconds and retry once
        if (!originalRequest._retry) {
          originalRequest._retry = true;
          await new Promise(resolve => setTimeout(resolve, 2000));
          return api(originalRequest);
        }
      }
    } else if (error.request) {
      // Request was made but no response received (network error)
      console.error('Network error, no response received');
    }
    
    return Promise.reject(error);
  }
);

// Simple in-memory cache for GET requests
const cache: Record<string, { data: any; timestamp: number }> = {};
const CACHE_DURATION = 60000; // 1 minute cache

// Generic request function with caching for GET requests
const apiRequest = async <T>(config: AxiosRequestConfig): Promise<T> => {
  // Only cache GET requests
  if (config.method === 'get' && config.url) {
    const cacheKey = `${config.url}${config.params ? JSON.stringify(config.params) : ''}`;
    const cachedItem = cache[cacheKey];
    
    // Return cached data if it exists and is not expired
    if (cachedItem && Date.now() - cachedItem.timestamp < CACHE_DURATION) {
      return cachedItem.data;
    }
    
    try {
      const response = await api(config);
      // Cache the response
      cache[cacheKey] = {
        data: response.data,
        timestamp: Date.now(),
      };
      return response.data;
    } catch (error) {
      throw error;
    }
  }
  
  // For non-GET requests or uncacheable requests
  const response = await api(config);
  return response.data;
};

// Search for cases with optional parameters
export const searchCases = async (
  query: string, 
  k: number = 5, 
  minScore: number = 0.6,
  includeSolutions: boolean = true
) => {
  try {
    return await apiRequest({
      method: 'post',
      url: '/api/cases/search',
      data: {
        query,
        k,
        min_score: minScore,
        include_solutions: includeSolutions
      }
    });
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

export const chatWithModel = async (chatRequest: ChatRequest): Promise<ChatResponse> => {
  try {
    return await apiRequest<ChatResponse>({
      method: 'post',
      url: '/api/chat',
      data: {
        messages: chatRequest.messages,
        temperature: chatRequest.temperature ?? 0.7,
        max_tokens: chatRequest.max_tokens ?? 1000,
        stream: chatRequest.stream ?? false
      }
    });
  } catch (error) {
    console.error('Chat API error:', error);
    throw error;
  }
};
