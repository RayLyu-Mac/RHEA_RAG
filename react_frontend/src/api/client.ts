import axios from 'axios';

// API base URL - change this to match your FastAPI server
const API_BASE_URL = 'http://localhost:8000';

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout for LLM operations
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types for API responses
export interface PaperInfo {
  file_name: string;
  figure_count: number;
  has_figures: boolean;
  folder: string;
}

export interface SearchResult {
  file_name: string;
  section: string;
  content: string;
  document_type: string;
  figure_count: number;
}

export interface SearchRequest {
  question: string;
  selected_papers?: string[];
  search_type?: string;
  k?: number;
  llm_model?: string;
}

export interface SearchResponse {
  answer: string;
  results: SearchResult[];
  total_results: number;
}

export interface PapersResponse {
  papers: Record<string, PaperInfo[]>;
  total_papers: number;
}

export interface CorrelationInfo {
  source: string;
  target: string;
  relationship_type: string;
  description: string;
  strength: number;
  evidence?: string;
}

export interface CorrelationsResponse {
  correlations: CorrelationInfo[];
  topics: string[];
}

export interface NetworkNode {
  id: string;
  connections: number;
}

export interface NetworkEdge {
  source: string;
  target: string;
  data: any;
}

export interface NetworkResponse {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  total_nodes: number;
  total_edges: number;
}

export interface StatsResponse {
  total_papers: number;
  total_figures: number;
  papers_with_figures: number;
  folder_stats: Record<string, { count: number; figures: number }>;
  vectorstore_loaded: boolean;
  llm_loaded: boolean;
}

// API functions
export const api = {
  // Health check
  health: async () => {
    const response = await apiClient.get('/health');
    return response.data;
  },

  // Get all papers
  getPapers: async (): Promise<PapersResponse> => {
    const response = await apiClient.get('/papers');
    return response.data;
  },

  // Search papers
  searchPapers: async (request: SearchRequest): Promise<SearchResponse> => {
    const response = await apiClient.post('/search', request);
    return response.data;
  },

  // Get available models
  getModels: async () => {
    const response = await apiClient.get('/models');
    return response.data;
  },

  // Get correlations
  getCorrelations: async (topic?: string): Promise<CorrelationsResponse> => {
    const params = topic ? { topic } : {};
    const response = await apiClient.get('/correlations', { params });
    return response.data;
  },

  // Get network data
  getNetworkData: async (topic?: string): Promise<NetworkResponse> => {
    const params = topic ? { topic } : {};
    const response = await apiClient.get('/network', { params });
    return response.data;
  },

  // Get system stats
  getStats: async (): Promise<StatsResponse> => {
    const response = await apiClient.get('/stats');
    return response.data;
  },
};

// Error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    
    if (error.response) {
      // Server responded with error status
      const message = error.response.data?.detail || error.response.data?.message || 'API Error';
      throw new Error(message);
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('No response from server. Please check if the FastAPI server is running.');
    } else {
      // Something else happened
      throw new Error('Network error occurred.');
    }
  }
);

export default apiClient;
