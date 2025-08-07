import React, { useState, useEffect } from 'react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { motion } from 'framer-motion';
import { 
  Search, 
  FileText, 
  Network, 
  Users, 
  Brain,
  Database,
  Star,
  TrendingUp
} from 'lucide-react';
import { api, type PapersResponse, type SearchResponse, type StatsResponse } from './api/client';
import SearchTab from './components/SearchTab';
import PapersTab from './components/PapersTab';
import NetworkTab from './components/NetworkTab';
import ScholarsTab from './components/ScholarsTab';
import CustomTabs from './components/CustomTabs';
import './App.css';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

interface TabItem {
  id: string;
  label: string;
  icon: React.ReactNode;
}

const tabItems: TabItem[] = [
  { id: "search", label: "Search", icon: <Search /> },
  { id: "papers", label: "Papers", icon: <FileText /> },
  { id: "network", label: "Network", icon: <Network /> },
  { id: "scholars", label: "Scholars", icon: <Users /> },
];

function App() {
  const [activeTab, setActiveTab] = useState("search");
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);

  // Check API connection on mount
  useEffect(() => {
    const checkConnection = async () => {
      try {
        await api.health();
        setIsConnected(true);
        setConnectionError(null);
        
        // Load stats
        const statsData = await api.getStats();
        setStats(statsData);
      } catch (error) {
        setIsConnected(false);
        setConnectionError(error instanceof Error ? error.message : 'Connection failed');
      }
    };

    checkConnection();
  }, []);

  if (!isConnected) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="bg-white rounded-xl border border-slate-200 p-8 shadow-lg max-w-md w-full">
          <div className="text-center">
            <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Database className="w-8 h-8 text-red-600" />
            </div>
            <h2 className="text-xl font-semibold text-slate-900 mb-2">Connection Error</h2>
            <p className="text-slate-600 mb-4">{connectionError}</p>
            <div className="space-y-2 text-sm text-slate-500">
              <p>Please ensure:</p>
              <ul className="list-disc list-inside space-y-1">
                <li>FastAPI server is running on port 8000</li>
                <li>Ollama is running locally</li>
                <li>Vector store is properly configured</li>
              </ul>
            </div>
            <button 
              onClick={() => window.location.reload()}
              className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Retry Connection
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-slate-50">
        <div className="max-w-7xl mx-auto p-6">
          {/* Header */}
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8"
          >
            <div className="flex items-center gap-3 mb-2">
              <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-slate-900">Material Research RAG</h1>
                <p className="text-slate-600">Advanced research discovery and analysis platform</p>
              </div>
            </div>
            
            {/* Connection Status */}
            <div className="flex items-center gap-4 mt-4">
              <div className="flex items-center gap-2 text-sm">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-green-700">Connected to API</span>
              </div>
              {stats && (
                <div className="flex items-center gap-4 text-sm text-slate-600">
                  <span className="flex items-center gap-1">
                    <FileText className="w-4 h-4" />
                    {stats.total_papers} papers
                  </span>
                  <span className="flex items-center gap-1">
                    <Star className="w-4 h-4" />
                    {stats.total_figures} figures
                  </span>
                  <span className="flex items-center gap-1">
                    <Database className="w-4 h-4" />
                    {Object.keys(stats.folder_stats).length} categories
                  </span>
                </div>
              )}
            </div>
          </motion.div>

          {/* Tabs Navigation */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6"
          >
            <CustomTabs
              items={tabItems}
              value={activeTab}
              onValueChange={setActiveTab}
              className="bg-white border border-slate-200 shadow-sm"
            />
          </motion.div>

          {/* Tab Content */}
          <motion.div 
            key={activeTab}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3 }}
            className="space-y-6"
          >
            {activeTab === "search" && <SearchTab />}
            {activeTab === "papers" && <PapersTab />}
            {activeTab === "network" && <NetworkTab />}
            {activeTab === "scholars" && <ScholarsTab />}
          </motion.div>
        </div>
      </div>
    </QueryClientProvider>
  );
}

export default App;
