import React, { useState, useCallback, useMemo, lazy, Suspense } from 'react';
import { 
  Container, 
  Typography, 
  Paper, 
  CssBaseline,
  AppBar,
  Toolbar,
  ThemeProvider,
  createTheme,
  Alert,
  Tabs,
  Tab,
  Box,
  CircularProgress,
  IconButton
} from '@mui/material';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import SearchBar from './components/SearchBar';
import CaseCard from './components/CaseCard';
import ErrorBoundary from './components/ErrorBoundary';
import { searchCases } from './services/api';
import { SearchResult } from './types';

// Use lazy loading for the Chat component which may not be used immediately
const Chat = lazy(() => import('./components/Chat'));

// Import theme hook
type ThemeMode = 'light' | 'dark';
interface ThemeHook {
  mode: ThemeMode;
  toggleTheme: () => void;
  isDark: boolean;
}

// Mock implementation for useTheme if the module is not found
const useTheme = (): ThemeHook => {
  const [mode, setMode] = useState<ThemeMode>('light');
  
  const toggleTheme = useCallback(() => {
    setMode(prevMode => (prevMode === 'light' ? 'dark' : 'light'));
  }, []);
  
  return {
    mode,
    toggleTheme,
    isDark: mode === 'dark'
  };
};

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `simple-tab-${index}`,
    'aria-controls': `simple-tabpanel-${index}`,
  };
}

function App() {
  // Use the custom theme hook
  const { mode, toggleTheme } = useTheme();
  
  // Create theme based on current mode
  const theme = useMemo(() => 
    createTheme({
      palette: {
        mode: mode as 'light' | 'dark',
        primary: {
          main: '#1976d2',
        },
        secondary: {
          main: '#dc004e',
        },
        background: {
          default: mode === 'light' ? '#f5f5f5' : '#121212',
          paper: mode === 'light' ? '#ffffff' : '#1e1e1e',
        },
      },
    }),
  [mode]);

  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [aiSummary, setAiSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [tabValue, setTabValue] = useState(0);
  const [includeAI, setIncludeAI] = useState(true);
  const [k, setK] = useState(10);
  const [minScore, setMinScore] = useState(0.6);

  // Memoize tab change handler to prevent unnecessary re-renders
  const handleTabChange = useCallback((event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  }, []);

  // Memoize search handler for better performance
  const handleSearch = useCallback(async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    setError('');
    setResults([]);
    setAiSummary('');
    
    try {
      const response = await searchCases(query, k, minScore, includeAI);
      
      // Type assertion for the response
      const data = response as { 
        ai_summary?: string; 
        results?: SearchResult[] 
      };
      
      if (includeAI) {
        // When includeAI is true, we only get ai_summary
        setAiSummary(data.ai_summary || 'No summary available.');
      } else {
        // When includeAI is false, we get full results
        setResults(data.results || []);
      }
      
      setTabValue(0); // Switch to search results tab
    } catch (err) {
      setError('Failed to fetch results. Make sure the backend is running.');
      console.error('Search error:', err);
    } finally {
      setLoading(false);
    }
  }, [query, k, minScore, includeAI]);

  return (
    <ErrorBoundary>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Case RAG Assistant
            </Typography>
            <IconButton onClick={toggleTheme} color="inherit" aria-label="toggle theme">
              {mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
            </IconButton>
          </Toolbar>
        </AppBar>
      
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Paper elevation={3} sx={{ mb: 4 }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs 
              value={tabValue} 
              onChange={handleTabChange} 
              aria-label="navigation tabs"
              variant="fullWidth"
            >
              <Tab label="Search Cases" {...a11yProps(0)} />
              <Tab label="Chat with AI" {...a11yProps(1)} />
            </Tabs>
          </Box>

          <TabPanel value={tabValue} index={0}>
            <Box sx={{ p: 3 }}>
              <SearchBar
                query={query}
                loading={loading}
                includeAI={includeAI}
                k={k}
                minScore={minScore}
                onSearch={handleSearch}
                onQueryChange={setQuery}
                onIncludeAIChange={setIncludeAI}
                onKChange={setK}
                onMinScoreChange={setMinScore}
              />
              
              {error && (
                <Alert severity="error" sx={{ mt: 2, mb: 2 }}>
                  {error}
                </Alert>
              )}

              {includeAI ? (
                // Show AI summary when includeAI is true
                aiSummary && (
                  <Box sx={{ 
                    mb: 4, 
                    p: 2, 
                    bgcolor: 'background.paper', 
                    borderRadius: 1,
                    borderLeft: '4px solid',
                    borderColor: 'primary.main'
                  }}>
                    <Typography variant="h6" gutterBottom>AI Analysis</Typography>
                    <Box 
                      component="div" 
                      sx={{ 
                        whiteSpace: 'pre-wrap',
                        '& ol, & ul': { pl: 2, mt: 1 },
                        '& li': { mb: 1 }
                      }}
                      dangerouslySetInnerHTML={{ 
                        __html: aiSummary
                          .split('\n4. Case References')[0]  // Remove everything after '4. Case References'
                          .replace(/\n/g, '<br />')
                          .replace(/(\d+\.\s+)([^<]+)/g, '<strong>$1</strong>$2')
                      }} 
                    />
                  </Box>
                )
              ) : (
                // Show regular search results when includeAI is false
                <Box sx={{ mt: 4 }}>
                  <Typography variant="h5" gutterBottom>
                    Search Results
                  </Typography>
                  {results.map((result, index) => (
                    <CaseCard 
                      key={`${result.case_number}-${index}`} 
                      result={result} 
                      includeSolutions={false}
                    />
                  ))}
                </Box>
              )}
            </Box>
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            <Box sx={{ p: 3 }}>
              <Suspense fallback={<Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}><CircularProgress /></Box>}>
                <Chat />
              </Suspense>
            </Box>
          </TabPanel>
        </Paper>
      </Container>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
