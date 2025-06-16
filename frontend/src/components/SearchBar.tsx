import React from 'react';
import { 
  TextField, 
  Button, 
  Box, 
  CircularProgress, 
  FormControlLabel, 
  Checkbox, 
  InputLabel, 
  Slider,
  Grid
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';

interface SearchBarProps {
  query: string;
  loading: boolean;
  includeAI: boolean;
  k: number;
  minScore: number;
  onSearch: () => void;
  onQueryChange: (value: string) => void;
  onIncludeAIChange: (checked: boolean) => void;
  onKChange: (value: number) => void;
  onMinScoreChange: (value: number) => void;
}

export const SearchBar: React.FC<SearchBarProps> = ({ 
  query, 
  loading, 
  includeAI,
  k,
  minScore,
  onSearch, 
  onQueryChange,
  onIncludeAIChange,
  onKChange,
  onMinScoreChange
}) => {
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && query.trim()) {
      onSearch();
    }
  };

  return (
    <Box display="flex" flexDirection="column" gap={2} mb={4}>
      <Box display="flex" gap={2} alignItems="center">
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Search for cases (e.g., 'login issue after update')"
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={loading}
          size="small"
        />
        <Button
          variant="contained"
          color="primary"
          onClick={onSearch}
          disabled={loading || !query.trim()}
          startIcon={loading ? <CircularProgress size={20} /> : <SearchIcon />}
          sx={{ whiteSpace: 'nowrap' }}
        >
          {loading ? 'Searching...' : 'Search'}
        </Button>
      </Box>
      
      <Box display="grid" gridTemplateColumns={{ xs: '1fr', sm: '1fr 1fr' }} gap={2}>
        <Box>
          <InputLabel>Number of results (k): {k}</InputLabel>
          <Slider
            value={k}
            onChange={(_, value) => onKChange(value as number)}
            min={1}
            max={20}
            step={1}
            valueLabelDisplay="auto"
            disabled={loading}
          />
        </Box>
        <Box>
          <InputLabel>Minimum similarity score: {minScore.toFixed(2)}</InputLabel>
          <Slider
            value={minScore}
            onChange={(_, value) => onMinScoreChange(value as number)}
            min={0}
            max={1}
            step={0.05}
            valueLabelDisplay="auto"
            disabled={loading}
          />
        </Box>
      </Box>
      
      <FormControlLabel
        control={
          <Checkbox
            checked={includeAI}
            onChange={(e) => onIncludeAIChange(e.target.checked)}
            color="primary"
            size="small"
          />
        }
        label="Enable AI-powered solutions"
        disabled={loading}
      />
    </Box>
  );
};

export default SearchBar;
