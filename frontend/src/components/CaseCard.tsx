import React from 'react';
import { 
  Paper, 
  Typography, 
  Box, 
  Divider,
  Collapse,
  IconButton
} from '@mui/material';
import { ExpandMore as ExpandMoreIcon } from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';

interface CaseCardProps {
  result: {
    case_number: string;
    solution?: string;
    similarity_score: number;
    issue?: string;
    root_cause?: string;
    resolution?: string;
    steps_support?: string;
    is_ai_generated?: boolean;
    metadata?: {
      parent_case?: string;
      [key: string]: any;
    };
  };
  includeSolutions?: boolean;
}

export const CaseCard: React.FC<CaseCardProps> = ({ result, includeSolutions = false }) => {
  const [expanded, setExpanded] = React.useState(false);

  const toggleExpand = () => {
    setExpanded(!expanded);
  };

  // Remove parent case number from metadata if it exists
  const { parent_case, ...filteredMetadata } = result.metadata || {};
  const displayMetadata = Object.entries(filteredMetadata || {}).filter(
    ([key]) => key !== 'parent_case' && key !== 'case_task_number'
  );

  return (
    <Paper elevation={2} sx={{ p: 3, mb: 3, position: 'relative' }}>
      <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
        <Box>
          <Typography variant="h6" color="primary">
            Case #{result.case_number}
          </Typography>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Match: {(result.similarity_score * 100).toFixed(1)}%
          </Typography>
          
          {/* Show AI-generated indicator if applicable */}
          {result.solution && (
            <Typography variant="caption" color="text.secondary" display="block">
              {result.is_ai_generated ? 'AI-Generated Solution' : 'Solution from case'}
            </Typography>
          )}
        </Box>
        
        <IconButton 
          onClick={toggleExpand}
          aria-expanded={expanded}
          aria-label={expanded ? 'Show less' : 'Show more'}
          size="small"
          sx={{ mt: -1 }}
        >
          <ExpandMoreIcon 
            sx={{ 
              transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.3s'
            }} 
          />
        </IconButton>
      </Box>

      {includeSolutions && result.solution && (
        <Box mb={2}>
          <Typography paragraph sx={{ whiteSpace: 'pre-line', wordBreak: 'break-word' }}>
            <ReactMarkdown>{result.solution}</ReactMarkdown>
          </Typography>
        </Box>
      )}

      <Collapse in={true} timeout="auto" unmountOnExit>
        {result.issue && (
          <Box mb={2}>
            <Typography variant="subtitle1" gutterBottom>
              <strong>Issue:</strong>
            </Typography>
            <Typography paragraph sx={{ whiteSpace: 'pre-line', wordBreak: 'break-word' }}>
              <ReactMarkdown>{result.issue}</ReactMarkdown>
            </Typography>
          </Box>
        )}

        {result.root_cause && (
          <Box mb={2}>
            <Typography variant="subtitle1" gutterBottom>
              <strong>Root Cause:</strong>
            </Typography>
            <Typography paragraph sx={{ whiteSpace: 'pre-line', wordBreak: 'break-word' }}>
              <ReactMarkdown>{result.root_cause}</ReactMarkdown>
            </Typography>
          </Box>
        )}

        {result.resolution && (
          <Box mb={2}>
            <Typography variant="subtitle1" gutterBottom>
              <strong>Resolution:</strong>
            </Typography>
            <Typography paragraph sx={{ whiteSpace: 'pre-line', wordBreak: 'break-word' }}>
              <ReactMarkdown>{result.resolution}</ReactMarkdown>
            </Typography>
          </Box>
        )}

        {result.steps_support && (
          <Box mb={2}>
            <Typography variant="subtitle1" gutterBottom>
              <strong>Support Steps:</strong>
            </Typography>
            <Typography paragraph sx={{ whiteSpace: 'pre-line', wordBreak: 'break-word' }}>
              <ReactMarkdown>{result.steps_support}</ReactMarkdown>
            </Typography>
          </Box>
        )}

        {displayMetadata.length > 0 && (
          <Box mt={3}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Additional Information:
            </Typography>
            <Box component="dl" sx={{ m: 0 }}>
              {displayMetadata.map(([key, value]) => (
                <Box key={key} display="flex" mb={1}>
                  <Box component="dt" sx={{ minWidth: 120, fontWeight: 'medium' }}>
                    {key.replace(/_/g, ' ')}:
                  </Box>
                  <Box component="dd" sx={{ flex: 1, ml: 2 }}>
                    {typeof value === 'string' ? value : JSON.stringify(value)}
                  </Box>
                </Box>
              ))}
            </Box>
          </Box>
        )}
      </Collapse>
      
      <Divider sx={{ my: 2 }} />
      <Box display="flex" justifyContent="flex-end">
        <Typography 
          variant="caption" 
          color="text.secondary"
          onClick={toggleExpand}
          sx={{ 
            cursor: 'pointer',
            '&:hover': { textDecoration: 'underline' } 
          }}
        >
          {expanded ? 'Show less' : 'Show more details'}
        </Typography>
      </Box>
    </Paper>
  );
};

export default CaseCard;
