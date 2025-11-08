import React from 'react';
import { Box, Typography } from '@mui/material';

const LoadingSpinner: React.FC = () => {
  return (
    <Box 
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        backgroundColor: '#ffffff',
        color: '#000000'
      }}
    >
      <Box
        sx={{
          width: 60,
          height: 60,
          border: '4px solid #e0e0e0',
          borderTop: '4px solid #000000',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
          mb: 2,
          '@keyframes spin': {
            '0%': { transform: 'rotate(0deg)' },
            '100%': { transform: 'rotate(360deg)' },
          },
        }}
      />
      <Typography 
        variant="h6" 
        sx={{ 
          fontFamily: 'monospace',
          fontWeight: 600,
          letterSpacing: '0.1em'
        }}
      >
        LOADING...
      </Typography>
    </Box>
  );
};

export default LoadingSpinner;