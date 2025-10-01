import React from 'react';
import { ThemeProvider, CssBaseline, Box, Typography, Button } from '@mui/material';
import theme from './theme';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box 
        sx={{ 
          minHeight: '100vh', 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'center', 
          justifyContent: 'center',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white'
        }}
      >
        <Typography variant="h2" gutterBottom>
          🔐 QFLARE
        </Typography>
        <Typography variant="h5" gutterBottom>
          Quantum-Safe Federated Learning
        </Typography>
        <Button 
          variant="contained" 
          size="large" 
          sx={{ mt: 3 }}
          onClick={() => alert('QFLARE is loading!')}
        >
          Get Started
        </Button>
      </Box>
    </ThemeProvider>
  );
}

export default App;