import React from 'react';
import { ThemeProvider, CssBaseline, Box, Typography, Button } from '@mui/material';
import theme from './theme';

function App() {
  const handleGetStarted = () => {
    // Test API connection
    fetch('http://localhost:8080/api/devices')
      .then(response => response.json())
      .then(data => {
        console.log('API Response:', data);
        alert('Backend connected! Check console for details.');
      })
      .catch(error => {
        console.error('API Error:', error);
        alert('Backend connection failed: ' + error.message);
      });
  };

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
          color: 'white',
          textAlign: 'center',
          p: 3
        }}
      >
        <Typography variant="h2" gutterBottom>
          🔐 QFLARE
        </Typography>
        <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
          Quantum-Safe Federated Learning
        </Typography>
        <Typography variant="body1" sx={{ mb: 4, maxWidth: 600 }}>
          Professional secure device registration with real SMS OTP delivery.
          Backend is running on port 8080, React on port 3000.
        </Typography>
        <Button 
          variant="contained" 
          size="large" 
          sx={{ 
            mt: 2,
            px: 4,
            py: 1.5,
            fontSize: '1.1rem',
            background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
            '&:hover': {
              background: 'linear-gradient(45deg, #1976D2 30%, #0097A7 90%)',
            }
          }}
          onClick={handleGetStarted}
        >
          Test Backend Connection
        </Button>

        <Box sx={{ mt: 4, p: 2, background: 'rgba(255,255,255,0.1)', borderRadius: 2 }}>
          <Typography variant="h6" gutterBottom>
            🚀 Ready Features:
          </Typography>
          <Typography variant="body2">
            ✅ Material UI Professional Design<br/>
            ✅ FastAPI Backend with CORS<br/>
            ✅ Twilio SMS Integration<br/>
            ✅ Device Registration System<br/>
            ✅ Admin Dashboard<br/>
            ✅ Real OTP Delivery
          </Typography>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;