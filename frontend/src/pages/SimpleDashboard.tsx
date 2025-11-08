import React from 'react';
import { Box, Typography, Paper, Grid, Button } from '@mui/material';
import { useAuth } from '../contexts/AuthContext';

const SimpleDashboard: React.FC = () => {
  const { user, logout } = useAuth();

  return (
    <Box sx={{ p: 4, minHeight: '100vh', backgroundColor: '#000000' }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom sx={{ color: '#ffffff' }}>
          QFLARE Dashboard
        </Typography>
        <Typography variant="body1" sx={{ color: '#ffffff', mb: 2 }}>
          Welcome, {user?.username || user?.email}
        </Typography>
        <Button 
          variant="outlined" 
          onClick={logout}
          sx={{ 
            color: '#ffffff', 
            borderColor: '#ffffff',
            '&:hover': {
              borderColor: '#ffffff',
              backgroundColor: '#333333'
            }
          }}
        >
          Logout
        </Button>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Paper 
            sx={{ 
              p: 3, 
              backgroundColor: '#ffffff', 
              color: '#000000',
              border: '1px solid #000000' 
            }}
          >
            <Typography variant="h6" gutterBottom>
              User Information
            </Typography>
            <Typography variant="body2">
              <strong>Email:</strong> {user?.email}
            </Typography>
            <Typography variant="body2">
              <strong>Username:</strong> {user?.username}
            </Typography>
            <Typography variant="body2">
              <strong>Role:</strong> {user?.role}
            </Typography>
            <Typography variant="body2">
              <strong>Status:</strong> {user?.is_active ? 'Active' : 'Inactive'}
            </Typography>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper 
            sx={{ 
              p: 3, 
              backgroundColor: '#ffffff', 
              color: '#000000',
              border: '1px solid #000000' 
            }}
          >
            <Typography variant="h6" gutterBottom>
              Federated Learning
            </Typography>
            <Typography variant="body2" gutterBottom>
              Secure distributed machine learning with post-quantum cryptography.
            </Typography>
            <Button 
              variant="contained" 
              sx={{ 
                mt: 2, 
                backgroundColor: '#000000', 
                color: '#ffffff',
                '&:hover': {
                  backgroundColor: '#333333'
                }
              }}
            >
              Start Training
            </Button>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper 
            sx={{ 
              p: 3, 
              backgroundColor: '#ffffff', 
              color: '#000000',
              border: '1px solid #000000' 
            }}
          >
            <Typography variant="h6" gutterBottom>
              System Status
            </Typography>
            <Typography variant="body2">
              <strong>Backend:</strong> Connected
            </Typography>
            <Typography variant="body2">
              <strong>Authentication:</strong> Active
            </Typography>
            <Typography variant="body2">
              <strong>Encryption:</strong> Post-Quantum Ready
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SimpleDashboard;