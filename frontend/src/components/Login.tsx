import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Alert,
  CircularProgress,
  Container,
  Avatar,
  Tabs,
  Tab,
  Divider
} from '@mui/material';
import {
  Login as LoginIcon,
  AdminPanelSettings as AdminIcon,
  Person as UserIcon,
  Security as SecurityIcon
} from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <Box
      role="tabpanel"
      hidden={value !== index}
      id={`login-tabpanel-${index}`}
      aria-labelledby={`login-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </Box>
  );
}

const Login = () => {
  const [tab, setTab] = useState(0);
  const [credentials, setCredentials] = useState({
    username: '',
    password: ''
  });
  const [error, setError] = useState<string | null>(null);
  const { login, loading } = useAuth();

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTab(newValue);
    setError(null);
    // Pre-fill credentials based on tab
    if (newValue === 0) {
      setCredentials({ username: 'admin', password: 'admin123' });
    } else {
      setCredentials({ username: 'user', password: 'user123' });
    }
  };

  const handleInputChange = (field: string) => (event: React.ChangeEvent<HTMLInputElement>) => {
    setCredentials(prev => ({
      ...prev,
      [field]: event.target.value
    }));
    if (error) setError(null);
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);

    if (!credentials.username || !credentials.password) {
      setError('Please enter both username and password');
      return;
    }

    const success = await login(credentials.username, credentials.password);
    if (!success) {
      setError('Invalid credentials. Please try again.');
    }
  };

  // Set default credentials on component mount
  React.useEffect(() => {
    setCredentials({ username: 'admin', password: 'admin123' });
  }, []);

  return (
    <Container maxWidth="sm" sx={{ 
      minHeight: '100vh', 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    }}>
      <Card sx={{ 
        width: '100%', 
        maxWidth: 450,
        boxShadow: '0 20px 40px rgba(0,0,0,0.1)',
        borderRadius: 3
      }}>
        <CardContent sx={{ p: 4 }}>
          {/* Header */}
          <Box sx={{ textAlign: 'center', mb: 3 }}>
            <Avatar sx={{ 
              width: 80, 
              height: 80, 
              mx: 'auto', 
              mb: 2,
              background: 'linear-gradient(45deg, #667eea, #764ba2)'
            }}>
              <SecurityIcon sx={{ fontSize: 40 }} />
            </Avatar>
            <Typography variant="h4" fontWeight="bold" color="primary">
              QFLARE
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Quantum-Safe Federated Learning Platform
            </Typography>
          </Box>

          {/* Role Selection Tabs */}
          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
            <Tabs value={tab} onChange={handleTabChange} variant="fullWidth">
              <Tab 
                label="Administrator" 
                icon={<AdminIcon />} 
                iconPosition="start"
                sx={{ textTransform: 'none' }}
              />
              <Tab 
                label="User" 
                icon={<UserIcon />} 
                iconPosition="start"
                sx={{ textTransform: 'none' }}
              />
            </Tabs>
          </Box>

          {/* Admin Login */}
          <TabPanel value={tab} index={0}>
            <Box sx={{ textAlign: 'center', mb: 3 }}>
              <Typography variant="h6" color="primary">
                Administrator Access
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Full system management and configuration
              </Typography>
            </Box>
          </TabPanel>

          {/* User Login */}
          <TabPanel value={tab} index={1}>
            <Box sx={{ textAlign: 'center', mb: 3 }}>
              <Typography variant="h6" color="primary">
                User Access
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Personal training and model submission
              </Typography>
            </Box>
          </TabPanel>

          {/* Login Form */}
          <Box component="form" onSubmit={handleSubmit}>
            <TextField
              fullWidth
              label="Username"
              value={credentials.username}
              onChange={handleInputChange('username')}
              margin="normal"
              required
              autoFocus
              disabled={loading}
            />
            <TextField
              fullWidth
              label="Password"
              type="password"
              value={credentials.password}
              onChange={handleInputChange('password')}
              margin="normal"
              required
              disabled={loading}
            />

            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}

            <Button
              type="submit"
              fullWidth
              variant="contained"
              size="large"
              disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : <LoginIcon />}
              sx={{ 
                mt: 3, 
                mb: 2,
                py: 1.5,
                background: 'linear-gradient(45deg, #667eea, #764ba2)',
                '&:hover': {
                  background: 'linear-gradient(45deg, #5a67d8, #6b46c1)',
                }
              }}
            >
              {loading ? 'Signing In...' : 'Sign In'}
            </Button>
          </Box>

          <Divider sx={{ my: 3 }} />

          {/* Demo Credentials */}
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="body2" color="textSecondary" gutterBottom>
              Demo Credentials:
            </Typography>
            <Typography variant="body2" color="primary">
              Admin: admin / admin123
            </Typography>
            <Typography variant="body2" color="secondary">
              User: user / user123
            </Typography>
          </Box>
        </CardContent>
      </Card>
    </Container>
  );
};

export default Login;