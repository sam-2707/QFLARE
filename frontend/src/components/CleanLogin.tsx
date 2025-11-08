import React, { useState } from 'react';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  Alert,
  InputAdornment,
  IconButton,
  Divider,
  Link
} from '@mui/material';
import {
  Visibility,
  VisibilityOff,
  Email,
  Lock,
  Person
} from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';

const CleanLogin: React.FC = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    username: '',
    fullName: '',
    confirmPassword: ''
  });

  const { login, register } = useAuth();

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (error) setError('');
    if (successMessage) setSuccessMessage('');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccessMessage('');

    try {
      if (isLogin) {
        const success = await login(formData.email, formData.password);
        if (!success) {
          setError('Invalid email or password');
        }
      } else {
        if (formData.password !== formData.confirmPassword) {
          setError('Passwords do not match');
          setLoading(false);
          return;
        }
        const success = await register({
          email: formData.email,
          username: formData.username,
          full_name: formData.fullName,
          password: formData.password
        });
        if (success) {
          setIsLogin(true);
          setSuccessMessage('Registration successful! Account pending admin approval. Use test credentials to login: admin@qflare.com/admin123');
          setFormData({ email: '', password: '', username: '', fullName: '', confirmPassword: '' });
        }
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="sm" sx={{ 
      minHeight: '100vh', 
      display: 'flex', 
      alignItems: 'center', 
      py: 4,
      bgcolor: 'white'
    }}>
      <Paper 
        elevation={0}
        sx={{ 
          width: '100%', 
          p: 4,
          border: '2px solid #000',
          borderRadius: 0,
          bgcolor: 'white'
        }}
      >
        {/* Logo/Title */}
        <Box sx={{ textAlign: 'center', mb: 4 }}>
          <Typography 
            variant="h4" 
            sx={{ 
              fontWeight: 700,
              color: '#000',
              fontFamily: 'monospace',
              letterSpacing: 2
            }}
          >
            QFLARE
          </Typography>
          <Typography 
            variant="body2" 
            sx={{ 
              color: '#666',
              mt: 1,
              fontFamily: 'monospace'
            }}
          >
            Quantum-Safe Federated Learning
          </Typography>
        </Box>

        {/* Toggle Login/Register */}
        <Box sx={{ display: 'flex', mb: 3, border: '1px solid #000' }}>
          <Button
            fullWidth
            onClick={() => setIsLogin(true)}
            sx={{
              py: 1.5,
              backgroundColor: isLogin ? '#000' : 'transparent',
              color: isLogin ? '#fff' : '#000',
              border: 'none',
              borderRadius: 0,
              fontWeight: 600,
              '&:hover': {
                backgroundColor: isLogin ? '#333' : '#f5f5f5',
              }
            }}
          >
            LOGIN
          </Button>
          <Button
            fullWidth
            onClick={() => setIsLogin(false)}
            sx={{
              py: 1.5,
              backgroundColor: !isLogin ? '#000' : 'transparent',
              color: !isLogin ? '#fff' : '#000',
              border: 'none',
              borderRadius: 0,
              fontWeight: 600,
              borderLeft: '1px solid #000',
              '&:hover': {
                backgroundColor: !isLogin ? '#333' : '#f5f5f5',
              }
            }}
          >
            REGISTER
          </Button>
        </Box>

        {/* Error Alert */}
        {error && (
          <Alert 
            severity="error" 
            sx={{ 
              mb: 3,
              bgcolor: 'white',
              color: '#000',
              border: '1px solid #000',
              borderRadius: 0,
              '& .MuiAlert-icon': {
                color: '#000'
              }
            }}
          >
            {error}
          </Alert>
        )}

        {/* Success Alert */}
        {successMessage && (
          <Alert 
            severity="success" 
            sx={{ 
              mb: 3,
              bgcolor: '#f0fff0',
              color: '#000',
              border: '1px solid #000',
              borderRadius: 0,
              '& .MuiAlert-icon': {
                color: '#000'
              }
            }}
          >
            {successMessage}
          </Alert>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit}>
          <Box sx={{ mb: 3 }}>
            <TextField
              fullWidth
              label="Email"
              type="email"
              value={formData.email}
              onChange={(e) => handleInputChange('email', e.target.value)}
              required
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Email sx={{ color: '#000' }} />
                  </InputAdornment>
                ),
              }}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: 0,
                  '& fieldset': {
                    borderColor: '#000',
                    borderWidth: 2,
                  },
                  '&:hover fieldset': {
                    borderColor: '#000',
                  },
                  '&.Mui-focused fieldset': {
                    borderColor: '#000',
                  },
                },
                '& .MuiInputLabel-root': {
                  color: '#666',
                  fontFamily: 'monospace',
                  '&.Mui-focused': {
                    color: '#000',
                  },
                },
              }}
            />
          </Box>

          {!isLogin && (
            <>
              <Box sx={{ mb: 3 }}>
                <TextField
                  fullWidth
                  label="Username"
                  value={formData.username}
                  onChange={(e) => handleInputChange('username', e.target.value)}
                  required
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <Person sx={{ color: '#000' }} />
                      </InputAdornment>
                    ),
                  }}
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 0,
                      '& fieldset': {
                        borderColor: '#000',
                        borderWidth: 2,
                      },
                      '&:hover fieldset': {
                        borderColor: '#000',
                      },
                      '&.Mui-focused fieldset': {
                        borderColor: '#000',
                      },
                    },
                    '& .MuiInputLabel-root': {
                      color: '#666',
                      fontFamily: 'monospace',
                      '&.Mui-focused': {
                        color: '#000',
                      },
                    },
                  }}
                />
              </Box>
              <Box sx={{ mb: 3 }}>
                <TextField
                  fullWidth
                  label="Full Name (Optional)"
                  value={formData.fullName}
                  onChange={(e) => handleInputChange('fullName', e.target.value)}
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 0,
                      '& fieldset': {
                        borderColor: '#000',
                        borderWidth: 2,
                      },
                      '&:hover fieldset': {
                        borderColor: '#000',
                      },
                      '&.Mui-focused fieldset': {
                        borderColor: '#000',
                      },
                    },
                    '& .MuiInputLabel-root': {
                      color: '#666',
                      fontFamily: 'monospace',
                      '&.Mui-focused': {
                        color: '#000',
                      },
                    },
                  }}
                />
              </Box>
            </>
          )}

          <Box sx={{ mb: 3 }}>
            <TextField
              fullWidth
              label="Password"
              type={showPassword ? 'text' : 'password'}
              value={formData.password}
              onChange={(e) => handleInputChange('password', e.target.value)}
              required
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Lock sx={{ color: '#000' }} />
                  </InputAdornment>
                ),
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      onClick={() => setShowPassword(!showPassword)}
                      sx={{ color: '#000' }}
                    >
                      {showPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: 0,
                  '& fieldset': {
                    borderColor: '#000',
                    borderWidth: 2,
                  },
                  '&:hover fieldset': {
                    borderColor: '#000',
                  },
                  '&.Mui-focused fieldset': {
                    borderColor: '#000',
                  },
                },
                '& .MuiInputLabel-root': {
                  color: '#666',
                  fontFamily: 'monospace',
                  '&.Mui-focused': {
                    color: '#000',
                  },
                },
              }}
            />
          </Box>

          {!isLogin && (
            <Box sx={{ mb: 3 }}>
              <TextField
                fullWidth
                label="Confirm Password"
                type={showPassword ? 'text' : 'password'}
                value={formData.confirmPassword}
                onChange={(e) => handleInputChange('confirmPassword', e.target.value)}
                required
                sx={{
                  '& .MuiOutlinedInput-root': {
                    borderRadius: 0,
                    '& fieldset': {
                      borderColor: '#000',
                      borderWidth: 2,
                    },
                    '&:hover fieldset': {
                      borderColor: '#000',
                    },
                    '&.Mui-focused fieldset': {
                      borderColor: '#000',
                    },
                  },
                  '& .MuiInputLabel-root': {
                    color: '#666',
                    fontFamily: 'monospace',
                    '&.Mui-focused': {
                      color: '#000',
                    },
                  },
                }}
              />
            </Box>
          )}

          <Button
            type="submit"
            fullWidth
            disabled={loading}
            sx={{
              py: 2,
              backgroundColor: '#000',
              color: '#fff',
              border: '2px solid #000',
              borderRadius: 0,
              fontWeight: 700,
              fontFamily: 'monospace',
              fontSize: '1rem',
              '&:hover': {
                backgroundColor: '#333',
              },
              '&:disabled': {
                backgroundColor: '#666',
                color: '#ccc',
              },
            }}
          >
            {loading ? 'PROCESSING...' : (isLogin ? 'LOGIN' : 'REGISTER')}
          </Button>
        </form>

        {/* Test Credentials */}
        <Box sx={{ mt: 4, p: 2, bgcolor: '#f9f9f9', border: '1px solid #000' }}>
          <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 600, mb: 1 }}>
            TEST CREDENTIALS:
          </Typography>
          <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
            Admin: admin@qflare.com / admin123<br/>
            User: user@qflare.com / user123
          </Typography>
        </Box>
      </Paper>
    </Container>
  );
};

export default CleanLogin;