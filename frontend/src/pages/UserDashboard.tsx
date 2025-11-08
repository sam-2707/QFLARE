import React, { useState, useEffect } from 'react';
import { Box, Typography, Paper, Grid, Button, Card, CardContent, LinearProgress, Chip, TextField, Dialog, DialogTitle, DialogContent, DialogActions, Alert, Tabs, Tab, AppBar, Toolbar, IconButton } from '@mui/material';
import { useAuth } from '../contexts/AuthContext';
import { PlayArrow, Stop, CloudUpload, Assessment, Security, Key, Timeline, Edit, VpnKey, Dashboard as DashboardIcon, QueryStats, Shield, AccountCircle, Logout } from '@mui/icons-material';
import TrainingDashboard from '../components/TrainingDashboard';
import PrivacyDashboard from '../components/PrivacyDashboard';
import SecurityDashboard from '../components/SecurityDashboard';
import NotificationBell from '../components/NotificationBell';

interface DashboardStats {
  username: string;
  email: string;
  full_name: string | null;
  account_age_days: number;
  is_active: boolean;
  is_verified: boolean;
  role: string;
  has_keys: boolean;
  last_login: string | null;
  total_models: number;
  active_training_sessions: number;
}

interface KeysInfo {
  has_keys: boolean;
  encryption_key_type: string | null;
  signature_key_type: string | null;
  created_at: string | null;
}

const UserDashboard: React.FC = () => {
  const { user, logout } = useAuth();
  const [currentTab, setCurrentTab] = useState(0);
  const [trainingStatus, setTrainingStatus] = useState<'idle' | 'training' | 'completed'>('idle');
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [keysInfo, setKeysInfo] = useState<KeysInfo | null>(null);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [fullName, setFullName] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  useEffect(() => {
    fetchDashboardStats();
    fetchKeysInfo();
  }, []);

  const fetchDashboardStats = async () => {
    try {
      const token = localStorage.getItem('qflare-token');
      const response = await fetch('http://localhost:8002/api/user/dashboard-stats', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setStats(data);
        setFullName(data.full_name || '');
      }
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const fetchKeysInfo = async () => {
    try {
      const token = localStorage.getItem('qflare-token');
      const response = await fetch('http://localhost:8002/api/user/keys', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setKeysInfo(data);
      }
    } catch (error) {
      console.error('Error fetching keys:', error);
    }
  };

  const generateKeys = async () => {
    try {
      const token = localStorage.getItem('qflare-token');
      const response = await fetch('http://localhost:8002/api/user/generate-keys', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        setSuccess('Keys generated successfully!');
        fetchKeysInfo();
        fetchDashboardStats();
        setTimeout(() => setSuccess(''), 3000);
      } else {
        setError('Failed to generate keys');
      }
    } catch (error) {
      console.error('Error generating keys:', error);
      setError('Error generating keys');
    }
  };

  const updateProfile = async () => {
    try {
      const token = localStorage.getItem('qflare-token');
      const response = await fetch('http://localhost:8002/api/user/profile', {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ full_name: fullName })
      });
      if (response.ok) {
        setSuccess('Profile updated successfully!');
        fetchDashboardStats();
        setEditDialogOpen(false);
        setTimeout(() => setSuccess(''), 3000);
      } else {
        setError('Failed to update profile');
      }
    } catch (error) {
      console.error('Error updating profile:', error);
      setError('Error updating profile');
    }
  };

  const startTraining = () => {
    setTrainingStatus('training');
    // Simulate training progress
    const interval = setInterval(() => {
      setTrainingProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setTrainingStatus('completed');
          return 100;
        }
        return prev + 10;
      });
    }, 1000);
  };

  const stopTraining = () => {
    setTrainingStatus('idle');
    setTrainingProgress(0);
  };

  return (
    <Box sx={{ minHeight: '100vh', backgroundColor: '#f5f7fa' }}>
      {/* App Bar with Notification Bell */}
      <AppBar 
        position="static" 
        elevation={0}
        sx={{ 
          backgroundColor: '#ffffff',
          borderBottom: '2px solid #000',
        }}
      >
        <Toolbar>
          <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
            <Typography 
              variant="h6" 
              sx={{ 
                color: '#000000', 
                fontFamily: 'Montserrat, sans-serif', 
                fontWeight: 700,
                letterSpacing: '0.5px'
              }}
            >
              QFLARE USER DASHBOARD
            </Typography>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography 
              variant="body2" 
              sx={{ 
                color: '#666', 
                fontFamily: 'Montserrat, sans-serif',
                mr: 2
              }}
            >
              {stats?.username || user?.username || user?.email}
            </Typography>
            
            <NotificationBell />
            
            <IconButton
              onClick={logout}
              sx={{
                color: '#000',
                '&:hover': {
                  backgroundColor: '#f5f5f5'
                }
              }}
            >
              <Logout />
            </IconButton>
          </Box>
        </Toolbar>

        {/* Tabs Navigation */}
        <Tabs
          value={currentTab}
          onChange={(e, newValue) => setCurrentTab(newValue)}
          sx={{
            backgroundColor: '#ffffff',
            '& .MuiTab-root': {
              fontFamily: 'Montserrat, sans-serif',
              fontWeight: 600,
              fontSize: '0.875rem',
              color: '#666',
              '&.Mui-selected': {
                color: '#000',
              }
            },
            '& .MuiTabs-indicator': {
              backgroundColor: '#000',
              height: 3,
            }
          }}
        >
          <Tab icon={<DashboardIcon />} label="Overview" iconPosition="start" />
          <Tab icon={<PlayArrow />} label="FL Training" iconPosition="start" />
          <Tab icon={<QueryStats />} label="Privacy" iconPosition="start" />
          <Tab icon={<Shield />} label="Security" iconPosition="start" />
          <Tab icon={<AccountCircle />} label="Profile" iconPosition="start" />
        </Tabs>
      </AppBar>

      {/* Alerts */}
      {error && (
        <Box sx={{ p: 2 }}>
          <Alert severity="error" sx={{ borderRadius: '8px' }}>{error}</Alert>
        </Box>
      )}
      {success && (
        <Box sx={{ p: 2 }}>
          <Alert severity="success" sx={{ borderRadius: '8px' }}>{success}</Alert>
        </Box>
      )}

      {/* Tab Content */}
      <Box sx={{ p: 4 }}>
        {/* Tab 0: Overview - Existing Dashboard Content */}
        {currentTab === 0 && (
          <Box>
            {/* Quick Stats */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} md={3}>
                <Card sx={{ 
                  borderRadius: '12px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                  background: '#ffffff',
                  border: '2px solid #000',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: '0px 8px 24px rgba(0,0,0,0.12)',
                  },
                }}>
                  <CardContent>
                    <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 1, fontWeight: 600, color: '#000' }}>
                      MY MODELS
                    </Typography>
                    <Typography variant="h4" sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 700, color: '#000' }}>
                      {stats?.total_models || 0}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={3}>
                <Card sx={{ 
                  borderRadius: '12px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                  background: '#f5f5f5',
                  border: '2px solid #000',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: '0px 8px 24px rgba(0,0,0,0.12)',
                  },
                }}>
                  <CardContent>
                    <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 1, fontWeight: 600, color: '#000' }}>
                      ACTIVE SESSIONS
                    </Typography>
                    <Typography variant="h4" sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 700, color: '#000' }}>
                      {stats?.active_training_sessions || 0}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={3}>
                <Card sx={{ 
                  borderRadius: '12px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                  background: '#e0e0e0',
                  border: '2px solid #000',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: '0px 8px 24px rgba(0,0,0,0.12)',
                  },
                }}>
                  <CardContent>
                    <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 1, fontWeight: 600, color: '#000' }}>
                      ACCOUNT AGE
                    </Typography>
                    <Typography variant="h4" sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 700, color: '#000' }}>
                      {stats?.account_age_days || 0}d
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={3}>
                <Card sx={{ 
                  borderRadius: '12px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                  background: '#ffffff',
                  border: '2px solid #000',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: '0px 8px 24px rgba(0,0,0,0.12)',
                  },
                }}>
                  <CardContent>
                    <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 1, fontWeight: 600, color: '#000' }}>
                      PRIVACY LEVEL
                    </Typography>
                    <Typography variant="h4" sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 700, color: '#000' }}>
                      HIGH
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Main Training Section */}
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <Paper sx={{ 
                  p: 4, 
                  backgroundColor: '#000', 
                  color: '#fff', 
                  borderRadius: '16px',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                  minHeight: 400 
                }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                    <PlayArrow sx={{ fontSize: 40, mr: 2 }} />
                    <Typography variant="h5" sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 600 }}>
                      FEDERATED LEARNING TRAINING
                    </Typography>
                  </Box>

                  <Typography variant="body1" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 3 }}>
                    Start a secure federated learning training session with post-quantum encryption
                  </Typography>

                  {trainingStatus === 'idle' && (
                    <Box>
                      <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 2 }}>
                        ✓ Post-Quantum Keys: {keysInfo?.has_keys ? 'Generated' : 'Not Generated'}<br/>
                        ✓ Differential Privacy: Enabled (ε=0.1, δ=10⁻⁶)<br/>
                        ✓ Byzantine Protection: Active<br/>
                        ✓ Local Data: Ready
                      </Typography>
                      <Button 
                        variant="contained" 
                        size="large"
                        startIcon={<PlayArrow />}
                        onClick={startTraining}
                        sx={{ 
                          backgroundColor: '#fff',
                          color: '#000',
                          borderRadius: '8px',
                          fontFamily: 'Montserrat, sans-serif',
                          fontWeight: 600,
                          mt: 2,
                          px: 4,
                          '&:hover': {
                            backgroundColor: '#f5f5f5'
                          }
                        }}
                      >
                        START TRAINING SESSION
                      </Button>
                    </Box>
                  )}

                  {trainingStatus === 'training' && (
                    <Box>
                      <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 2 }}>
                        Training in progress... Round {Math.floor(trainingProgress / 10)} of 10
                      </Typography>
                      <LinearProgress 
                        variant="determinate" 
                        value={trainingProgress} 
                        sx={{ 
                          height: 10, 
                          mb: 2,
                          borderRadius: '5px',
                          backgroundColor: '#333',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: '#fff',
                            borderRadius: '5px'
                          }
                        }} 
                      />
                      <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 2 }}>
                        {trainingProgress}% Complete
                      </Typography>
                      <Button 
                        variant="outlined" 
                        size="large"
                        startIcon={<Stop />}
                        onClick={stopTraining}
                        sx={{ 
                          color: '#fff',
                          borderColor: '#fff',
                          borderRadius: '8px',
                          fontFamily: 'Montserrat, sans-serif',
                          fontWeight: 600,
                          mt: 2,
                          px: 4,
                          '&:hover': {
                            borderColor: '#fff',
                            backgroundColor: '#333'
                          }
                        }}
                      >
                        STOP TRAINING
                      </Button>
                    </Box>
                  )}

                  {trainingStatus === 'completed' && (
                    <Box>
                      <Typography variant="h6" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 2, color: '#4caf50' }}>
                        ✓ TRAINING COMPLETED
                      </Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 2 }}>
                        Final Accuracy: 89.3%<br/>
                        Privacy Budget Used: 0.08/0.1<br/>
                        Model Updates: 10<br/>
                        Training Time: 45s
                      </Typography>
                      <Button 
                        variant="contained" 
                        size="large"
                        startIcon={<PlayArrow />}
                        onClick={() => {
                          setTrainingStatus('idle');
                          setTrainingProgress(0);
                        }}
                        sx={{ 
                          backgroundColor: '#fff',
                          color: '#000',
                          borderRadius: '8px',
                          fontFamily: 'Montserrat, sans-serif',
                          fontWeight: 600,
                          mt: 2,
                          px: 4,
                          '&:hover': {
                            backgroundColor: '#f5f5f5'
                          }
                        }}
                      >
                        START NEW SESSION
                      </Button>
                    </Box>
                  )}
                </Paper>
              </Grid>

              <Grid item xs={12} md={4}>
                {/* User Info */}
                <Paper sx={{ 
                  p: 3, 
                  backgroundColor: '#fff', 
                  borderRadius: '12px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                  mb: 3 
                }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6" sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 600 }}>
                      ACCOUNT INFO
                    </Typography>
                    <Button
                      size="small"
                      startIcon={<Edit />}
                      onClick={() => setEditDialogOpen(true)}
                      sx={{
                        color: '#000',
                        fontFamily: 'Montserrat, sans-serif',
                        minWidth: 'auto'
                      }}
                    >
                      EDIT
                    </Button>
                  </Box>
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 1 }}>
                      <strong>Email:</strong> {stats?.email || user?.email}
                    </Typography>
                    <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 1 }}>
                      <strong>Username:</strong> {stats?.username || user?.username}
                    </Typography>
                    <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 1 }}>
                      <strong>Full Name:</strong> {stats?.full_name || 'Not set'}
                    </Typography>
                    <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 1 }}>
                      <strong>Role:</strong> {stats?.role?.toUpperCase() || user?.role?.toUpperCase()}
                    </Typography>
                    <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 1 }}>
                      <strong>Status:</strong>{' '}
                      <Chip 
                        label={stats?.is_active ? 'ACTIVE' : 'INACTIVE'} 
                        size="small"
                        sx={{ 
                          backgroundColor: stats?.is_active ? '#e8f5e9' : '#ffebee',
                          color: '#000',
                          fontFamily: 'Montserrat, sans-serif',
                          borderRadius: '6px',
                        }}
                      />
                    </Typography>
                    {stats?.last_login && (
                      <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mt: 1 }}>
                        <strong>Last Login:</strong> {new Date(stats.last_login).toLocaleDateString()}
                      </Typography>
                    )}
                  </Box>
                </Paper>

                {/* Cryptographic Keys Status */}
                <Paper sx={{ 
                  p: 3, 
                  backgroundColor: '#fff', 
                  borderRadius: '12px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                  mb: 3 
                }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <VpnKey sx={{ fontSize: 30, mr: 1 }} />
                    <Typography variant="h6" sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 600 }}>
                      CRYPTO KEYS
                    </Typography>
                  </Box>
                  {keysInfo?.has_keys ? (
                    <>
                      <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 1 }}>
                        <strong>Encryption:</strong> {keysInfo.encryption_key_type || 'Kyber-1024'}
                      </Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 1 }}>
                        <strong>Signature:</strong> {keysInfo.signature_key_type || 'Dilithium-2'}
                      </Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 1 }}>
                        <strong>Created:</strong> {keysInfo.created_at ? new Date(keysInfo.created_at).toLocaleDateString() : 'N/A'}
                      </Typography>
                      <Chip 
                        label="QUANTUM-SAFE" 
                        size="small"
                        sx={{ 
                          backgroundColor: '#e8f5e9',
                          color: '#000',
                          fontFamily: 'Montserrat, sans-serif',
                          borderRadius: '6px',
                          mt: 1
                        }}
                      />
                    </>
                  ) : (
                    <>
                      <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 2, color: '#d32f2f' }}>
                        No cryptographic keys generated yet
                      </Typography>
                      <Button
                        fullWidth
                        variant="contained"
                        onClick={generateKeys}
                        sx={{
                          bgcolor: '#000',
                          color: '#fff',
                          borderRadius: '8px',
                          fontFamily: 'Montserrat, sans-serif',
                          '&:hover': {
                            bgcolor: '#333'
                          }
                        }}
                      >
                        GENERATE KEYS
                      </Button>
                    </>
                  )}
                </Paper>

                {/* Security Status */}
                <Paper sx={{ 
                  p: 3, 
                  backgroundColor: '#fff', 
                  borderRadius: '12px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                  mb: 3 
                }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Security sx={{ fontSize: 30, mr: 1 }} />
                    <Typography variant="h6" sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 600 }}>
                      SECURITY
                    </Typography>
                  </Box>
                  <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 1 }}>
                    <strong>Encryption:</strong> Kyber-1024
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 1 }}>
                    <strong>Signatures:</strong> Dilithium-2
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 1 }}>
                    <strong>Privacy:</strong> Differential (DP)
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', mb: 1 }}>
                    <strong>Status:</strong>{' '}
                    <Chip 
                      label="PROTECTED" 
                      size="small"
                      sx={{ 
                        backgroundColor: '#e8f5e9',
                        color: '#000',
                        fontFamily: 'Montserrat, sans-serif',
                        borderRadius: '6px',
                      }}
                    />
                  </Typography>
                </Paper>

                {/* Quick Actions */}
                <Paper sx={{ 
                  p: 3, 
                  backgroundColor: '#fff', 
                  borderRadius: '12px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                }}>
                  <Typography variant="h6" gutterBottom sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 600 }}>
                    QUICK ACTIONS
                  </Typography>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<CloudUpload />}
                    sx={{
                      color: '#000',
                      borderColor: '#e0e0e0',
                      borderRadius: '8px',
                      fontFamily: 'Montserrat, sans-serif',
                      mb: 1,
                      '&:hover': {
                        borderColor: '#000',
                        backgroundColor: '#f5f5f5'
                      }
                    }}
                  >
                    UPLOAD DATA
                  </Button>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<Assessment />}
                    sx={{
                      color: '#000',
                      borderColor: '#e0e0e0',
                      borderRadius: '8px',
                      fontFamily: 'Montserrat, sans-serif',
                      mb: 1,
                      '&:hover': {
                        borderColor: '#000',
                        backgroundColor: '#f5f5f5'
                      }
                    }}
                  >
                    VIEW REPORTS
                  </Button>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<Key />}
                    onClick={() => !keysInfo?.has_keys && generateKeys()}
                    disabled={keysInfo?.has_keys}
                    sx={{
                      color: '#000',
                      borderColor: '#e0e0e0',
                      borderRadius: '8px',
                      fontFamily: 'Montserrat, sans-serif',
                      '&:hover': {
                        borderColor: '#000',
                        backgroundColor: '#f5f5f5'
                      }
                    }}
                  >
                    {keysInfo?.has_keys ? 'KEYS READY' : 'GENERATE KEYS'}
                  </Button>
                </Paper>
              </Grid>
            </Grid>
          </Box>
        )}

        {/* Tab 1: FL Training Dashboard */}
        {currentTab === 1 && <TrainingDashboard />}

        {/* Tab 2: Privacy Dashboard */}
        {currentTab === 2 && <PrivacyDashboard />}

        {/* Tab 3: Security Dashboard */}
        {currentTab === 3 && <SecurityDashboard />}

        {/* Tab 4: Profile Settings */}
        {currentTab === 4 && (
          <Box>
            <Paper sx={{ 
              p: 4, 
              borderRadius: '16px',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
              maxWidth: 800,
              mx: 'auto'
            }}>
              <Typography variant="h5" gutterBottom sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 700, mb: 3 }}>
                Profile Settings
              </Typography>
              
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Email"
                    value={stats?.email || user?.email || ''}
                    disabled
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        borderRadius: '8px',
                        fontFamily: 'Montserrat, sans-serif'
                      }
                    }}
                  />
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Username"
                    value={stats?.username || user?.username || ''}
                    disabled
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        borderRadius: '8px',
                        fontFamily: 'Montserrat, sans-serif'
                      }
                    }}
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Full Name"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        borderRadius: '8px',
                        fontFamily: 'Montserrat, sans-serif'
                      }
                    }}
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <Button
                    variant="contained"
                    onClick={updateProfile}
                    sx={{
                      bgcolor: '#000',
                      color: '#fff',
                      borderRadius: '8px',
                      fontFamily: 'Montserrat, sans-serif',
                      px: 4,
                      py: 1.5,
                      '&:hover': { bgcolor: '#333' }
                    }}
                  >
                    Save Changes
                  </Button>
                </Grid>
              </Grid>

              <Box sx={{ mt: 4, pt: 4, borderTop: '1px solid #e0e0e0' }}>
                <Typography variant="h6" gutterBottom sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 600, mb: 2 }}>
                  Account Information
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', color: '#666' }}>
                      Role
                    </Typography>
                    <Typography variant="body1" sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 600 }}>
                      {stats?.role?.toUpperCase() || user?.role?.toUpperCase()}
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', color: '#666' }}>
                      Account Age
                    </Typography>
                    <Typography variant="body1" sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 600 }}>
                      {stats?.account_age_days || 0} days
                    </Typography>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', color: '#666' }}>
                      Status
                    </Typography>
                    <Chip 
                      label={stats?.is_active ? 'ACTIVE' : 'INACTIVE'} 
                      size="small"
                      sx={{ 
                        backgroundColor: stats?.is_active ? '#e8f5e9' : '#ffebee',
                        color: '#000',
                        fontFamily: 'Montserrat, sans-serif',
                        borderRadius: '6px',
                        mt: 0.5
                      }}
                    />
                  </Grid>
                  
                  {stats?.last_login && (
                    <Grid item xs={12} md={6}>
                      <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', color: '#666' }}>
                        Last Login
                      </Typography>
                      <Typography variant="body1" sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 600 }}>
                        {new Date(stats.last_login).toLocaleString()}
                      </Typography>
                    </Grid>
                  )}
                </Grid>
              </Box>
            </Paper>
          </Box>
        )}
      </Box>

      {/* Edit Profile Dialog - Keep for backward compatibility */}
      <Dialog
        open={editDialogOpen}
        onClose={() => setEditDialogOpen(false)}
        PaperProps={{ sx: { borderRadius: '12px', minWidth: 400 } }}
      >
        <DialogTitle sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 700 }}>
          EDIT PROFILE
        </DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Full Name"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            sx={{
              mt: 2,
              '& .MuiOutlinedInput-root': {
                borderRadius: '8px',
                fontFamily: 'Montserrat, sans-serif'
              }
            }}
          />
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => setEditDialogOpen(false)}
            sx={{
              color: '#000',
              borderRadius: '8px',
              fontFamily: 'Montserrat, sans-serif'
            }}
          >
            CANCEL
          </Button>
          <Button
            onClick={updateProfile}
            variant="contained"
            sx={{
              bgcolor: '#000',
              color: '#fff',
              borderRadius: '8px',
              fontFamily: 'Montserrat, sans-serif',
              '&:hover': { bgcolor: '#333' }
            }}
          >
            SAVE
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default UserDashboard;
