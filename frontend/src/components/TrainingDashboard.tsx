import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  LinearProgress,
  Chip,
  Button,
  Alert,
  IconButton,
  Tooltip,
  Skeleton,
  Snackbar,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Refresh,
  TrendingUp,
  Security,
  Speed,
  Memory,
  CloudOff,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { authenticatedFetch, parseErrorResponse } from '../utils/api';
import FederatedNodes from './FederatedNodes';
import ModelConvergence from './ModelConvergence';
import FLConfigPanel from './FLConfigPanel';
import ModelExport from './ModelExport';

interface TrainingSession {
  training_id: string;
  status: string;
  current_round: number;
  total_rounds: number;
  accuracy: number;
  loss: number;
  epsilon_used: number;
  epsilon_budget: number;
  delta: number;
  participants: number;
  byzantine_detected: number;
  current_learning_rate: number;
  computation_time: number;
  communication_overhead: number;
}

const TrainingDashboard: React.FC = () => {
  const [session, setSession] = useState<TrainingSession | null>(null);
  const [loading, setLoading] = useState(false);
  const [accuracyHistory, setAccuracyHistory] = useState<any[]>([]);
  const [lossHistory, setLossHistory] = useState<any[]>([]);
  const [error, setError] = useState<string>('');
  const [showError, setShowError] = useState(false);
  const [success, setSuccess] = useState<string>('');
  const [showSuccess, setShowSuccess] = useState(false);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      fetchTrainingStatus();
    };
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    await fetchTrainingStatus();
    setRefreshing(false);
  };

  const fetchTrainingStatus = async () => {
    try {
      const response = await authenticatedFetch('/api/training/status', {
        timeout: 10000,
      });
      if (!response.ok) {
        const errorMsg = await parseErrorResponse(response);
        throw new Error(errorMsg);
      }
      const data = await response.json();
      setSession(data);
    } catch (err: any) {
      console.error('Status fetch error:', err);
    }
  };

  const startTraining = async () => {
    setLoading(true);
    try {
      const formData = new URLSearchParams({
        model_type: 'CNN',
        dataset: 'MNIST',
        rounds: '10',
        epsilon: '0.1',
        delta: '1e-6',
      });

      const response = await authenticatedFetch('/api/training/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formData,
        timeout: 20000, // 20 seconds for starting training
      });

      if (!response.ok) {
        const errorMsg = await parseErrorResponse(response);
        throw new Error(errorMsg);
      }

      const data = await response.json();
      if (data.training_id) {
        setSuccess('Training started successfully!');
        setShowSuccess(true);
        pollTrainingStatus(data.training_id);
      }
    } catch (error: any) {
      console.error('Failed to start training:', error);
      setError(error.message || 'Failed to start training. Please try again.');
      setShowError(true);
    } finally {
      setLoading(false);
    }
  };

  const pollTrainingStatus = async (trainingId: string) => {
    try {
      const response = await authenticatedFetch(`/api/training/${trainingId}/status`, {
        timeout: 10000,
      });

      if (!response.ok) return;

      const data = await response.json();
      setSession(data);

      // Update history for charts
      setAccuracyHistory((prev) => [
        ...prev,
        { round: data.current_round, accuracy: (data.accuracy * 100).toFixed(2) },
      ]);

      setLossHistory((prev) => [
        ...prev,
        { round: data.current_round, loss: data.loss.toFixed(4) },
      ]);

      // Continue polling if training is active
      if (data.status === 'training') {
        setTimeout(() => pollTrainingStatus(trainingId), 3000);
      }
    } catch (error) {
      console.error('Failed to fetch training status:', error);
    }
  };

  const stopTraining = async () => {
    if (!session) return;
    
    // Optimistic update
    const previousSession = session;
    setSession({ ...session, status: 'stopped' });
    
    try {
      const response = await authenticatedFetch(`/api/training/${session.training_id}/stop`, {
        method: 'POST',
        timeout: 10000,
      });
      
      if (!response.ok) {
        const errorMsg = await parseErrorResponse(response);
        throw new Error(errorMsg);
      }
      
      setSession(null);
      setAccuracyHistory([]);
      setLossHistory([]);
      setSuccess('Training stopped successfully!');
      setShowSuccess(true);
    } catch (error: any) {
      console.error('Failed to stop training:', error);
      // Rollback on error
      setSession(previousSession);
      setError(error.message || 'Failed to stop training. Please try again.');
      setShowError(true);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'training':
        return 'primary';
      case 'completed':
        return 'success';
      case 'stopped':
        return 'warning';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  const privacyData = session
    ? [
        { name: 'Used', value: session.epsilon_used, color: '#000' },
        {
          name: 'Remaining',
          value: session.epsilon_budget - session.epsilon_used,
          color: '#666',
        },
      ]
    : [];

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 600, color: '#000', fontFamily: 'Montserrat, sans-serif' }}>
          🔬 Federated Learning Training
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          {!isOnline && (
            <Chip 
              icon={<CloudOff />} 
              label="Offline" 
              color="error" 
              size="small" 
              sx={{ fontFamily: 'Montserrat, sans-serif' }}
            />
          )}
          <Tooltip title="Refresh status">
            <IconButton 
              onClick={handleRefresh} 
              disabled={refreshing || !isOnline}
              sx={{ 
                border: '2px solid #000',
                '&:hover': { backgroundColor: '#f5f5f5' }
              }}
            >
              <Refresh sx={{ color: '#000' }} />
            </IconButton>
          </Tooltip>
          {!session && (
            <Button
              variant="contained"
              startIcon={<PlayArrow />}
              onClick={startTraining}
              disabled={loading || !isOnline}
              sx={{ borderRadius: '8px', textTransform: 'none', fontWeight: 600 }}
            >
              Start Training
            </Button>
          )}
          {session && session.status === 'training' && (
            <Button
              variant="contained"
              color="error"
              startIcon={<Stop />}
              onClick={stopTraining}
              sx={{ borderRadius: '8px', textTransform: 'none', fontWeight: 600 }}
            >
              Stop Training
            </Button>
          )}
          <IconButton onClick={() => session && pollTrainingStatus(session.training_id)}>
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      {!session && (
        <Alert severity="info" sx={{ mb: 3, borderRadius: '12px' }}>
          Click "Start Training" to begin a federated learning session with differential privacy and Byzantine detection.
        </Alert>
      )}

      {session && (
        <>
          {/* Status Header */}
          <Card sx={{ mb: 3, borderRadius: '12px', boxShadow: '0px 4px 12px rgba(0,0,0,0.08)' }}>
            <CardContent>
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="body2" color="textSecondary">
                    Status
                  </Typography>
                  <Chip
                    label={session.status.toUpperCase()}
                    color={getStatusColor(session.status)}
                    sx={{ mt: 1, fontWeight: 600 }}
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="body2" color="textSecondary">
                    Progress
                  </Typography>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Round {session.current_round} / {session.total_rounds}
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={(session.current_round / session.total_rounds) * 100}
                    sx={{ mt: 1, height: 8, borderRadius: 4 }}
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="body2" color="textSecondary" sx={{ fontFamily: 'Montserrat, sans-serif' }}>
                    Accuracy
                  </Typography>
                  <Typography variant="h6" sx={{ fontWeight: 600, color: '#000', fontFamily: 'Montserrat, sans-serif' }}>
                    {(session.accuracy * 100).toFixed(2)}%
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="body2" color="textSecondary" sx={{ fontFamily: 'Montserrat, sans-serif' }}>
                    Loss
                  </Typography>
                  <Typography variant="h6" sx={{ fontWeight: 600, color: '#000', fontFamily: 'Montserrat, sans-serif' }}>
                    {session.loss.toFixed(4)}
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Metrics Cards */}
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} md={3}>
              <Card
                sx={{
                  borderRadius: '12px',
                  boxShadow: '0px 4px 12px rgba(0,0,0,0.08)',
                  background: '#ffffff',
                  border: '2px solid #000',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: '0px 8px 24px rgba(0,0,0,0.12)',
                  },
                }}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box>
                      <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 600 }}>
                        Participants
                      </Typography>
                      <Typography variant="h4" sx={{ fontWeight: 700, mt: 1, color: '#000', fontFamily: 'Montserrat, sans-serif' }}>
                        {session.participants}
                      </Typography>
                    </Box>
                    <Speed sx={{ fontSize: 48, color: '#000', opacity: 0.2 }} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card
                sx={{
                  borderRadius: '12px',
                  boxShadow: '0px 4px 12px rgba(0,0,0,0.08)',
                  background: '#f5f5f5',
                  border: '2px solid #000',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: '0px 8px 24px rgba(0,0,0,0.12)',
                  },
                }}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box>
                      <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 600 }}>
                        Byzantine Detected
                      </Typography>
                      <Typography variant="h4" sx={{ fontWeight: 700, mt: 1, color: '#000', fontFamily: 'Montserrat, sans-serif' }}>
                        {session.byzantine_detected}
                      </Typography>
                    </Box>
                    <Security sx={{ fontSize: 48, color: '#000', opacity: 0.2 }} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card
                sx={{
                  borderRadius: '12px',
                  boxShadow: '0px 4px 12px rgba(0,0,0,0.08)',
                  background: '#e0e0e0',
                  border: '2px solid #000',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: '0px 8px 24px rgba(0,0,0,0.12)',
                  },
                }}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box>
                      <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 600 }}>
                        Computation Time
                      </Typography>
                      <Typography variant="h4" sx={{ fontWeight: 700, mt: 1, color: '#000', fontFamily: 'Montserrat, sans-serif' }}>
                        {session.computation_time.toFixed(1)}s
                      </Typography>
                    </Box>
                    <TrendingUp sx={{ fontSize: 48, color: '#000', opacity: 0.2 }} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={3}>
              <Card
                sx={{
                  borderRadius: '12px',
                  boxShadow: '0px 4px 12px rgba(0,0,0,0.08)',
                  background: '#ffffff',
                  border: '2px solid #000',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: '0px 8px 24px rgba(0,0,0,0.12)',
                  },
                }}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box>
                      <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 600 }}>
                        Communication
                      </Typography>
                      <Typography variant="h4" sx={{ fontWeight: 700, mt: 1, color: '#000', fontFamily: 'Montserrat, sans-serif' }}>
                        {session.communication_overhead.toFixed(1)} MB
                      </Typography>
                    </Box>
                    <Memory sx={{ fontSize: 48, color: '#000', opacity: 0.2 }} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Charts */}
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} md={8}>
              <Card sx={{ borderRadius: '12px', boxShadow: '0px 4px 12px rgba(0,0,0,0.08)', border: '2px solid #000' }}>
                <CardContent>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 2, fontFamily: 'Montserrat, sans-serif' }}>
                    📈 Training Metrics Over Rounds
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={accuracyHistory}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                      <XAxis dataKey="round" label={{ value: 'Round', position: 'insideBottom', offset: -5, style: { fontFamily: 'Montserrat, sans-serif' } }} tick={{ fontFamily: 'Montserrat, sans-serif' }} />
                      <YAxis label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft', style: { fontFamily: 'Montserrat, sans-serif' } }} tick={{ fontFamily: 'Montserrat, sans-serif' }} />
                      <RechartsTooltip />
                      <Legend wrapperStyle={{ fontFamily: 'Montserrat, sans-serif' }} />
                      <Line
                        type="monotone"
                        dataKey="accuracy"
                        stroke="#000"
                        strokeWidth={3}
                        dot={{ r: 4, fill: '#000' }}
                        name="Accuracy (%)"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card sx={{ borderRadius: '12px', boxShadow: '0px 4px 12px rgba(0,0,0,0.08)', height: '100%', border: '2px solid #000' }}>
                <CardContent>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 2, fontFamily: 'Montserrat, sans-serif' }}>
                    🔒 Privacy Budget
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 2 }}>
                    <ResponsiveContainer width="100%" height={200}>
                      <PieChart>
                        <Pie
                          data={privacyData}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={80}
                          paddingAngle={5}
                          dataKey="value"
                        >
                          {privacyData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <RechartsTooltip />
                      </PieChart>
                    </ResponsiveContainer>
                    <Typography variant="h5" sx={{ fontWeight: 700, mt: 2, fontFamily: 'Montserrat, sans-serif', color: '#000' }}>
                      ε = {session.epsilon_used.toFixed(3)} / {session.epsilon_budget}
                    </Typography>
                    <Typography variant="body2" color="textSecondary" sx={{ fontFamily: 'Montserrat, sans-serif' }}>
                      δ = {session.delta.toExponential(1)}
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={(session.epsilon_used / session.epsilon_budget) * 100}
                      sx={{ 
                        width: '100%', 
                        mt: 2, 
                        height: 8, 
                        borderRadius: 4,
                        backgroundColor: '#e0e0e0',
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: '#000',
                        }
                      }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12}>
              <Card sx={{ borderRadius: '12px', boxShadow: '0px 4px 12px rgba(0,0,0,0.08)', border: '2px solid #000' }}>
                <CardContent>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 2, fontFamily: 'Montserrat, sans-serif' }}>
                    📉 Loss Over Rounds
                  </Typography>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={lossHistory}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                      <XAxis dataKey="round" label={{ value: 'Round', position: 'insideBottom', offset: -5, style: { fontFamily: 'Montserrat, sans-serif' } }} tick={{ fontFamily: 'Montserrat, sans-serif' }} />
                      <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft', style: { fontFamily: 'Montserrat, sans-serif' } }} tick={{ fontFamily: 'Montserrat, sans-serif' }} />
                      <RechartsTooltip />
                      <Legend wrapperStyle={{ fontFamily: 'Montserrat, sans-serif' }} />
                      <Line
                        type="monotone"
                        dataKey="loss"
                        stroke="#666"
                        strokeWidth={3}
                        dot={{ r: 4, fill: '#666' }}
                        name="Loss"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Byzantine Alert */}
          {session.byzantine_detected > 0 && (
            <Alert severity="warning" sx={{ borderRadius: '12px' }}>
              ⚠️ <strong>Byzantine Attack Detected!</strong> {session.byzantine_detected} suspicious node(s) detected
              and quarantined using cosine similarity analysis.
            </Alert>
          )}
        </>
      )}

      {/* FL Configuration Panel */}
      {!session && (
        <FLConfigPanel onSave={(config) => console.log('FL Config:', config)} />
      )}

      {/* Federated Learning Nodes */}
      {session && session.status === 'training' && (
        <Box sx={{ mt: 3 }}>
          <FederatedNodes trainingId={session.training_id} autoRefresh={true} />
        </Box>
      )}

      {/* Model Convergence Chart */}
      {session && session.status === 'training' && (
        <Box sx={{ mt: 3 }}>
          <ModelConvergence trainingId={session.training_id} autoRefresh={true} />
        </Box>
      )}

      {/* Model Export & Reports */}
      {session && session.status === 'completed' && (
        <Box sx={{ mt: 3 }}>
          <ModelExport trainingId={session.training_id} />
        </Box>
      )}

      {/* Success Snackbar */}
      <Snackbar 
        open={showSuccess} 
        autoHideDuration={4000} 
        onClose={() => setShowSuccess(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={() => setShowSuccess(false)} severity="success" sx={{ width: '100%' }}>
          {success}
        </Alert>
      </Snackbar>

      {/* Error Snackbar */}
      <Snackbar 
        open={showError} 
        autoHideDuration={6000} 
        onClose={() => setShowError(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={() => setShowError(false)} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default TrainingDashboard;
