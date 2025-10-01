import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Container,
  Grid,
  Card,
  CardContent,
  Button,
  Chip,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Alert,
  CircularProgress,
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Refresh,
  Computer,
  CloudUpload,
  Analytics,
  Security,
} from '@mui/icons-material';

interface FLStatus {
  available: boolean;
  current_round: number;
  total_rounds: number;
  status: string;
  registered_devices: number;
  active_devices: number;
  participants_this_round: number;
  round_start_time: string | null;
  training_history: Array<any>;
}

interface TrainingHistory {
  round: number;
  timestamp: string;
  participants: number;
  avg_loss: number;
  total_samples: number;
}

const FederatedLearningPage: React.FC = () => {
  const [flStatus, setFlStatus] = useState<FLStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [startRoundDialog, setStartRoundDialog] = useState(false);
  const [roundConfig, setRoundConfig] = useState({
    target_participants: 3,
    local_epochs: 5,
    learning_rate: 0.01,
  });

  // Fetch FL status
  const fetchFLStatus = async () => {
    try {
      const response = await fetch('/api/fl/status');
      const data = await response.json();
      
      if (data.success) {
        setFlStatus(data.fl_status);
        setError(null);
      } else {
        setError('Failed to fetch FL status');
      }
    } catch (err) {
      setError('Failed to connect to FL service');
    } finally {
      setLoading(false);
    }
  };

  // Start training round
  const startTrainingRound = async () => {
    try {
      const formData = new FormData();
      formData.append('target_participants', roundConfig.target_participants.toString());
      formData.append('local_epochs', roundConfig.local_epochs.toString());
      formData.append('learning_rate', roundConfig.learning_rate.toString());

      const response = await fetch('/api/fl/start_round', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      
      if (data.success) {
        setStartRoundDialog(false);
        fetchFLStatus(); // Refresh status
      } else {
        setError(data.detail || 'Failed to start training round');
      }
    } catch (err) {
      setError('Failed to start training round');
    }
  };

  // Reset FL system
  const resetFLSystem = async () => {
    if (!window.confirm('Are you sure you want to reset the FL system?')) {
      return;
    }

    try {
      const response = await fetch('/api/fl/reset', { method: 'POST' });
      const data = await response.json();
      
      if (data.success) {
        fetchFLStatus(); // Refresh status
      } else {
        setError('Failed to reset FL system');
      }
    } catch (err) {
      setError('Failed to reset FL system');
    }
  };

  useEffect(() => {
    fetchFLStatus();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchFLStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'idle': return 'success';
      case 'training': return 'warning';
      case 'aggregating': return 'info';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'idle': return <Stop />;
      case 'training': return <PlayArrow />;
      case 'aggregating': return <CloudUpload />;
      default: return <Computer />;
    }
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
          <CircularProgress size={60} />
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom>
        🤖 Federated Learning Dashboard
      </Typography>
      
      <Typography variant="subtitle1" color="text.secondary" gutterBottom>
        Manage and monitor federated learning training rounds
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {!flStatus?.available && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          Federated Learning service is not available. Please check the server configuration.
        </Alert>
      )}

      {flStatus && (
        <>
          {/* Status Overview */}
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid item xs={12} md={6} lg={3}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={2}>
                    {getStatusIcon(flStatus.status)}
                    <Typography variant="h6" sx={{ ml: 1 }}>
                      System Status
                    </Typography>
                  </Box>
                  <Chip 
                    label={flStatus.status.toUpperCase()} 
                    color={getStatusColor(flStatus.status) as any}
                    size="large"
                  />
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6} lg={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Current Round
                  </Typography>
                  <Typography variant="h3" color="primary">
                    {flStatus.current_round}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    of {flStatus.total_rounds} total rounds
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6} lg={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Active Devices
                  </Typography>
                  <Typography variant="h3" color="success.main">
                    {flStatus.active_devices}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    of {flStatus.registered_devices} registered
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6} lg={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Participants
                  </Typography>
                  <Typography variant="h3" color="info.main">
                    {flStatus.participants_this_round}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    this round
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Control Panel */}
          <Card sx={{ mb: 4 }}>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                🎛️ Control Panel
              </Typography>
              
              <Box display="flex" gap={2} flexWrap="wrap">
                <Button
                  variant="contained"
                  startIcon={<PlayArrow />}
                  onClick={() => setStartRoundDialog(true)}
                  disabled={flStatus.status !== 'idle' || flStatus.active_devices < 2}
                >
                  Start Training Round
                </Button>
                
                <Button
                  variant="outlined"
                  startIcon={<Refresh />}
                  onClick={fetchFLStatus}
                >
                  Refresh Status
                </Button>
                
                <Button
                  variant="outlined"
                  startIcon={<Analytics />}
                  href="/api/fl/training_history"
                  target="_blank"
                >
                  Export History
                </Button>
                
                <Button
                  variant="outlined"
                  color="error"
                  onClick={resetFLSystem}
                >
                  Reset System
                </Button>
              </Box>

              {flStatus.status !== 'idle' && flStatus.round_start_time && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Round started at: {flStatus.round_start_time}
                  </Typography>
                  <LinearProgress sx={{ mt: 1 }} />
                </Box>
              )}
            </CardContent>
          </Card>

          {/* Training History */}
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                📊 Training History
              </Typography>
              
              {flStatus.training_history.length === 0 ? (
                <Typography color="text.secondary">
                  No training history yet. Start your first round!
                </Typography>
              ) : (
                <TableContainer component={Paper} sx={{ mt: 2 }}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Round</TableCell>
                        <TableCell>Timestamp</TableCell>
                        <TableCell align="right">Participants</TableCell>
                        <TableCell align="right">Avg Loss</TableCell>
                        <TableCell align="right">Total Samples</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {flStatus.training_history.slice(-10).reverse().map((history: TrainingHistory) => (
                        <TableRow key={history.round}>
                          <TableCell component="th" scope="row">
                            {history.round}
                          </TableCell>
                          <TableCell>{history.timestamp}</TableCell>
                          <TableCell align="right">{history.participants}</TableCell>
                          <TableCell align="right">{history.avg_loss?.toFixed(4) || 'N/A'}</TableCell>
                          <TableCell align="right">{history.total_samples}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </>
      )}

      {/* Start Round Dialog */}
      <Dialog open={startRoundDialog} onClose={() => setStartRoundDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Start New Training Round</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 1 }}>
            <TextField
              label="Target Participants"
              type="number"
              fullWidth
              margin="normal"
              value={roundConfig.target_participants}
              onChange={(e) => setRoundConfig({
                ...roundConfig,
                target_participants: parseInt(e.target.value) || 3
              })}
              inputProps={{ min: 2, max: flStatus?.active_devices || 10 }}
            />
            
            <TextField
              label="Local Epochs"
              type="number"
              fullWidth
              margin="normal"
              value={roundConfig.local_epochs}
              onChange={(e) => setRoundConfig({
                ...roundConfig,
                local_epochs: parseInt(e.target.value) || 5
              })}
              inputProps={{ min: 1, max: 20 }}
            />
            
            <TextField
              label="Learning Rate"
              type="number"
              fullWidth
              margin="normal"
              value={roundConfig.learning_rate}
              onChange={(e) => setRoundConfig({
                ...roundConfig,
                learning_rate: parseFloat(e.target.value) || 0.01
              })}
              inputProps={{ min: 0.001, max: 1, step: 0.001 }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setStartRoundDialog(false)}>Cancel</Button>
          <Button onClick={startTrainingRound} variant="contained">
            Start Round
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default FederatedLearningPage;