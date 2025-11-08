import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  LinearProgress,
  Chip,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Switch,
  FormControlLabel,
  Slider,
  Divider,
} from '@mui/material';
import {
  ModelTraining,
  Security,
  Speed,
  Devices,
  CloudSync,
  PrivacyTip,
  PlayArrow,
  Pause,
  Stop,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

// Mock data for QFLARE federated learning demonstration
const trainingProgress = [
  { round: 1, accuracy: 0.72, loss: 0.89, participants: 12 },
  { round: 2, accuracy: 0.78, loss: 0.76, participants: 15 },
  { round: 3, accuracy: 0.82, loss: 0.68, participants: 18 },
  { round: 4, accuracy: 0.85, loss: 0.61, participants: 20 },
  { round: 5, accuracy: 0.87, loss: 0.55, participants: 22 },
  { round: 6, accuracy: 0.89, loss: 0.49, participants: 24 },
];

const edgeNodes = [
  { id: 'edge-001', name: 'Healthcare Node', status: 'active', accuracy: 0.91, samples: 2847, location: 'Hospital Network' },
  { id: 'edge-002', name: 'Financial Node', status: 'active', accuracy: 0.88, samples: 1923, location: 'Bank Datacenter' },
  { id: 'edge-003', name: 'Research Node', status: 'training', accuracy: 0.85, samples: 3421, location: 'University Lab' },
  { id: 'edge-004', name: 'IoT Node', status: 'idle', accuracy: 0.79, samples: 1256, location: 'Smart City' },
];

const securityMetrics = [
  { metric: 'PQC Encryption', value: 100, fullMark: 100 },
  { metric: 'Differential Privacy', value: 95, fullMark: 100 },
  { metric: 'Byzantine Tolerance', value: 88, fullMark: 100 },
  { metric: 'Secure Aggregation', value: 98, fullMark: 100 },
  { metric: 'Edge Security', value: 92, fullMark: 100 },
];

const FederatedLearning: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [currentRound, setCurrentRound] = useState(6);
  const [privacyBudget, setPrivacyBudget] = useState(1.0);
  const [aggregationMethod, setAggregationMethod] = useState('fedavg');

  const handleStartTraining = () => {
    setIsTraining(true);
    // Simulate training progress
    setTimeout(() => setIsTraining(false), 5000);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success';
      case 'training': return 'warning';
      case 'idle': return 'default';
      default: return 'error';
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 'bold', mb: 1 }}>
          🔮 QFLARE Federated Learning
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Quantum-Safe Distributed Machine Learning with Privacy Preservation
        </Typography>
      </Box>

      {/* Quick Stats */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <ModelTraining sx={{ mr: 1 }} />
                <Typography variant="h6">Global Model</Typography>
              </Box>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>89.2%</Typography>
              <Typography variant="body2">Accuracy</Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Devices sx={{ mr: 1 }} />
                <Typography variant="h6">Active Nodes</Typography>
              </Box>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>24</Typography>
              <Typography variant="body2">Edge Participants</Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Security sx={{ mr: 1 }} />
                <Typography variant="h6">Security Level</Typography>
              </Box>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>99.8%</Typography>
              <Typography variant="body2">PQC Protection</Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <PrivacyTip sx={{ mr: 1 }} />
                <Typography variant="h6">Privacy Budget</Typography>
              </Box>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>{privacyBudget}</Typography>
              <Typography variant="body2">ε-Differential Privacy</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Training Control Panel */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 3, display: 'flex', alignItems: 'center' }}>
                <ModelTraining sx={{ mr: 1, color: 'primary.main' }} />
                Training Control
              </Typography>

              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>Training Round: {currentRound}</Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={(currentRound / 10) * 100} 
                  sx={{ height: 8, borderRadius: 4, mb: 2 }}
                />
                <Typography variant="body2" color="text.secondary">
                  {isTraining ? 'Training in progress...' : 'Ready for next round'}
                </Typography>
              </Box>

              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" sx={{ mb: 2 }}>Privacy Budget (ε)</Typography>
                <Slider
                  value={privacyBudget}
                  onChange={(_, value) => setPrivacyBudget(value as number)}
                  min={0.1}
                  max={2.0}
                  step={0.1}
                  marks={[
                    { value: 0.1, label: '0.1' },
                    { value: 1.0, label: '1.0' },
                    { value: 2.0, label: '2.0' }
                  ]}
                  valueLabelDisplay="on"
                />
              </Box>

              <Box sx={{ mb: 3 }}>
                <FormControlLabel
                  control={<Switch checked={aggregationMethod === 'secure'} />}
                  label="Secure Aggregation"
                  onChange={() => setAggregationMethod(
                    aggregationMethod === 'secure' ? 'fedavg' : 'secure'
                  )}
                />
              </Box>

              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                <Button
                  variant="contained"
                  startIcon={<PlayArrow />}
                  onClick={handleStartTraining}
                  disabled={isTraining}
                  size="small"
                >
                  Start Round
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<Pause />}
                  disabled={!isTraining}
                  size="small"
                >
                  Pause
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<Stop />}
                  color="error"
                  size="small"
                >
                  Stop
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Training Progress Chart */}
        <Grid item xs={12} md={8}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 3 }}>
                Training Progress & Model Performance
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={trainingProgress}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="round" />
                  <YAxis yAxisId="left" domain={[0.7, 1]} />
                  <YAxis yAxisId="right" orientation="right" domain={[0, 1]} />
                  <Tooltip />
                  <Line 
                    yAxisId="left" 
                    type="monotone" 
                    dataKey="accuracy" 
                    stroke="#8884d8" 
                    strokeWidth={3}
                    name="Accuracy"
                  />
                  <Line 
                    yAxisId="right" 
                    type="monotone" 
                    dataKey="loss" 
                    stroke="#82ca9d" 
                    strokeWidth={3}
                    name="Loss"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Edge Nodes Status */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 3 }}>
                Edge Node Status & Performance
              </Typography>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Node</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Accuracy</TableCell>
                      <TableCell>Samples</TableCell>
                      <TableCell>Location</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {edgeNodes.map((node) => (
                      <TableRow key={node.id}>
                        <TableCell>
                          <Box>
                            <Typography variant="subtitle2">{node.name}</Typography>
                            <Typography variant="caption" color="text.secondary">
                              {node.id}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Chip 
                            label={node.status} 
                            color={getStatusColor(node.status) as any}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                            {(node.accuracy * 100).toFixed(1)}%
                          </Typography>
                        </TableCell>
                        <TableCell>{node.samples.toLocaleString()}</TableCell>
                        <TableCell>
                          <Typography variant="body2" color="text.secondary">
                            {node.location}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Security Radar Chart */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 3 }}>
                Security & Privacy Metrics
              </Typography>
              <ResponsiveContainer width="100%" height={250}>
                <RadarChart data={securityMetrics}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="metric" fontSize={10} />
                  <PolarRadiusAxis domain={[0, 100]} fontSize={10} />
                  <Radar
                    name="Security Score"
                    dataKey="value"
                    stroke="#8884d8"
                    fill="#8884d8"
                    fillOpacity={0.3}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* QFLARE Features Alert */}
      <Alert severity="info" sx={{ mt: 3 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>
          🔮 QFLARE Quantum-Safe Features Active:
        </Typography>
        <Typography variant="body2">
          • CRYSTALS-Kyber Post-Quantum Encryption | • Differential Privacy (ε = {privacyBudget}) | 
          • Byzantine Fault Tolerance | • Secure Multi-Party Computation | • Hardware Security Modules (HSM)
        </Typography>
      </Alert>
    </Box>
  );
};

export default FederatedLearning;