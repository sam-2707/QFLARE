/**
 * Training Comparison Component
 * Compare multiple training sessions side-by-side
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  CompareArrows,
  Delete,
  Add,
  TrendingUp,
  TrendingDown,
  Remove,
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
  BarChart,
  Bar,
} from 'recharts';
import { authenticatedFetch } from '../utils/api';

interface TrainingComparison {
  training_id: string;
  model_type: string;
  aggregation_method: string;
  num_rounds: number;
  final_accuracy: number;
  final_loss: number;
  epsilon_used: number;
  training_time_minutes: number;
  nodes_participated: number;
  byzantine_detected: number;
  convergence_rate: number;
}

const TrainingComparison: React.FC = () => {
  const [availableSessions, setAvailableSessions] = useState<string[]>([]);
  const [selectedSessions, setSelectedSessions] = useState<string[]>([]);
  const [comparisonData, setComparisonData] = useState<TrainingComparison[]>([]);
  const [convergenceData, setConvergenceData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAvailableSessions();
  }, []);

  useEffect(() => {
    if (selectedSessions.length > 0) {
      fetchComparisonData();
    }
  }, [selectedSessions]);

  const fetchAvailableSessions = async () => {
    try {
      const response = await authenticatedFetch('/api/training/sessions', { timeout: 10000 });
      if (response.ok) {
        const data = await response.json();
        setAvailableSessions(data.sessions || []);
      }
    } catch (error) {
      console.error('Failed to fetch sessions:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchComparisonData = async () => {
    try {
      const promises = selectedSessions.map((id) =>
        authenticatedFetch(`/api/training/${id}/summary`, { timeout: 10000 })
      );
      
      const responses = await Promise.all(promises);
      const data = await Promise.all(responses.map((r) => r.json()));
      
      setComparisonData(data);
      
      // Fetch convergence data for comparison chart
      const convergencePromises = selectedSessions.map((id) =>
        authenticatedFetch(`/api/training/${id}/convergence`, { timeout: 10000 })
      );
      
      const convergenceResponses = await Promise.all(convergencePromises);
      const convergenceResults = await Promise.all(convergenceResponses.map((r) => r.json()));
      
      // Transform for comparison chart
      const maxRounds = Math.max(...convergenceResults.map((c) => c.history.length));
      const chartData = Array.from({ length: maxRounds }, (_, i) => {
        const point: any = { round: i + 1 };
        convergenceResults.forEach((result, idx) => {
          const roundData = result.history[i];
          if (roundData) {
            point[`session_${idx + 1}`] = roundData.global_accuracy * 100;
          }
        });
        return point;
      });
      
      setConvergenceData(chartData);
    } catch (error) {
      console.error('Failed to fetch comparison data:', error);
    }
  };

  const handleAddSession = (sessionId: string) => {
    if (!selectedSessions.includes(sessionId) && selectedSessions.length < 5) {
      setSelectedSessions([...selectedSessions, sessionId]);
    }
  };

  const handleRemoveSession = (sessionId: string) => {
    setSelectedSessions(selectedSessions.filter((id) => id !== sessionId));
  };

  const getWinner = (metric: keyof TrainingComparison, higherIsBetter: boolean = true) => {
    if (comparisonData.length === 0) return null;
    
    const values = comparisonData.map((d) => d[metric] as number);
    const bestValue = higherIsBetter ? Math.max(...values) : Math.min(...values);
    const winnerIndex = values.indexOf(bestValue);
    
    return comparisonData[winnerIndex].training_id;
  };

  const colors = ['#4caf50', '#2196f3', '#ff9800', '#9c27b0', '#f44336'];

  return (
    <Box>
      <Typography variant="h5" sx={{ fontWeight: 600, fontFamily: 'Montserrat, sans-serif', mb: 3 }}>
        🔍 Training Session Comparison
      </Typography>

      {/* Session Selection */}
      <Card sx={{ borderRadius: '12px', mb: 3 }}>
        <CardContent>
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
            Select Sessions to Compare (max 5)
          </Typography>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={8}>
              <FormControl fullWidth size="small">
                <InputLabel>Add Training Session</InputLabel>
                <Select
                  label="Add Training Session"
                  onChange={(e) => handleAddSession(e.target.value)}
                  value=""
                  disabled={selectedSessions.length >= 5}
                >
                  {availableSessions
                    .filter((id) => !selectedSessions.includes(id))
                    .map((id) => (
                      <MenuItem key={id} value={id}>
                        Training Session {id}
                      </MenuItem>
                    ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={4}>
              <Typography variant="caption" color="textSecondary">
                {selectedSessions.length}/5 sessions selected
              </Typography>
            </Grid>
          </Grid>

          {/* Selected Sessions */}
          {selectedSessions.length > 0 && (
            <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {selectedSessions.map((id, index) => (
                <Chip
                  key={id}
                  label={`Session ${id}`}
                  onDelete={() => handleRemoveSession(id)}
                  sx={{
                    bgcolor: colors[index],
                    color: '#fff',
                    '& .MuiChip-deleteIcon': { color: '#fff' },
                  }}
                />
              ))}
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Comparison Table */}
      {comparisonData.length > 0 && (
        <>
          <Card sx={{ borderRadius: '12px', mb: 3 }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                📊 Performance Metrics
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ fontWeight: 600 }}>Metric</TableCell>
                      {comparisonData.map((session, idx) => (
                        <TableCell key={session.training_id} align="center" sx={{ fontWeight: 600 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, justifyContent: 'center' }}>
                            <Box
                              sx={{
                                width: 12,
                                height: 12,
                                borderRadius: '50%',
                                bgcolor: colors[idx],
                              }}
                            />
                            Session {idx + 1}
                          </Box>
                        </TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell>Final Accuracy</TableCell>
                      {comparisonData.map((session) => (
                        <TableCell key={session.training_id} align="center">
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, justifyContent: 'center' }}>
                            {(session.final_accuracy * 100).toFixed(2)}%
                            {session.training_id === getWinner('final_accuracy', true) && (
                              <Chip label="Best" size="small" color="success" />
                            )}
                          </Box>
                        </TableCell>
                      ))}
                    </TableRow>
                    <TableRow>
                      <TableCell>Final Loss</TableCell>
                      {comparisonData.map((session) => (
                        <TableCell key={session.training_id} align="center">
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, justifyContent: 'center' }}>
                            {session.final_loss.toFixed(4)}
                            {session.training_id === getWinner('final_loss', false) && (
                              <Chip label="Best" size="small" color="success" />
                            )}
                          </Box>
                        </TableCell>
                      ))}
                    </TableRow>
                    <TableRow>
                      <TableCell>Training Time</TableCell>
                      {comparisonData.map((session) => (
                        <TableCell key={session.training_id} align="center">
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, justifyContent: 'center' }}>
                            {session.training_time_minutes} min
                            {session.training_id === getWinner('training_time_minutes', false) && (
                              <Chip label="Fastest" size="small" color="success" />
                            )}
                          </Box>
                        </TableCell>
                      ))}
                    </TableRow>
                    <TableRow>
                      <TableCell>Model Type</TableCell>
                      {comparisonData.map((session) => (
                        <TableCell key={session.training_id} align="center">
                          {session.model_type}
                        </TableCell>
                      ))}
                    </TableRow>
                    <TableRow>
                      <TableCell>Aggregation Method</TableCell>
                      {comparisonData.map((session) => (
                        <TableCell key={session.training_id} align="center">
                          {session.aggregation_method}
                        </TableCell>
                      ))}
                    </TableRow>
                    <TableRow>
                      <TableCell>Rounds</TableCell>
                      {comparisonData.map((session) => (
                        <TableCell key={session.training_id} align="center">
                          {session.num_rounds}
                        </TableCell>
                      ))}
                    </TableRow>
                    <TableRow>
                      <TableCell>Nodes Participated</TableCell>
                      {comparisonData.map((session) => (
                        <TableCell key={session.training_id} align="center">
                          {session.nodes_participated}
                        </TableCell>
                      ))}
                    </TableRow>
                    <TableRow>
                      <TableCell>Epsilon Used</TableCell>
                      {comparisonData.map((session) => (
                        <TableCell key={session.training_id} align="center">
                          {session.epsilon_used.toFixed(2)}
                        </TableCell>
                      ))}
                    </TableRow>
                    <TableRow>
                      <TableCell>Byzantine Detected</TableCell>
                      {comparisonData.map((session) => (
                        <TableCell key={session.training_id} align="center">
                          {session.byzantine_detected}
                        </TableCell>
                      ))}
                    </TableRow>
                    <TableRow>
                      <TableCell>Convergence Rate</TableCell>
                      {comparisonData.map((session) => (
                        <TableCell key={session.training_id} align="center">
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, justifyContent: 'center' }}>
                            {session.convergence_rate.toFixed(2)}%
                            {session.training_id === getWinner('convergence_rate', true) && (
                              <Chip label="Best" size="small" color="success" />
                            )}
                          </Box>
                        </TableCell>
                      ))}
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>

          {/* Convergence Comparison Chart */}
          <Card sx={{ borderRadius: '12px', mb: 3 }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                📈 Convergence Comparison
              </Typography>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={convergenceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis
                    dataKey="round"
                    label={{ value: 'Training Round', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis
                    label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }}
                    domain={[0, 100]}
                  />
                  <RechartsTooltip
                    contentStyle={{
                      backgroundColor: '#fff',
                      border: '1px solid #e0e0e0',
                      borderRadius: '8px',
                    }}
                  />
                  <Legend />
                  {selectedSessions.map((_, idx) => (
                    <Line
                      key={idx}
                      type="monotone"
                      dataKey={`session_${idx + 1}`}
                      stroke={colors[idx]}
                      strokeWidth={2}
                      name={`Session ${idx + 1}`}
                      dot={{ r: 3 }}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Bar Chart Comparison */}
          <Card sx={{ borderRadius: '12px' }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                📊 Side-by-Side Comparison
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={comparisonData.map((session, idx) => ({
                    name: `Session ${idx + 1}`,
                    accuracy: session.final_accuracy * 100,
                    time: session.training_time_minutes,
                  }))}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <RechartsTooltip />
                  <Legend />
                  <Bar dataKey="accuracy" fill="#4caf50" name="Accuracy (%)" />
                  <Bar dataKey="time" fill="#2196f3" name="Time (min)" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </>
      )}

      {/* Empty State */}
      {selectedSessions.length === 0 && (
        <Card sx={{ borderRadius: '12px', textAlign: 'center', py: 6 }}>
          <CompareArrows sx={{ fontSize: 64, color: '#bdbdbd', mb: 2 }} />
          <Typography variant="h6" color="textSecondary" gutterBottom>
            No Sessions Selected
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Select 2 or more training sessions above to compare their performance
          </Typography>
        </Card>
      )}
    </Box>
  );
};

export default TrainingComparison;
