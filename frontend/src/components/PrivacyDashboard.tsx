import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
  Skeleton,
  Alert,
  Snackbar,
  Button,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Shield,
  Lock,
  TrendingDown,
  QueryStats,
  Refresh,
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
  RadialBarChart,
  RadialBar,
} from 'recharts';
import { authenticatedFetch, parseErrorResponse } from '../utils/api';

interface PrivacyMetrics {
  epsilon_budget: number;
  epsilon_used: number;
  epsilon_remaining: number;
  delta: number;
  noise_multiplier: number;
  clipping_threshold: number;
  privacy_level: string;
  queries_executed: number;
  privacy_accountant: {
    method: string;
    orders: number[];
    rdp_values: number[];
  };
}

const PrivacyDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<PrivacyMetrics | null>(null);
  const [budgetHistory, setBudgetHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [showError, setShowError] = useState(false);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    fetchPrivacyMetrics();
    fetchBudgetHistory();
    
    // Online/Offline detection
    const handleOnline = () => {
      setIsOnline(true);
      setError('Connection restored! Refreshing data...');
      setShowError(true);
      fetchPrivacyMetrics();
      fetchBudgetHistory();
    };
    const handleOffline = () => {
      setIsOnline(false);
      setError('No internet connection. Please check your network.');
      setShowError(true);
    };
    
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    await Promise.all([fetchPrivacyMetrics(), fetchBudgetHistory()]);
    setRefreshing(false);
  };

  const fetchPrivacyMetrics = async () => {
    try {
      const response = await authenticatedFetch('/api/privacy/metrics', {
        timeout: 15000, // 15 second timeout
      });
      
      if (!response.ok) {
        const errorMsg = await parseErrorResponse(response);
        throw new Error(errorMsg);
      }
      
      const data = await response.json();
      setMetrics(data);
      setError('');
    } catch (error: any) {
      console.error('Failed to fetch privacy metrics:', error);
      setError(error.message || 'Failed to load privacy metrics. Please try again.');
      setShowError(true);
    } finally {
      setLoading(false);
    }
  };

  const fetchBudgetHistory = async () => {
    try {
      const response = await authenticatedFetch('/api/privacy/budget-history', {
        timeout: 15000,
      });
      
      if (!response.ok) {
        const errorMsg = await parseErrorResponse(response);
        throw new Error(errorMsg);
      }
      
      const data = await response.json();
      setBudgetHistory(
        data.history.map((item: any) => ({
          time: new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          epsilon: item.epsilon_used,
          queries: item.queries_executed,
        }))
      );
    } catch (error: any) {
      console.error('Failed to fetch budget history:', error);
      setError(error.message || 'Failed to load budget history. Please try again.');
      setShowError(true);
    }
  };

  const getPrivacyLevelColor = (level?: string) => {
    if (!level) return '#000000';
    switch (level.toLowerCase()) {
      case 'high':
        return '#000000'; // Black
      case 'medium':
        return '#666666'; // Dark gray
      case 'low':
        return '#333333'; // Darker gray
      default:
        return '#000000';
    }
  };

  const radialData = metrics
    ? [
        {
          name: 'Budget',
          value: (metrics.epsilon_used / metrics.epsilon_budget) * 100,
          fill: '#000000', // Black for all levels
        },
      ]
    : [];

  if (loading) {
    return (
      <Box sx={{ p: 3 }}>
        <Skeleton variant="rectangular" height={60} sx={{ mb: 3, borderRadius: '12px' }} />
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} md={3}>
            <Skeleton variant="rectangular" height={120} sx={{ borderRadius: '12px' }} />
          </Grid>
          <Grid item xs={12} md={3}>
            <Skeleton variant="rectangular" height={120} sx={{ borderRadius: '12px' }} />
          </Grid>
          <Grid item xs={12} md={3}>
            <Skeleton variant="rectangular" height={120} sx={{ borderRadius: '12px' }} />
          </Grid>
          <Grid item xs={12} md={3}>
            <Skeleton variant="rectangular" height={120} sx={{ borderRadius: '12px' }} />
          </Grid>
        </Grid>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Skeleton variant="rectangular" height={300} sx={{ borderRadius: '12px' }} />
          </Grid>
          <Grid item xs={12} md={8}>
            <Skeleton variant="rectangular" height={300} sx={{ borderRadius: '12px' }} />
          </Grid>
        </Grid>
      </Box>
    );
  }

  if (!metrics) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert 
          severity="warning" 
          action={
            <Button 
              color="inherit" 
              size="small" 
              onClick={() => {
                setLoading(true);
                fetchPrivacyMetrics();
                fetchBudgetHistory();
              }}
            >
              Retry
            </Button>
          }
        >
          No privacy metrics available. Please start a training session first or click Retry.
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header with Offline Indicator and Refresh */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 600, color: '#000', fontFamily: 'Montserrat, sans-serif' }}>
          🔒 Differential Privacy Metrics
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          {!isOnline && (
            <Chip 
              icon={<CloudOff />} 
              label="Offline" 
              color="error" 
              size="small" 
              sx={{ fontFamily: 'Montserrat, sans-serif' }}
            />
          )}
          <Tooltip title="Refresh data">
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
        </Box>
      </Box>

      {/* Privacy Level Banner */}
      <Card
        sx={{
          mb: 3,
          borderRadius: '12px',
          background: '#000000',
          color: 'white',
          boxShadow: '0px 8px 24px rgba(0,0,0,0.15)',
          border: '2px solid #000',
        }}
      >
        <CardContent sx={{ py: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Shield sx={{ fontSize: 64 }} />
              <Box>
                <Typography variant="h3" sx={{ fontWeight: 700, fontFamily: 'Montserrat, sans-serif' }}>
                  {metrics?.privacy_level || 'UNKNOWN'} Privacy
                </Typography>
                <Typography variant="body1" sx={{ opacity: 0.9, mt: 1, fontFamily: 'Montserrat, sans-serif' }}>
                  Your data is protected with differential privacy guarantees
                </Typography>
              </Box>
            </Box>
            <Chip
              label={`${metrics?.queries_executed || 0} Queries`}
              sx={{
                backgroundColor: '#ffffff',
                color: '#000',
                fontWeight: 600,
                fontSize: '1rem',
                px: 2,
                fontFamily: 'Montserrat, sans-serif',
              }}
            />
          </Box>
        </CardContent>
      </Card>

      {/* Key Metrics Cards */}
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
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="body2" sx={{ fontWeight: 600, fontFamily: 'Montserrat, sans-serif' }}>
                  Epsilon Budget
                </Typography>
                <Lock sx={{ color: '#000', fontSize: 32 }} />
              </Box>
              <Typography variant="h4" sx={{ fontWeight: 700, color: '#000', fontFamily: 'Montserrat, sans-serif' }}>
                {metrics?.epsilon_budget || 0}
              </Typography>
              <Typography variant="body2" color="textSecondary" sx={{ mt: 1, fontFamily: 'Montserrat, sans-serif' }}>
                Total allocated budget
              </Typography>
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
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="body2" sx={{ fontWeight: 600, fontFamily: 'Montserrat, sans-serif' }}>
                  Epsilon Used
                </Typography>
                <TrendingDown sx={{ color: '#000', fontSize: 32 }} />
              </Box>
              <Typography variant="h4" sx={{ fontWeight: 700, color: '#000', fontFamily: 'Montserrat, sans-serif' }}>
                {metrics?.epsilon_used ? metrics.epsilon_used.toFixed(3) : '0.000'}
              </Typography>
              <Typography variant="body2" color="textSecondary" sx={{ mt: 1, fontFamily: 'Montserrat, sans-serif' }}>
                {metrics?.epsilon_used && metrics?.epsilon_budget ? ((metrics.epsilon_used / metrics.epsilon_budget) * 100).toFixed(1) : '0.0'}% consumed
              </Typography>
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
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="body2" sx={{ fontWeight: 600, fontFamily: 'Montserrat, sans-serif' }}>
                  Epsilon Remaining
                </Typography>
                <Shield sx={{ color: '#000', fontSize: 32 }} />
              </Box>
              <Typography variant="h4" sx={{ fontWeight: 700, color: '#000', fontFamily: 'Montserrat, sans-serif' }}>
                {metrics?.epsilon_remaining ? metrics.epsilon_remaining.toFixed(3) : '0.000'}
              </Typography>
              <Typography variant="body2" color="textSecondary" sx={{ mt: 1, fontFamily: 'Montserrat, sans-serif' }}>
                Available for queries
              </Typography>
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
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="body2" sx={{ fontWeight: 600, fontFamily: 'Montserrat, sans-serif' }}>
                  Queries
                </Typography>
                <QueryStats sx={{ color: '#000', fontSize: 32 }} />
              </Box>
              <Typography variant="h4" sx={{ fontWeight: 700, color: '#000', fontFamily: 'Montserrat, sans-serif' }}>
                {metrics?.queries_executed || 0}
              </Typography>
              <Typography variant="body2" color="textSecondary" sx={{ mt: 1, fontFamily: 'Montserrat, sans-serif' }}>
                Total queries executed
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Budget Visualization and Parameters */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Card sx={{ borderRadius: '12px', boxShadow: '0px 4px 12px rgba(0,0,0,0.08)', height: '100%', border: '2px solid #000' }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 3, fontFamily: 'Montserrat, sans-serif' }}>
                📊 Budget Consumption
              </Typography>
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 250 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <RadialBarChart
                    cx="50%"
                    cy="50%"
                    innerRadius="70%"
                    outerRadius="100%"
                    data={radialData}
                    startAngle={180}
                    endAngle={0}
                  >
                    <RadialBar
                      background
                      dataKey="value"
                    />
                  </RadialBarChart>
                </ResponsiveContainer>
              </Box>
              <Box sx={{ textAlign: 'center', mt: 2 }}>
                <Typography variant="h3" sx={{ fontWeight: 700, color: '#000', fontFamily: 'Montserrat, sans-serif' }}>
                  {metrics?.epsilon_budget && metrics?.epsilon_used 
                    ? ((metrics.epsilon_used / metrics.epsilon_budget) * 100).toFixed(1) 
                    : '0.0'}%
                </Typography>
                <Typography variant="body2" color="textSecondary" sx={{ fontFamily: 'Montserrat, sans-serif' }}>
                  Budget Consumed
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={metrics?.epsilon_budget && metrics?.epsilon_used 
                  ? (metrics.epsilon_used / metrics.epsilon_budget) * 100 
                  : 0}
                sx={{ 
                  mt: 2, 
                  height: 10, 
                  borderRadius: 5,
                  backgroundColor: '#e0e0e0',
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: '#000',
                  }
                }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          <Card sx={{ borderRadius: '12px', boxShadow: '0px 4px 12px rgba(0,0,0,0.08)', height: '100%', border: '2px solid #000' }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2, fontFamily: 'Montserrat, sans-serif' }}>
                📈 Budget Consumption History (24 Hours)
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={budgetHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis dataKey="time" tick={{ fontFamily: 'Montserrat, sans-serif' }} />
                  <YAxis yAxisId="left" label={{ value: 'Epsilon', angle: -90, position: 'insideLeft', style: { fontFamily: 'Montserrat, sans-serif' } }} tick={{ fontFamily: 'Montserrat, sans-serif' }} />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    label={{ value: 'Queries', angle: 90, position: 'insideRight', style: { fontFamily: 'Montserrat, sans-serif' } }}
                    tick={{ fontFamily: 'Montserrat, sans-serif' }}
                  />
                  <RechartsTooltip />
                  <Legend wrapperStyle={{ fontFamily: 'Montserrat, sans-serif' }} />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="epsilon"
                    stroke="#000"
                    strokeWidth={3}
                    dot={{ r: 3, fill: '#000' }}
                    name="Epsilon Used"
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="queries"
                    stroke="#666"
                    strokeWidth={3}
                    dot={{ r: 3, fill: '#666' }}
                    name="Queries"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Privacy Parameters */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Card sx={{ borderRadius: '12px', boxShadow: '0px 4px 12px rgba(0,0,0,0.08)', border: '2px solid #000' }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 3, fontFamily: 'Montserrat, sans-serif' }}>
                🔧 Privacy Parameters
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: '8px', border: '1px solid #e0e0e0' }}>
                    <Typography variant="body2" color="textSecondary" sx={{ fontFamily: 'Montserrat, sans-serif' }}>
                      Delta (δ)
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600, mt: 1, fontFamily: 'Montserrat, sans-serif', color: '#000' }}>
                      {metrics?.delta ? metrics.delta.toExponential(1) : '0.0e+0'}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: '8px', border: '1px solid #e0e0e0' }}>
                    <Typography variant="body2" color="textSecondary" sx={{ fontFamily: 'Montserrat, sans-serif' }}>
                      Noise Multiplier
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600, mt: 1, fontFamily: 'Montserrat, sans-serif', color: '#000' }}>
                      {metrics?.noise_multiplier || '0.0'}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: '8px', border: '1px solid #e0e0e0' }}>
                    <Typography variant="body2" color="textSecondary" sx={{ fontFamily: 'Montserrat, sans-serif' }}>
                      Clipping Threshold
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600, mt: 1, fontFamily: 'Montserrat, sans-serif', color: '#000' }}>
                      {metrics?.clipping_threshold || '0.0'}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: '8px', border: '1px solid #e0e0e0' }}>
                    <Typography variant="body2" color="textSecondary" sx={{ fontFamily: 'Montserrat, sans-serif' }}>
                      Privacy Method
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 600, mt: 1, fontFamily: 'Montserrat, sans-serif', color: '#000' }}>
                      {metrics?.privacy_accountant?.method || 'N/A'}
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card sx={{ borderRadius: '12px', boxShadow: '0px 4px 12px rgba(0,0,0,0.08)', border: '2px solid #000' }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 3, fontFamily: 'Montserrat, sans-serif' }}>
                🧮 RDP Accountant (Rényi Differential Privacy)
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ fontWeight: 600, fontFamily: 'Montserrat, sans-serif', color: '#000' }}>Order (α)</TableCell>
                      <TableCell sx={{ fontWeight: 600, fontFamily: 'Montserrat, sans-serif', color: '#000' }}>RDP Value</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {metrics?.privacy_accountant?.orders?.map((order, index) => (
                      <TableRow key={order}>
                        <TableCell sx={{ fontFamily: 'Montserrat, sans-serif' }}>{order}</TableCell>
                        <TableCell sx={{ fontFamily: 'Montserrat, sans-serif' }}>
                          {metrics.privacy_accountant.rdp_values[index]?.toFixed(3) || '0.000'}
                        </TableCell>
                      </TableRow>
                    )) || (
                      <TableRow>
                        <TableCell colSpan={2} sx={{ textAlign: 'center', fontFamily: 'Montserrat, sans-serif' }}>
                          No RDP data available
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

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

export default PrivacyDashboard;
