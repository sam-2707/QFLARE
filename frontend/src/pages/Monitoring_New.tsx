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
  Switch,
  FormControlLabel,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Monitor,
  Speed,
  Memory,
  Storage,
  NetworkCheck,
  Computer,
  Refresh,
  Timeline,
  TrendingUp,
  TrendingDown,
  Warning,
  CheckCircle,
  Error,
  Devices,
} from '@mui/icons-material';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  ResponsiveContainer, 
  AreaChart, 
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts';

// Mock monitoring data
const systemMetrics = [
  { time: '00:00', cpu: 45, memory: 62, network: 78, throughput: 234 },
  { time: '04:00', cpu: 52, memory: 58, network: 85, throughput: 189 },
  { time: '08:00', cpu: 38, memory: 71, network: 92, throughput: 267 },
  { time: '12:00', cpu: 67, memory: 84, network: 76, throughput: 298 },
  { time: '16:00', cpu: 73, memory: 79, network: 88, throughput: 312 },
  { time: '20:00', cpu: 41, memory: 65, network: 91, throughput: 289 },
];

const nodeStatus = [
  { id: 'node-001', name: 'Healthcare Edge', status: 'healthy', cpu: 34, memory: 67, uptime: '15d 4h', location: 'US-East' },
  { id: 'node-002', name: 'Finance Hub', status: 'healthy', cpu: 56, memory: 82, uptime: '22d 1h', location: 'EU-West' },
  { id: 'node-003', name: 'Research Lab', status: 'warning', cpu: 89, memory: 91, uptime: '8d 12h', location: 'Asia-Pacific' },
  { id: 'node-004', name: 'IoT Gateway', status: 'healthy', cpu: 23, memory: 45, uptime: '31d 2h', location: 'US-West' },
  { id: 'node-005', name: 'Mobile Edge', status: 'critical', cpu: 95, memory: 97, uptime: '2d 6h', location: 'EU-Central' },
];

const trainingMetrics = [
  { round: 1, accuracy: 0.72, participants: 12, avgTime: 4.2 },
  { round: 2, accuracy: 0.78, participants: 15, avgTime: 3.8 },
  { round: 3, accuracy: 0.82, participants: 18, avgTime: 4.1 },
  { round: 4, accuracy: 0.85, participants: 20, avgTime: 3.9 },
  { round: 5, accuracy: 0.87, participants: 22, avgTime: 4.3 },
  { round: 6, accuracy: 0.89, participants: 24, avgTime: 4.0 },
];

const resourceDistribution = [
  { name: 'CPU Usage', value: 67, color: '#8884d8' },
  { name: 'Memory Usage', value: 84, color: '#82ca9d' },
  { name: 'Network I/O', value: 76, color: '#ffc658' },
  { name: 'Storage I/O', value: 52, color: '#ff7c7c' },
];

const SystemMonitoring: React.FC = () => {
  const [realTimeMode, setRealTimeMode] = useState(true);
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h');
  const [lastUpdated, setLastUpdated] = useState(new Date());

  useEffect(() => {
    if (realTimeMode) {
      const interval = setInterval(() => {
        setLastUpdated(new Date());
      }, 30000); // Update every 30 seconds
      return () => clearInterval(interval);
    }
  }, [realTimeMode]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'warning': return 'warning';
      case 'critical': return 'error';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle color="success" />;
      case 'warning': return <Warning color="warning" />;
      case 'critical': return <Error color="error" />;
      default: return <Devices color="disabled" />;
    }
  };

  const formatUptime = (uptime: string) => {
    return uptime;
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 'bold', mb: 1 }}>
            📊 QFLARE System Monitoring
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Real-time Performance & Health Monitoring Dashboard
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <FormControlLabel
            control={
              <Switch 
                checked={realTimeMode} 
                onChange={(e) => setRealTimeMode(e.target.checked)}
              />
            }
            label="Real-time"
          />
          <Tooltip title="Refresh Data">
            <IconButton onClick={() => setLastUpdated(new Date())}>
              <Refresh />
            </IconButton>
          </Tooltip>
          <Typography variant="caption" color="text.secondary">
            Last updated: {lastUpdated.toLocaleTimeString()}
          </Typography>
        </Box>
      </Box>

      {/* System Overview Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Computer sx={{ mr: 1 }} />
                <Typography variant="h6">System Load</Typography>
              </Box>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>67%</Typography>
              <Typography variant="body2">Average CPU Usage</Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Memory sx={{ mr: 1 }} />
                <Typography variant="h6">Memory Usage</Typography>
              </Box>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>84%</Typography>
              <Typography variant="body2">System RAM</Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <NetworkCheck sx={{ mr: 1 }} />
                <Typography variant="h6">Network I/O</Typography>
              </Box>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>312</Typography>
              <Typography variant="body2">MB/s Throughput</Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Devices sx={{ mr: 1 }} />
                <Typography variant="h6">Active Nodes</Typography>
              </Box>
              <Typography variant="h3" sx={{ fontWeight: 'bold' }}>24/25</Typography>
              <Typography variant="body2">Edge Participants</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* System Performance Chart */}
        <Grid item xs={12} md={8}>
          <Card sx={{ height: '400px' }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Typography variant="h6">
                  System Performance Metrics
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  {['1h', '6h', '24h', '7d'].map((range) => (
                    <Button
                      key={range}
                      size="small"
                      variant={selectedTimeRange === range ? 'contained' : 'outlined'}
                      onClick={() => setSelectedTimeRange(range)}
                    >
                      {range}
                    </Button>
                  ))}
                </Box>
              </Box>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={systemMetrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <RechartsTooltip />
                  <Area 
                    type="monotone" 
                    dataKey="cpu" 
                    stackId="1" 
                    stroke="#8884d8" 
                    fill="#8884d8" 
                    fillOpacity={0.3}
                    name="CPU %"
                  />
                  <Area 
                    type="monotone" 
                    dataKey="memory" 
                    stackId="2" 
                    stroke="#82ca9d" 
                    fill="#82ca9d" 
                    fillOpacity={0.3}
                    name="Memory %"
                  />
                  <Area 
                    type="monotone" 
                    dataKey="network" 
                    stackId="3" 
                    stroke="#ffc658" 
                    fill="#ffc658" 
                    fillOpacity={0.3}
                    name="Network %"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Resource Distribution */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '400px' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 3 }}>
                Current Resource Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={resourceDistribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {resourceDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <RechartsTooltip />
                </PieChart>
              </ResponsiveContainer>
              <Box sx={{ mt: 2 }}>
                {resourceDistribution.map((item, index) => (
                  <Box key={index} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <Box 
                      sx={{ 
                        width: 12, 
                        height: 12, 
                        backgroundColor: item.color, 
                        borderRadius: '50%', 
                        mr: 1 
                      }} 
                    />
                    <Typography variant="body2" sx={{ flexGrow: 1 }}>
                      {item.name}
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                      {item.value}%
                    </Typography>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Edge Nodes Status */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 3 }}>
                Edge Node Health & Performance
              </Typography>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Node</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>CPU</TableCell>
                      <TableCell>Memory</TableCell>
                      <TableCell>Uptime</TableCell>
                      <TableCell>Location</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {nodeStatus.map((node) => (
                      <TableRow key={node.id}>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            {getStatusIcon(node.status)}
                            <Box sx={{ ml: 1 }}>
                              <Typography variant="subtitle2">{node.name}</Typography>
                              <Typography variant="caption" color="text.secondary">
                                {node.id}
                              </Typography>
                            </Box>
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
                          <Box>
                            <Typography variant="body2">{node.cpu}%</Typography>
                            <LinearProgress 
                              variant="determinate" 
                              value={node.cpu} 
                              sx={{ width: 60, height: 4 }}
                              color={node.cpu > 80 ? 'error' : node.cpu > 60 ? 'warning' : 'success'}
                            />
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Box>
                            <Typography variant="body2">{node.memory}%</Typography>
                            <LinearProgress 
                              variant="determinate" 
                              value={node.memory} 
                              sx={{ width: 60, height: 4 }}
                              color={node.memory > 80 ? 'error' : node.memory > 60 ? 'warning' : 'success'}
                            />
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">{formatUptime(node.uptime)}</Typography>
                        </TableCell>
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

        {/* Training Performance */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 3 }}>
                Training Round Performance
              </Typography>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={trainingMetrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="round" />
                  <YAxis />
                  <RechartsTooltip />
                  <Bar dataKey="participants" fill="#8884d8" name="Participants" />
                </BarChart>
              </ResponsiveContainer>
              
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>Latest Metrics:</Typography>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">Accuracy:</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 'bold' }}>89.2%</Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">Participants:</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 'bold' }}>24 nodes</Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">Avg. Round Time:</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 'bold' }}>4.0 min</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* System Status Alert */}
      <Alert 
        severity={nodeStatus.some(n => n.status === 'critical') ? 'error' : 
                 nodeStatus.some(n => n.status === 'warning') ? 'warning' : 'success'} 
        sx={{ mt: 3 }}
      >
        <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>
          📊 System Status: {nodeStatus.some(n => n.status === 'critical') ? 'CRITICAL ATTENTION REQUIRED' : 
                             nodeStatus.some(n => n.status === 'warning') ? 'WARNING - MONITORING' : 'ALL SYSTEMS OPERATIONAL'}
        </Typography>
        <Typography variant="body2">
          {nodeStatus.filter(n => n.status === 'healthy').length} healthy nodes • 
          {nodeStatus.filter(n => n.status === 'warning').length} warnings • 
          {nodeStatus.filter(n => n.status === 'critical').length} critical • 
          Real-time monitoring active with 30-second refresh intervals
        </Typography>
      </Alert>
    </Box>
  );
};

export default SystemMonitoring;