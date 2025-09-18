import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Avatar,
  Button,
  Alert,
} from '@mui/material';
import {
  Security as SecurityIcon,
  Devices as DevicesIcon,
  ModelTraining as MLIcon,
  Speed as SpeedIcon,
  TrendingUp,
  Warning,
  CheckCircle,
  Error,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { useAuth } from '../contexts/AuthContext';
import UserDashboard from '../components/UserDashboard';

// Mock data for the dashboard
const performanceData = [
  { time: '00:00', throughput: 234, latency: 2.4 },
  { time: '04:00', throughput: 189, latency: 3.1 },
  { time: '08:00', throughput: 267, latency: 2.1 },
  { time: '12:00', throughput: 298, latency: 1.8 },
  { time: '16:00', throughput: 312, latency: 1.9 },
  { time: '20:00', throughput: 289, latency: 2.2 },
];

const deviceData = [
  { category: 'Healthcare', count: 12 },
  { category: 'Finance', count: 8 },
  { category: 'Automotive', count: 15 },
  { category: 'Research', count: 4 },
];

const recentActivities = [
  { id: 1, type: 'device', message: 'New device registered: MedTech-Sensor-007', time: '2 minutes ago', status: 'success' },
  { id: 2, type: 'security', message: 'Quantum key exchange completed successfully', time: '5 minutes ago', status: 'success' },
  { id: 3, type: 'training', message: 'Federated learning round 15 completed', time: '8 minutes ago', status: 'success' },
  { id: 4, type: 'warning', message: 'High latency detected on edge node EN-03', time: '12 minutes ago', status: 'warning' },
  { id: 5, type: 'security', message: 'Security scan completed - No threats detected', time: '15 minutes ago', status: 'success' },
];

const Dashboard = () => {
  const { isAdmin } = useAuth();
  
  // Always call hooks first, regardless of conditions
  const [stats, setStats] = useState({
    totalDevices: 39,
    onlineDevices: 37,
    activeTraining: 3,
    securityEvents: 0,
    avgThroughput: 267,
    avgLatency: 2.1,
  });

  useEffect(() => {
    // Only run effect for admin dashboard
    if (!isAdmin) return;
    
    // Simulate real-time updates
    const interval = setInterval(() => {
      setStats(prev => ({
        ...prev,
        onlineDevices: 37 + Math.floor(Math.random() * 3),
        avgThroughput: 250 + Math.floor(Math.random() * 50),
        avgLatency: 2.0 + Math.random() * 0.5,
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, [isAdmin]);

  // If user is not admin, show user-specific dashboard
  if (!isAdmin) {
    return <UserDashboard />;
  }

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'device':
        return <DevicesIcon sx={{ color: 'primary.main' }} />;
      case 'security':
        return <SecurityIcon sx={{ color: 'success.main' }} />;
      case 'training':
        return <MLIcon sx={{ color: 'info.main' }} />;
      case 'warning':
        return <Warning sx={{ color: 'warning.main' }} />;
      default:
        return <CheckCircle sx={{ color: 'success.main' }} />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success':
        return 'success';
      case 'warning':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
        Dashboard Overview
      </Typography>

      {/* Alert Banner */}
      <Alert severity="success" sx={{ mb: 3 }}>
        <strong>System Status:</strong> All quantum-safe protocols active. 37 devices online and secure.
      </Alert>

      {/* Key Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #1976d2 0%, #42a5f5 100%)', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                    {stats.onlineDevices}
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9 }}>
                    Devices Online
                  </Typography>
                </Box>
                <DevicesIcon sx={{ fontSize: 40, opacity: 0.8 }} />
              </Box>
              <Box sx={{ mt: 2 }}>
                <LinearProgress 
                  variant="determinate" 
                  value={(stats.onlineDevices / stats.totalDevices) * 100}
                  sx={{ 
                    backgroundColor: 'rgba(255,255,255,0.3)',
                    '& .MuiLinearProgress-bar': { backgroundColor: 'white' }
                  }}
                />
                <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
                  {stats.onlineDevices}/{stats.totalDevices} Active
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #388e3c 0%, #66bb6a 100%)', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                    {stats.activeTraining}
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9 }}>
                    Active Training
                  </Typography>
                </Box>
                <MLIcon sx={{ fontSize: 40, opacity: 0.8 }} />
              </Box>
              <Chip 
                label="FL Round 15" 
                size="small" 
                sx={{ 
                  mt: 2, 
                  backgroundColor: 'rgba(255,255,255,0.2)', 
                  color: 'white' 
                }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #f57c00 0%, #ffb74d 100%)', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                    {stats.avgThroughput}
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9 }}>
                    Ops/sec
                  </Typography>
                </Box>
                <SpeedIcon sx={{ fontSize: 40, opacity: 0.8 }} />
              </Box>
              <Typography variant="caption" sx={{ mt: 2, display: 'block' }}>
                Avg Latency: {stats.avgLatency.toFixed(1)}ms
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #7b1fa2 0%, #ba68c8 100%)', color: 'white' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                    {stats.securityEvents}
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9 }}>
                    Security Events
                  </Typography>
                </Box>
                <SecurityIcon sx={{ fontSize: 40, opacity: 0.8 }} />
              </Box>
              <Chip 
                label="Quantum Safe" 
                size="small" 
                sx={{ 
                  mt: 2, 
                  backgroundColor: 'rgba(255,255,255,0.2)', 
                  color: 'white' 
                }}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts Section */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Performance Metrics
                </Typography>
                <Button size="small" startIcon={<TrendingUp />}>
                  View Details
                </Button>
              </Box>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip />
                    <Line 
                      yAxisId="left"
                      type="monotone" 
                      dataKey="throughput" 
                      stroke="#1976d2" 
                      strokeWidth={2}
                      name="Throughput (ops/sec)"
                    />
                    <Line 
                      yAxisId="right"
                      type="monotone" 
                      dataKey="latency" 
                      stroke="#f57c00" 
                      strokeWidth={2}
                      name="Latency (ms)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                Device Distribution
              </Typography>
              <Box sx={{ height: 250 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={deviceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="category" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#1976d2" />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Recent Activity */}
      <Card>
        <CardContent>
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
            Recent Activity
          </Typography>
          <List>
            {recentActivities.map((activity) => (
              <ListItem key={activity.id} divider>
                <ListItemIcon>
                  <Avatar sx={{ width: 32, height: 32, bgcolor: 'transparent' }}>
                    {getActivityIcon(activity.type)}
                  </Avatar>
                </ListItemIcon>
                <ListItemText
                  primary={activity.message}
                  secondary={activity.time}
                />
                <Chip 
                  label={activity.status}
                  color={getStatusColor(activity.status) as any}
                  size="small"
                  variant="outlined"
                />
              </ListItem>
            ))}
          </List>
          <Box sx={{ mt: 2, textAlign: 'center' }}>
            <Button variant="outlined">View All Activities</Button>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Dashboard;