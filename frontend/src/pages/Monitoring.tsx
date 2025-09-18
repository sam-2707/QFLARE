import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Tooltip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip
} from '@mui/material';
import {
  MonitorHeart as MonitorIcon,
  Speed as PerformanceIcon,
  Memory as MemoryIcon,
  Storage as StorageIcon,
  NetworkCheck as NetworkIcon,
  Refresh as RefreshIcon,
  CheckCircle as CheckIcon,
  Warning as WarningIcon,
  Error as ErrorIcon
} from '@mui/icons-material';

interface SystemMetric {
  name: string;
  value: number;
  unit: string;
  status: 'healthy' | 'warning' | 'critical';
  threshold: number;
}

interface Alert {
  id: string;
  type: string;
  severity: 'info' | 'warning' | 'error';
  message: string;
  timestamp: string;
}

const Monitoring = () => {
  const [metrics, setMetrics] = useState<SystemMetric[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Mock data - replace with actual API calls
    setMetrics([
      { name: 'CPU Usage', value: 45, unit: '%', status: 'healthy', threshold: 80 },
      { name: 'Memory Usage', value: 67, unit: '%', status: 'warning', threshold: 75 },
      { name: 'Disk Usage', value: 34, unit: '%', status: 'healthy', threshold: 85 },
      { name: 'Network Latency', value: 12, unit: 'ms', status: 'healthy', threshold: 50 }
    ]);

    setAlerts([
      {
        id: '1',
        type: 'Performance',
        severity: 'warning',
        message: 'Memory usage approaching threshold',
        timestamp: '2024-01-15T10:30:00Z'
      },
      {
        id: '2',
        type: 'Connectivity',
        severity: 'info',
        message: 'Device DEV-003 reconnected',
        timestamp: '2024-01-15T10:25:00Z'
      }
    ]);

    setLoading(false);
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckIcon color="success" />;
      case 'warning': return <WarningIcon color="warning" />;
      case 'critical': return <ErrorIcon color="error" />;
      default: return <CheckIcon />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'info': return 'info';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  if (loading) {
    return (
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <LinearProgress />
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ flexGrow: 1 }}>
          System Monitoring
        </Typography>
        <Tooltip title="Refresh metrics">
          <IconButton>
            <RefreshIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {/* System Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {metrics.map((metric, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  {metric.name === 'CPU Usage' && <PerformanceIcon sx={{ mr: 1 }} />}
                  {metric.name === 'Memory Usage' && <MemoryIcon sx={{ mr: 1 }} />}
                  {metric.name === 'Disk Usage' && <StorageIcon sx={{ mr: 1 }} />}
                  {metric.name === 'Network Latency' && <NetworkIcon sx={{ mr: 1 }} />}
                  <Typography variant="h6" sx={{ flexGrow: 1 }}>
                    {metric.name}
                  </Typography>
                  {getStatusIcon(metric.status)}
                </Box>
                <Typography variant="h4" color="primary" gutterBottom>
                  {metric.value}{metric.unit}
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={(metric.value / metric.threshold) * 100}
                  color={metric.status === 'critical' ? 'error' : metric.status === 'warning' ? 'warning' : 'primary'}
                />
                <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                  Threshold: {metric.threshold}{metric.unit}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Grid container spacing={3}>
        {/* Performance Chart Placeholder */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Performance Trends
            </Typography>
            <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: 'grey.100' }}>
              <Typography color="textSecondary">
                Performance charts will be displayed here
              </Typography>
            </Box>
          </Paper>
        </Grid>

        {/* Recent Alerts */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Alerts
            </Typography>
            <List>
              {alerts.map((alert) => (
                <ListItem key={alert.id}>
                  <ListItemIcon>
                    {alert.severity === 'error' && <ErrorIcon color="error" />}
                    {alert.severity === 'warning' && <WarningIcon color="warning" />}
                    {alert.severity === 'info' && <CheckIcon color="info" />}
                  </ListItemIcon>
                  <ListItemText
                    primary={alert.message}
                    secondary={`${alert.type} - ${new Date(alert.timestamp).toLocaleString()}`}
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        {/* System Status Table */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              System Status
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Service</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Uptime</TableCell>
                    <TableCell>Last Check</TableCell>
                    <TableCell>Response Time</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  <TableRow>
                    <TableCell>API Server</TableCell>
                    <TableCell>
                      <Chip label="Healthy" color="success" size="small" />
                    </TableCell>
                    <TableCell>99.9%</TableCell>
                    <TableCell>Just now</TableCell>
                    <TableCell>45ms</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Database</TableCell>
                    <TableCell>
                      <Chip label="Healthy" color="success" size="small" />
                    </TableCell>
                    <TableCell>99.8%</TableCell>
                    <TableCell>30s ago</TableCell>
                    <TableCell>12ms</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Redis Cache</TableCell>
                    <TableCell>
                      <Chip label="Warning" color="warning" size="small" />
                    </TableCell>
                    <TableCell>98.5%</TableCell>
                    <TableCell>1m ago</TableCell>
                    <TableCell>78ms</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Monitoring;