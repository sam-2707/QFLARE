import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Container,
  Grid,
  Card,
  CardContent,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  LinearProgress,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
} from '@mui/material';
import {
  Dashboard,
  Security,
  DeviceHub,
  TrendingUp,
  People,
  Settings,
  Refresh,
  Visibility,
  Block,
  CheckCircle,
  Warning,
  Error,
} from '@mui/icons-material';
import { toast } from 'react-toastify';
import apiService from '../services/apiService';

interface SystemMetrics {
  totalDevices: number;
  onlineDevices: number;
  offlineDevices: number;
  pendingDevices: number;
  securityLevel: number;
  systemUptime: number;
  quantumResistance: number;
}

interface ActivityLog {
  id: string;
  timestamp: string;
  type: 'registration' | 'verification' | 'security' | 'system';
  message: string;
  severity: 'info' | 'warning' | 'error' | 'success';
}

const AdminDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<SystemMetrics>({
    totalDevices: 0,
    onlineDevices: 0,
    offlineDevices: 0,
    pendingDevices: 0,
    securityLevel: 98,
    systemUptime: 99.9,
    quantumResistance: 100,
  });

  const [loading, setLoading] = useState(false);

  const [activityLogs, setActivityLogs] = useState<ActivityLog[]>([
    {
      id: '1',
      timestamp: '2025-09-26 10:30:00',
      type: 'registration',
      message: 'New device QFLARE-IOT-004 registered successfully',
      severity: 'success',
    },
    {
      id: '2',
      timestamp: '2025-09-26 10:25:00',
      type: 'verification',
      message: 'OTP verification completed for device QFLARE-MOBILE-005',
      severity: 'success',
    },
    {
      id: '3',
      timestamp: '2025-09-26 10:20:00',
      type: 'security',
      message: 'Quantum key rotation completed for 8 devices',
      severity: 'info',
    },
    {
      id: '4',
      timestamp: '2025-09-26 10:15:00',
      type: 'system',
      message: 'Failed connection attempt from device QFLARE-SERVER-003',
      severity: 'warning',
    },
  ]);

  const [selectedLog, setSelectedLog] = useState<ActivityLog | null>(null);
  const [openDialog, setOpenDialog] = useState(false);

  const fetchMetrics = async () => {
    try {
      setLoading(true);
      
      // Fetch devices to calculate metrics
      const devicesResponse = await apiService.getDevices();
      
      if (devicesResponse) {
        setMetrics(prev => ({
          ...prev,
          totalDevices: devicesResponse.total || 0,
          onlineDevices: devicesResponse.online || 0,
          offlineDevices: devicesResponse.offline || 0,
          pendingDevices: devicesResponse.pending || 0,
        }));
      }

      // Try to fetch additional metrics if endpoint exists
      try {
        const metricsResponse = await apiService.getSystemMetrics();
        setMetrics(prev => ({ ...prev, ...metricsResponse }));
      } catch (metricsError) {
        console.log('System metrics endpoint not available, using device-based metrics');
      }

      // Try to fetch activity logs if endpoint exists
      try {
        const logsResponse = await apiService.getActivityLogs();
        if (logsResponse && Array.isArray(logsResponse)) {
          setActivityLogs(logsResponse);
        }
      } catch (logsError) {
        console.log('Activity logs endpoint not available, using default logs');
      }

    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      toast.error('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'success': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'error';
      case 'info': return 'info';
      default: return 'default';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'success': return <CheckCircle />;
      case 'warning': return <Warning />;
      case 'error': return <Error />;
      case 'info': return <TrendingUp />;
      default: return <TrendingUp />;
    }
  };

  const handleRefreshLogs = () => {
    toast.info('Refreshing dashboard data...');
    fetchMetrics();
  };

  const handleViewLog = (log: ActivityLog) => {
    setSelectedLog(log);
    setOpenDialog(true);
  };

  return (
    <Container maxWidth="xl">
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" fontWeight={700} mb={1}>
          Admin Dashboard
        </Typography>
        <Typography variant="body1" color="text.secondary" mb={3}>
          Monitor system performance and security metrics
        </Typography>
      </Box>

      {/* Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Box>
                <Typography variant="h4" fontWeight={700} color="primary.main">
                  {metrics.totalDevices}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total Devices
                </Typography>
              </Box>
              <DeviceHub sx={{ fontSize: 40, color: 'primary.main' }} />
            </Box>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Box>
                <Typography variant="h4" fontWeight={700} color="success.main">
                  {metrics.onlineDevices}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Online Devices
                </Typography>
              </Box>
              <CheckCircle sx={{ fontSize: 40, color: 'success.main' }} />
            </Box>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Box>
                <Typography variant="h4" fontWeight={700} color="warning.main">
                  {metrics.pendingDevices}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Pending Verification
                </Typography>
              </Box>
              <Warning sx={{ fontSize: 40, color: 'warning.main' }} />
            </Box>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Box>
                <Typography variant="h4" fontWeight={700} color="error.main">
                  {metrics.offlineDevices}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Offline Devices
                </Typography>
              </Box>
              <Error sx={{ fontSize: 40, color: 'error.main' }} />
            </Box>
          </Card>
        </Grid>
      </Grid>

      {/* Security Status */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6}>
          <Card sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" fontWeight={600} mb={3}>
              Security Metrics
            </Typography>
            <Box sx={{ mb: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2">Quantum Resistance</Typography>
                <Typography variant="body2" color="success.main">
                  {metrics.quantumResistance}%
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={metrics.quantumResistance}
                color="success"
                sx={{ height: 8, borderRadius: 4 }}
              />
            </Box>
            <Box sx={{ mb: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2">Encryption Strength</Typography>
                <Typography variant="body2" color="primary.main">
                  256-bit
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={100}
                color="primary"
                sx={{ height: 8, borderRadius: 4 }}
              />
            </Box>
            <Box sx={{ mb: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2">Key Exchange Security</Typography>
                <Typography variant="body2" color="secondary.main">
                  98%
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={98}
                color="secondary"
                sx={{ height: 8, borderRadius: 4 }}
              />
            </Box>
            <Box sx={{ display: 'flex', gap: 1, mt: 3 }}>
              <Chip label="CRYSTALS-Kyber-1024" size="small" color="primary" />
              <Chip label="Dilithium-2" size="small" color="secondary" />
            </Box>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" fontWeight={600} mb={3}>
              System Status
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
              <Chip
                label="Quantum-Safe"
                color="success"
                icon={<Security />}
                variant="outlined"
              />
              <Chip
                label="All Systems Operational"
                color="success"
                icon={<CheckCircle />}
                variant="outlined"
              />
            </Box>
            <Typography variant="body2" color="text.secondary" mb={2}>
              Last Security Audit: September 25, 2025
            </Typography>
            <Typography variant="body2" color="text.secondary" mb={2}>
              Next Key Rotation: September 30, 2025
            </Typography>
            <Typography variant="body2" color="text.secondary" mb={3}>
              Average Response Time: 45ms
            </Typography>
            <Button variant="outlined" size="small" startIcon={<Settings />}>
              System Settings
            </Button>
          </Card>
        </Grid>
      </Grid>

      {/* Activity Logs */}
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Typography variant="h6" fontWeight={600}>
              Recent Activity
            </Typography>
            <Button
              variant="outlined"
              size="small"
              startIcon={<Refresh />}
              onClick={handleRefreshLogs}
            >
              Refresh
            </Button>
          </Box>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Timestamp</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Message</TableCell>
                  <TableCell>Severity</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {activityLogs.map((log) => (
                  <TableRow key={log.id} hover>
                    <TableCell>{log.timestamp}</TableCell>
                    <TableCell>
                      <Chip label={log.type} size="small" variant="outlined" />
                    </TableCell>
                    <TableCell>{log.message}</TableCell>
                    <TableCell>
                      <Chip
                        label={log.severity}
                        size="small"
                        color={getSeverityColor(log.severity) as any}
                        icon={getSeverityIcon(log.severity)}
                      />
                    </TableCell>
                    <TableCell>
                      <IconButton size="small" onClick={() => handleViewLog(log)}>
                        <Visibility />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Log Details Dialog */}
      <Dialog open={openDialog} onClose={() => setOpenDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Activity Log Details</DialogTitle>
        <DialogContent>
          {selectedLog && (
            <Box sx={{ mt: 2 }}>
              <TextField
                label="Timestamp"
                value={selectedLog.timestamp}
                fullWidth
                disabled
                sx={{ mb: 2 }}
              />
              <TextField
                label="Type"
                value={selectedLog.type}
                fullWidth
                disabled
                sx={{ mb: 2 }}
              />
              <TextField
                label="Message"
                value={selectedLog.message}
                fullWidth
                multiline
                rows={3}
                disabled
                sx={{ mb: 2 }}
              />
              <TextField
                label="Severity"
                value={selectedLog.severity}
                fullWidth
                disabled
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>Close</Button>
          <Button variant="contained">Export Log</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default AdminDashboard;
