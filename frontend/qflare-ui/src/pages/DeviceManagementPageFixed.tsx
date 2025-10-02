import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Alert,
  LinearProgress
} from '@mui/material';
import {
  Add as AddIcon,
  Refresh as RefreshIcon,
  Delete as DeleteIcon,
  Computer as ComputerIcon,
  Smartphone as SmartphoneIcon,
  Cloud as CloudIcon,
  Memory as MemoryIcon
} from '@mui/icons-material';

interface Device {
  device_id: string;
  device_name: string;
  device_type: string;
  status: 'online' | 'offline' | 'training' | 'idle' | 'error' | 'maintenance';
  capabilities: string[];
  location?: string;
  contact_info?: string;
  registered_at: string;
  last_seen: string;
  total_training_sessions: number;
  total_training_time: number;
  success_rate: number;
  current_task?: string;
}

interface DeviceRegistration {
  device_name: string;
  device_type: string;
  capabilities: string[];
  location?: string;
  contact_info?: string;
  max_concurrent_tasks: number;
}

const API_BASE_URL = 'http://localhost:8000';

const DeviceManagementPage: React.FC = () => {
  const [devices, setDevices] = useState<Device[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [openDialog, setOpenDialog] = useState(false);
  const [newDevice, setNewDevice] = useState<DeviceRegistration>({
    device_name: '',
    device_type: 'desktop',
    capabilities: [],
    location: '',
    contact_info: '',
    max_concurrent_tasks: 1
  });

  const deviceTypes = ['desktop', 'mobile', 'server', 'edge', 'cloud', 'workstation'];
  const capabilityOptions = ['cpu', 'gpu', 'tpu', 'mobile', 'edge', 'cloud'];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return '#4caf50';
      case 'training': return '#ff9800';
      case 'idle': return '#2196f3';
      case 'offline': return '#f44336';
      case 'error': return '#f44336';
      case 'maintenance': return '#9c27b0';
      default: return '#757575';
    }
  };

  const getDeviceIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'mobile': return <SmartphoneIcon />;
      case 'server':
      case 'cloud': return <CloudIcon />;
      case 'edge': return <MemoryIcon />;
      default: return <ComputerIcon />;
    }
  };

  const fetchDevices = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/devices/`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setDevices(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching devices:', err);
      setError('Failed to fetch devices. Please check if the server is running.');
    } finally {
      setLoading(false);
    }
  };

  const registerDevice = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/devices/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newDevice),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to register device');
      }

      setOpenDialog(false);
      setNewDevice({
        device_name: '',
        device_type: 'desktop',
        capabilities: [],
        location: '',
        contact_info: '',
        max_concurrent_tasks: 1
      });
      fetchDevices();
    } catch (err) {
      console.error('Error registering device:', err);
      setError(err instanceof Error ? err.message : 'Failed to register device');
    }
  };

  const deleteDevice = async (deviceId: string) => {
    if (!window.confirm('Are you sure you want to unregister this device?')) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/devices/${deviceId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to delete device');
      }

      fetchDevices();
    } catch (err) {
      console.error('Error deleting device:', err);
      setError('Failed to delete device');
    }
  };

  const sendHeartbeat = async (deviceId: string) => {
    try {
      await fetch(`${API_BASE_URL}/api/devices/${deviceId}/heartbeat`, {
        method: 'POST',
      });
      fetchDevices();
    } catch (err) {
      console.error('Error sending heartbeat:', err);
    }
  };

  useEffect(() => {
    fetchDevices();
    const interval = setInterval(fetchDevices, 30000);
    return () => clearInterval(interval);
  }, []);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const formatDuration = (hours: number) => {
    if (hours < 1) {
      return `${Math.round(hours * 60)}m`;
    }
    return `${hours.toFixed(1)}h`;
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1">
          Device Management
        </Typography>
        <Box>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={fetchDevices}
            sx={{ mr: 2 }}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setOpenDialog(true)}
          >
            Register Device
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Network Overview
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={3}>
                  <Box textAlign="center">
                    <Typography variant="h3" color="primary">
                      {devices.length}
                    </Typography>
                    <Typography variant="body2">Total Devices</Typography>
                  </Box>
                </Grid>
                <Grid item xs={3}>
                  <Box textAlign="center">
                    <Typography variant="h3" color="success.main">
                      {devices.filter(d => d.status === 'online').length}
                    </Typography>
                    <Typography variant="body2">Online</Typography>
                  </Box>
                </Grid>
                <Grid item xs={3}>
                  <Box textAlign="center">
                    <Typography variant="h3" color="warning.main">
                      {devices.filter(d => d.status === 'training').length}
                    </Typography>
                    <Typography variant="body2">Training</Typography>
                  </Box>
                </Grid>
                <Grid item xs={3}>
                  <Box textAlign="center">
                    <Typography variant="h3" color="error.main">
                      {devices.filter(d => d.status === 'offline').length}
                    </Typography>
                    <Typography variant="body2">Offline</Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Registered Devices
              </Typography>
              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Device</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Type</TableCell>
                      <TableCell>Capabilities</TableCell>
                      <TableCell>Training Stats</TableCell>
                      <TableCell>Last Seen</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {devices.map((device) => (
                      <TableRow key={device.device_id}>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            {getDeviceIcon(device.device_type)}
                            <Box sx={{ ml: 1 }}>
                              <Typography variant="body1" fontWeight="bold">
                                {device.device_name}
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                {device.location || 'No location'}
                              </Typography>
                            </Box>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={device.status}
                            size="small"
                            sx={{
                              backgroundColor: getStatusColor(device.status),
                              color: 'white',
                              fontWeight: 'bold'
                            }}
                          />
                          {device.current_task && (
                            <Typography variant="caption" display="block">
                              Task: {device.current_task.substring(0, 10)}...
                            </Typography>
                          )}
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">{device.device_type}</Typography>
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                            {device.capabilities.map((cap) => (
                              <Chip key={cap} label={cap} size="small" variant="outlined" />
                            ))}
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            Sessions: {device.total_training_sessions}
                          </Typography>
                          <Typography variant="body2">
                            Time: {formatDuration(device.total_training_time)}
                          </Typography>
                          <Typography variant="body2">
                            Success: {device.success_rate.toFixed(1)}%
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            {formatDate(device.last_seen)}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', gap: 1 }}>
                            <Button
                              size="small"
                              variant="outlined"
                              onClick={() => sendHeartbeat(device.device_id)}
                            >
                              Ping
                            </Button>
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => deleteDevice(device.device_id)}
                            >
                              <DeleteIcon />
                            </IconButton>
                          </Box>
                        </TableCell>
                      </TableRow>
                    ))}
                    {devices.length === 0 && !loading && (
                      <TableRow>
                        <TableCell colSpan={7} align="center">
                          <Typography variant="body1" color="text.secondary">
                            No devices registered yet. Register your first device to get started!
                          </Typography>
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

      <Dialog open={openDialog} onClose={() => setOpenDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Register New Device</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <TextField
              label="Device Name"
              value={newDevice.device_name}
              onChange={(e) => setNewDevice({ ...newDevice, device_name: e.target.value })}
              required
              fullWidth
            />
            
            <FormControl fullWidth>
              <InputLabel>Device Type</InputLabel>
              <Select
                value={newDevice.device_type}
                onChange={(e) => setNewDevice({ ...newDevice, device_type: e.target.value })}
              >
                {deviceTypes.map((type) => (
                  <MenuItem key={type} value={type}>
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth>
              <InputLabel>Capabilities</InputLabel>
              <Select
                multiple
                value={newDevice.capabilities}
                onChange={(e) => setNewDevice({ 
                  ...newDevice, 
                  capabilities: typeof e.target.value === 'string' ? e.target.value.split(',') : e.target.value 
                })}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {capabilityOptions.map((cap) => (
                  <MenuItem key={cap} value={cap}>
                    {cap.toUpperCase()}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <TextField
              label="Location (Optional)"
              value={newDevice.location}
              onChange={(e) => setNewDevice({ ...newDevice, location: e.target.value })}
              placeholder="e.g., New York, Data Center A"
            />

            <TextField
              label="Contact Info (Optional)"
              value={newDevice.contact_info}
              onChange={(e) => setNewDevice({ ...newDevice, contact_info: e.target.value })}
              placeholder="e.g., admin@company.com"
            />

            <TextField
              label="Max Concurrent Tasks"
              type="number"
              value={newDevice.max_concurrent_tasks}
              onChange={(e) => setNewDevice({ 
                ...newDevice, 
                max_concurrent_tasks: parseInt(e.target.value) || 1 
              })}
              inputProps={{ min: 1, max: 10 }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>Cancel</Button>
          <Button 
            onClick={registerDevice} 
            variant="contained"
            disabled={!newDevice.device_name.trim()}
          >
            Register Device
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DeviceManagementPage;