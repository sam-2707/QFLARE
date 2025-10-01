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
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  DeviceHub,
  Security,
  Add,
  Refresh,
  Visibility,
  Delete,
  Edit,
  CheckCircle,
  Error,
  Warning,
} from '@mui/icons-material';
import { toast } from 'react-toastify';
import apiService from '../services/apiService';

interface Device {
  id: string;
  device_id: string;
  device_type: string;
  organization: string;
  contact_email?: string;
  use_case?: string;
  status: 'online' | 'offline' | 'pending';
  security_level: string;
  enrolled_at: string;
  registered_at?: string;
  last_seen: string;
}

const DevicesPage: React.FC = () => {
  const [devices, setDevices] = useState<Device[]>([]);
  const [loading, setLoading] = useState(false);
  const [openDialog, setOpenDialog] = useState(false);
  const [selectedDevice, setSelectedDevice] = useState<Device | null>(null);
  const [viewMode, setViewMode] = useState<'grid' | 'table'>('grid');

  const fetchDevices = async () => {
    try {
      setLoading(true);
      const response = await apiService.getDevices();
      console.log('Devices API response:', response);
      
      if (response.devices) {
        setDevices(response.devices);
        toast.success(`Loaded ${response.devices.length} devices`);
      } else {
        setDevices([]);
        toast.info('No devices found');
      }
    } catch (error) {
      console.error('Error fetching devices:', error);
      toast.error('Failed to load devices');
      setDevices([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDevices();
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'success';
      case 'offline': return 'error';
      case 'pending': return 'warning';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online': return <CheckCircle />;
      case 'offline': return <Error />;
      case 'pending': return <Warning />;
      default: return <DeviceHub />;
    }
  };

  const handleRefresh = () => {
    toast.info('Refreshing device list...');
    fetchDevices();
  };

  const handleViewDevice = (device: Device) => {
    setSelectedDevice(device);
    setOpenDialog(true);
  };

  const DeviceCard = ({ device }: { device: Device }) => (
    <Card
      sx={{
        height: '100%',
        transition: 'transform 0.2s ease-in-out',
        '&:hover': {
          transform: 'translateY(-4px)',
        },
      }}
    >
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
          <DeviceHub sx={{ fontSize: 40, color: 'primary.main' }} />
          <Chip
            label={device.status}
            color={getStatusColor(device.status) as any}
            size="small"
            icon={getStatusIcon(device.status)}
          />
        </Box>
        <Typography variant="h6" fontWeight={600} mb={1} noWrap>
          {device.device_id}
        </Typography>
        <Typography variant="body2" color="text.secondary" mb={1}>
          {device.device_type}
        </Typography>
        <Typography variant="body2" color="text.secondary" mb={2}>
          {device.organization}
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
          <Chip label={device.security_level} size="small" color="primary" />
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            size="small"
            variant="outlined"
            onClick={() => handleViewDevice(device)}
            startIcon={<Visibility />}
          >
            View
          </Button>
          <IconButton size="small" color="primary">
            <Edit />
          </IconButton>
          <IconButton size="small" color="error">
            <Delete />
          </IconButton>
        </Box>
      </CardContent>
    </Card>
  );

  return (
    <Container maxWidth="xl">
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" fontWeight={700} mb={1}>
          Device Management
        </Typography>
        <Typography variant="body1" color="text.secondary" mb={3}>
          Monitor and manage your quantum-secure devices
        </Typography>
        
        {/* Stats Cards */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ p: 3, textAlign: 'center' }}>
              <DeviceHub sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
              <Typography variant="h4" fontWeight={700} color="primary.main">
                {devices.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Devices
              </Typography>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ p: 3, textAlign: 'center' }}>
              <CheckCircle sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
              <Typography variant="h4" fontWeight={700} color="success.main">
                {devices.filter(d => d.status === 'online').length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Online
              </Typography>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ p: 3, textAlign: 'center' }}>
              <Error sx={{ fontSize: 40, color: 'error.main', mb: 1 }} />
              <Typography variant="h4" fontWeight={700} color="error.main">
                {devices.filter(d => d.status === 'offline').length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Offline
              </Typography>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card sx={{ p: 3, textAlign: 'center' }}>
              <Security sx={{ fontSize: 40, color: 'secondary.main', mb: 1 }} />
              <Typography variant="h4" fontWeight={700} color="secondary.main">
                100%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Quantum-Safe
              </Typography>
            </Card>
          </Grid>
        </Grid>

        {/* Actions */}
        <Box sx={{ display: 'flex', gap: 2, mb: 4 }}>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => window.location.href = '/secure-register'}
          >
            Add Device
          </Button>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={handleRefresh}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {/* Device Grid */}
      <Grid container spacing={3}>
        {devices.map((device) => (
          <Grid item xs={12} sm={6} md={4} lg={3} key={device.id}>
            <DeviceCard device={device} />
          </Grid>
        ))}
      </Grid>

      {devices.length === 0 && (
        <Paper sx={{ p: 6, textAlign: 'center' }}>
          <DeviceHub sx={{ fontSize: 80, color: 'text.disabled', mb: 2 }} />
          <Typography variant="h5" color="text.secondary" mb={2}>
            No devices registered
          </Typography>
          <Typography variant="body1" color="text.secondary" mb={4}>
            Start by registering your first quantum-secure device
          </Typography>
          <Button
            variant="contained"
            size="large"
            onClick={() => window.location.href = '/secure-register'}
            startIcon={<Add />}
          >
            Register First Device
          </Button>
        </Paper>
      )}

      {/* Device Details Dialog */}
      <Dialog open={openDialog} onClose={() => setOpenDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <DeviceHub color="primary" />
            Device Details
          </Box>
        </DialogTitle>
        <DialogContent>
          {selectedDevice && (
            <Grid container spacing={3} sx={{ mt: 1 }}>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Device ID"
                  value={selectedDevice.device_id}
                  fullWidth
                  disabled
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Device Type"
                  value={selectedDevice.device_type}
                  fullWidth
                  disabled
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Organization"
                  value={selectedDevice.organization}
                  fullWidth
                  disabled
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Status"
                  value={selectedDevice.status}
                  fullWidth
                  disabled
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Security Level"
                  value={selectedDevice.security_level}
                  fullWidth
                  disabled
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Enrolled At"
                  value={selectedDevice.enrolled_at}
                  fullWidth
                  disabled
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  label="Last Seen"
                  value={selectedDevice.last_seen}
                  fullWidth
                  disabled
                />
              </Grid>
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>Close</Button>
          <Button variant="contained">Edit Device</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default DevicesPage;
