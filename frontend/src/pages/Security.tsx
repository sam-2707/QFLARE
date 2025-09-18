import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
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
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  LinearProgress,
  Alert,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  Security as SecurityIcon,
  Key as KeyIcon,
  Shield as ShieldIcon,
  Warning as WarningIcon,
  CheckCircle as CheckIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Visibility as VisibilityIcon
} from '@mui/icons-material';

interface SecurityEvent {
  id: string;
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: string;
  status: string;
}

interface SecurityMetric {
  name: string;
  value: string;
  status: 'healthy' | 'warning' | 'critical';
  lastUpdate: string;
}

const Security: React.FC = () => {
  const [securityEvents, setSecurityEvents] = useState<SecurityEvent[]>([]);
  const [securityMetrics, setSecurityMetrics] = useState<SecurityMetric[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Mock data - replace with actual API calls
    setSecurityEvents([
      {
        id: '1',
        type: 'Failed Authentication',
        severity: 'medium',
        message: 'Multiple failed login attempts detected from IP 192.168.1.100',
        timestamp: '2024-01-15T10:30:00Z',
        status: 'investigating'
      },
      {
        id: '2',
        type: 'Quantum Key Rotation',
        severity: 'low',
        message: 'Scheduled quantum key rotation completed successfully',
        timestamp: '2024-01-15T06:00:00Z',
        status: 'resolved'
      },
      {
        id: '3',
        type: 'Certificate Expiration',
        severity: 'high',
        message: 'Device certificate for DEV-001 expires in 7 days',
        timestamp: '2024-01-15T09:15:00Z',
        status: 'open'
      }
    ]);

    setSecurityMetrics([
      {
        name: 'Quantum Key Health',
        value: '98%',
        status: 'healthy',
        lastUpdate: '2024-01-15T10:00:00Z'
      },
      {
        name: 'Certificate Validity',
        value: '94%',
        status: 'warning',
        lastUpdate: '2024-01-15T10:00:00Z'
      },
      {
        name: 'Encryption Status',
        value: '100%',
        status: 'healthy',
        lastUpdate: '2024-01-15T10:00:00Z'
      },
      {
        name: 'Access Control',
        value: '99%',
        status: 'healthy',
        lastUpdate: '2024-01-15T10:00:00Z'
      }
    ]);

    setLoading(false);
  }, []);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low': return 'info';
      case 'medium': return 'warning';
      case 'high': return 'error';
      case 'critical': return 'error';
      default: return 'default';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'warning': return 'warning';
      case 'critical': return 'error';
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
      <Typography variant="h4" gutterBottom>
        Security Dashboard
      </Typography>

      {/* Security Overview */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {securityMetrics.map((metric, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <SecurityIcon sx={{ mr: 1 }} />
                  <Typography variant="h6" sx={{ flexGrow: 1 }}>
                    {metric.name}
                  </Typography>
                  {metric.status === 'healthy' && <CheckIcon color="success" />}
                  {metric.status === 'warning' && <WarningIcon color="warning" />}
                </Box>
                <Typography variant="h4" color="primary" gutterBottom>
                  {metric.value}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Last update: {new Date(metric.lastUpdate).toLocaleString()}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Grid container spacing={3}>
        {/* Quantum Key Management */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <KeyIcon sx={{ mr: 1 }} />
              <Typography variant="h6" sx={{ flexGrow: 1 }}>
                Quantum Key Management
              </Typography>
              <Tooltip title="Refresh key status">
                <IconButton>
                  <RefreshIcon />
                </IconButton>
              </Tooltip>
            </Box>
            
            <Grid container spacing={2} sx={{ mb: 2 }}>
              <Grid item xs={6}>
                <Typography variant="body2" color="textSecondary">
                  Last Rotation
                </Typography>
                <Typography variant="body1">
                  6 hours ago
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body2" color="textSecondary">
                  Next Rotation
                </Typography>
                <Typography variant="body1">
                  In 18 hours
                </Typography>
              </Grid>
            </Grid>

            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" gutterBottom>
                Key Rotation Progress
              </Typography>
              <LinearProgress variant="determinate" value={75} sx={{ mb: 1 }} />
              <Typography variant="body2" color="textSecondary">
                75% of devices updated
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button variant="outlined" startIcon={<KeyIcon />} size="small">
                Force Rotation
              </Button>
              <Button variant="outlined" startIcon={<DownloadIcon />} size="small">
                Export Keys
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* Security Scan Results */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <ShieldIcon sx={{ mr: 1 }} />
              <Typography variant="h6" sx={{ flexGrow: 1 }}>
                Security Scan Results
              </Typography>
              <Tooltip title="View detailed report">
                <IconButton>
                  <VisibilityIcon />
                </IconButton>
              </Tooltip>
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Last Scan: 2 hours ago
              </Typography>
              <Alert severity="success" sx={{ mb: 2 }}>
                No critical vulnerabilities detected
              </Alert>
            </Box>

            <List dense>
              <ListItem>
                <ListItemIcon>
                  <CheckIcon color="success" />
                </ListItemIcon>
                <ListItemText
                  primary="Encryption Protocols"
                  secondary="All communications encrypted"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckIcon color="success" />
                </ListItemIcon>
                <ListItemText
                  primary="Access Controls"
                  secondary="RBAC properly configured"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <WarningIcon color="warning" />
                </ListItemIcon>
                <ListItemText
                  primary="Certificate Expiration"
                  secondary="3 certificates expire within 30 days"
                />
              </ListItem>
            </List>

            <Button variant="outlined" startIcon={<ShieldIcon />} fullWidth>
              Run Full Security Scan
            </Button>
          </Paper>
        </Grid>

        {/* Security Events */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Security Events
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Type</TableCell>
                    <TableCell>Severity</TableCell>
                    <TableCell>Message</TableCell>
                    <TableCell>Timestamp</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {securityEvents.map((event) => (
                    <TableRow key={event.id}>
                      <TableCell>{event.type}</TableCell>
                      <TableCell>
                        <Chip
                          label={event.severity}
                          color={getSeverityColor(event.severity) as any}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{event.message}</TableCell>
                      <TableCell>
                        {new Date(event.timestamp).toLocaleString()}
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={event.status}
                          color={getStatusColor(event.status) as any}
                          size="small"
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell>
                        <Button size="small">
                          Investigate
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Security;