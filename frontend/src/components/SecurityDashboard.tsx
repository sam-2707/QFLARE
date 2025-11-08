import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Skeleton,
  Snackbar,
  Button,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Timeline,
  TimelineItem,
  TimelineSeparator,
  TimelineConnector,
  TimelineContent,
  TimelineDot,
  TimelineOppositeContent,
} from '@mui/lab';
import {
  Security,
  Warning,
  CheckCircle,
  Block,
  Gavel,
  Refresh,
  CloudOff,
} from '@mui/icons-material';
import { authenticatedFetch, parseErrorResponse } from '../utils/api';

interface ByzantineStatus {
  total_nodes: number;
  suspicious_nodes: number;
  attacks_detected: number;
  attacks_blocked: number;
  detection_method: string;
  threshold: number;
  last_scan: string;
  security_level: string;
  suspicious_activities: Array<{
    node_id: string;
    timestamp: string;
    threat_level: string;
    reason: string;
  }>;
}

interface AuditEvent {
  id: string;
  timestamp: string;
  event_type: string;
  severity: string;
  description: string;
  user_id: number;
}

const SecurityDashboard: React.FC = () => {
  const [byzantineStatus, setByzantineStatus] = useState<ByzantineStatus | null>(null);
  const [auditLog, setAuditLog] = useState<AuditEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [showError, setShowError] = useState(false);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      fetchByzantineStatus();
      fetchAuditLog();
    };
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    await Promise.all([fetchByzantineStatus(), fetchAuditLog()]);
    setRefreshing(false);
  };

  useEffect(() => {
    fetchByzantineStatus();
    fetchAuditLog();
    
    // Auto-refresh every 10 seconds
    const interval = setInterval(() => {
      fetchByzantineStatus();
    }, 10000);
    
    return () => clearInterval(interval);
  }, []);

  const fetchByzantineStatus = async () => {
    try {
      const response = await authenticatedFetch('/api/security/byzantine-status', {
        timeout: 15000,
      });
      
      if (!response.ok) {
        const errorMsg = await parseErrorResponse(response);
        throw new Error(errorMsg);
      }
      
      const data = await response.json();
      setByzantineStatus(data);
      setError('');
    } catch (error: any) {
      console.error('Failed to fetch Byzantine status:', error);
      setError(error.message || 'Failed to load security status. Please try again.');
      setShowError(true);
    } finally {
      setLoading(false);
    }
  };

  const fetchAuditLog = async () => {
    try {
      const response = await authenticatedFetch('/api/security/audit-log?limit=20', {
        timeout: 15000,
      });
      
      if (!response.ok) {
        const errorMsg = await parseErrorResponse(response);
        throw new Error(errorMsg);
      }
      
      const data = await response.json();
      setAuditLog(data.events);
    } catch (error: any) {
      console.error('Failed to fetch audit log:', error);
      setError(error.message || 'Failed to load audit log. Please try again.');
      setShowError(true);
    }
  };

  const getSecurityLevelColor = (level?: string) => {
    if (!level) return '#000000';
    switch (level.toLowerCase()) {
      case 'protected':
        return '#000000'; // Black for protected
      case 'warning':
        return '#666666'; // Dark gray for warning
      case 'critical':
        return '#333333'; // Darker gray for critical
      default:
        return '#000000';
    }
  };

  const getThreatLevelColor = (level?: string) => {
    if (!level) return 'default';
    switch (level.toLowerCase()) {
      case 'low':
        return 'default';
      case 'medium':
        return 'default';
      case 'high':
        return 'default';
      default:
        return 'default';
    }
  };

  const getSeverityIcon = (severity?: string) => {
    if (!severity) return <Gavel sx={{ color: '#000' }} />;
    switch (severity.toLowerCase()) {
      case 'info':
        return <CheckCircle sx={{ color: '#000' }} />;
      case 'warning':
        return <Warning sx={{ color: '#000' }} />;
      case 'critical':
        return <Block sx={{ color: '#000' }} />;
      default:
        return <Gavel sx={{ color: '#000' }} />;
    }
  };

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
        <Skeleton variant="rectangular" height={400} sx={{ borderRadius: '12px' }} />
      </Box>
    );
  }

  if (!byzantineStatus) {
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
                fetchByzantineStatus();
                fetchAuditLog();
              }}
            >
              Retry
            </Button>
          }
        >
          No security data available. Please start a training session first.
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header with Offline Indicator and Refresh */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 600, color: '#000', fontFamily: 'Montserrat, sans-serif' }}>
          🛡️ Security & Byzantine Detection
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

      {/* Security Level Banner */}
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
              <Security sx={{ fontSize: 64 }} />
              <Box>
                <Typography variant="h3" sx={{ fontWeight: 700, fontFamily: 'Montserrat, sans-serif' }}>
                  {byzantineStatus?.security_level || 'UNKNOWN'}
                </Typography>
                <Typography variant="body1" sx={{ opacity: 0.9, mt: 1, fontFamily: 'Montserrat, sans-serif' }}>
                  System is actively monitoring for Byzantine attacks
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.8, mt: 0.5, fontFamily: 'Montserrat, sans-serif' }}>
                  Last scan: {byzantineStatus?.last_scan ? new Date(byzantineStatus.last_scan).toLocaleString() : 'N/A'}
                </Typography>
              </Box>
            </Box>
            <Box sx={{ textAlign: 'right' }}>
              <Typography variant="h6" sx={{ opacity: 0.9, fontFamily: 'Montserrat, sans-serif' }}>
                Detection Method
              </Typography>
              <Typography variant="body1" sx={{ fontWeight: 600, mt: 1, fontFamily: 'Montserrat, sans-serif' }}>
                {byzantineStatus?.detection_method || 'N/A'}
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.8, mt: 0.5, fontFamily: 'Montserrat, sans-serif' }}>
                Threshold: {byzantineStatus?.threshold || 'N/A'}
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* Statistics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card
            sx={{
              borderRadius: '12px',
              boxShadow: '0px 4px 12px rgba(0,0,0,0.08)',
              background: '#ffffff',
              color: '#000',
              border: '2px solid #000',
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: '0px 8px 24px rgba(0,0,0,0.12)',
              },
            }}
          >
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 600 }}>
                    Total Nodes
                  </Typography>
                  <Typography variant="h3" sx={{ fontWeight: 700, mt: 1, fontFamily: 'Montserrat, sans-serif' }}>
                    {byzantineStatus?.total_nodes || 0}
                  </Typography>
                </Box>
                <Security sx={{ fontSize: 48, opacity: 0.2 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card
            sx={{
              borderRadius: '12px',
              boxShadow: '0px 4px 12px rgba(0,0,0,0.08)',
              background: '#f5f5f5',
              color: '#000',
              border: '2px solid #000',
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: '0px 8px 24px rgba(0,0,0,0.12)',
              },
            }}
          >
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 600 }}>
                    Suspicious Nodes
                  </Typography>
                  <Typography variant="h3" sx={{ fontWeight: 700, mt: 1, fontFamily: 'Montserrat, sans-serif' }}>
                    {byzantineStatus?.suspicious_nodes || 0}
                  </Typography>
                </Box>
                <Warning sx={{ fontSize: 48, opacity: 0.2 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card
            sx={{
              borderRadius: '12px',
              boxShadow: '0px 4px 12px rgba(0,0,0,0.08)',
              background: '#e0e0e0',
              color: '#000',
              border: '2px solid #000',
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: '0px 8px 24px rgba(0,0,0,0.12)',
              },
            }}
          >
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 600 }}>
                    Attacks Detected
                  </Typography>
                  <Typography variant="h3" sx={{ fontWeight: 700, mt: 1, fontFamily: 'Montserrat, sans-serif' }}>
                    {byzantineStatus?.attacks_detected || 0}
                  </Typography>
                </Box>
                <Block sx={{ fontSize: 48, opacity: 0.2 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card
            sx={{
              borderRadius: '12px',
              boxShadow: '0px 4px 12px rgba(0,0,0,0.08)',
              background: '#ffffff',
              color: '#000',
              border: '2px solid #000',
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: '0px 8px 24px rgba(0,0,0,0.12)',
              },
            }}
          >
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif', fontWeight: 600 }}>
                    Attacks Blocked
                  </Typography>
                  <Typography variant="h3" sx={{ fontWeight: 700, mt: 1, fontFamily: 'Montserrat, sans-serif' }}>
                    {byzantineStatus?.attacks_blocked || 0}
                  </Typography>
                </Box>
                <CheckCircle sx={{ fontSize: 48, opacity: 0.2 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Suspicious Activities Alert */}
      {byzantineStatus?.suspicious_activities && byzantineStatus.suspicious_activities.length > 0 && (
        <Alert 
          severity="warning" 
          sx={{ 
            mb: 3, 
            borderRadius: '12px',
            border: '2px solid #000',
            backgroundColor: '#f5f5f5',
            color: '#000',
            '& .MuiAlert-icon': {
              color: '#000'
            }
          }}
        >
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 1, fontFamily: 'Montserrat, sans-serif' }}>
            ⚠️ Active Threats Detected
          </Typography>
          <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif' }}>
            {byzantineStatus.suspicious_activities.length} suspicious activit{byzantineStatus.suspicious_activities.length === 1 ? 'y' : 'ies'} detected in the last scan. Review details below.
          </Typography>
        </Alert>
      )}

      {/* Suspicious Activities Table */}
      {byzantineStatus?.suspicious_activities && byzantineStatus.suspicious_activities.length > 0 && (
        <Card sx={{ mb: 3, borderRadius: '12px', boxShadow: '0px 4px 12px rgba(0,0,0,0.08)', border: '2px solid #000' }}>
          <CardContent>
            <Typography variant="h6" sx={{ fontWeight: 600, mb: 2, fontFamily: 'Montserrat, sans-serif' }}>
              🚨 Suspicious Activities
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow sx={{ backgroundColor: '#f5f5f5' }}>
                    <TableCell sx={{ fontWeight: 600, fontFamily: 'Montserrat, sans-serif' }}>Node ID</TableCell>
                    <TableCell sx={{ fontWeight: 600, fontFamily: 'Montserrat, sans-serif' }}>Timestamp</TableCell>
                    <TableCell sx={{ fontWeight: 600, fontFamily: 'Montserrat, sans-serif' }}>Threat Level</TableCell>
                    <TableCell sx={{ fontWeight: 600, fontFamily: 'Montserrat, sans-serif' }}>Reason</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {byzantineStatus.suspicious_activities.map((activity, idx) => (
                    <TableRow key={`${activity.node_id}-${activity.timestamp}-${idx}`}>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 600 }}>
                          {activity.node_id}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif' }}>
                          {new Date(activity.timestamp).toLocaleString()}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={activity.threat_level}
                          size="small"
                          sx={{ 
                            fontWeight: 600,
                            fontFamily: 'Montserrat, sans-serif',
                            backgroundColor: '#000',
                            color: '#fff',
                            border: '1px solid #000'
                          }}
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif' }}>{activity.reason}</Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}

      {/* Security Audit Log */}
      <Card sx={{ borderRadius: '12px', boxShadow: '0px 4px 12px rgba(0,0,0,0.08)', border: '2px solid #000' }}>
        <CardContent>
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 3, fontFamily: 'Montserrat, sans-serif' }}>
            📋 Security Audit Log
          </Typography>
          {!auditLog || auditLog.length === 0 ? (
            <Typography variant="body2" color="textSecondary" sx={{ fontFamily: 'Montserrat, sans-serif' }}>
              No audit events recorded
            </Typography>
          ) : (
            <Timeline position="right">
              {auditLog.map((event, index) => (
                <TimelineItem key={event.id}>
                  <TimelineOppositeContent color="textSecondary" sx={{ flex: 0.3 }}>
                    <Typography variant="body2" sx={{ fontFamily: 'Montserrat, sans-serif' }}>
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </Typography>
                    <Typography variant="caption" sx={{ fontFamily: 'Montserrat, sans-serif' }}>
                      {new Date(event.timestamp).toLocaleDateString()}
                    </Typography>
                  </TimelineOppositeContent>
                  <TimelineSeparator>
                    <TimelineDot sx={{ backgroundColor: '#000', color: '#fff' }}>
                      {getSeverityIcon(event.severity)}
                    </TimelineDot>
                    {index < auditLog.length - 1 && <TimelineConnector sx={{ backgroundColor: '#000' }} />}
                  </TimelineSeparator>
                  <TimelineContent>
                    <Card
                      sx={{
                        p: 2,
                        borderRadius: '8px',
                        border: '2px solid #000',
                        backgroundColor: '#ffffff',
                      }}
                    >
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 1 }}>
                        <Typography variant="body2" sx={{ fontWeight: 600, fontFamily: 'Montserrat, sans-serif' }}>
                          {event.event_type.replace(/_/g, ' ').toUpperCase()}
                        </Typography>
                        <Chip
                          label={event.severity}
                          size="small"
                          sx={{
                            fontFamily: 'Montserrat, sans-serif',
                            backgroundColor: '#000',
                            color: '#fff',
                            fontWeight: 600,
                          }}
                        />
                      </Box>
                      <Typography variant="body2" color="textSecondary" sx={{ fontFamily: 'Montserrat, sans-serif' }}>
                        {event.description}
                      </Typography>
                    </Card>
                  </TimelineContent>
                </TimelineItem>
              ))}
            </Timeline>
          )}
        </CardContent>
      </Card>

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

export default SecurityDashboard;
