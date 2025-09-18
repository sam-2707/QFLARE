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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Switch,
  FormControlLabel,
  Alert,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Security as SecurityIcon,
  People as UsersIcon,
  Business as OrganizationIcon,
  Storage as DatabaseIcon,
  MonitorHeart as MonitoringIcon,
  Key as KeyIcon,
  Assignment as ReportsIcon,
  Backup as BackupIcon,
  Update as UpdateIcon,
  Warning as WarningIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Upload as UploadIcon
} from '@mui/icons-material';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`admin-tabpanel-${index}`}
      aria-labelledby={`admin-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

interface User {
  id: string;
  username: string;
  email: string;
  role: string;
  status: string;
  lastLogin: string;
  permissions: string[];
}

interface Organization {
  id: string;
  name: string;
  deviceCount: number;
  userCount: number;
  status: string;
  plan: string;
  createdAt: string;
}

interface SystemMetric {
  name: string;
  value: string;
  status: 'healthy' | 'warning' | 'error';
  description: string;
}

interface SecurityEvent {
  id: string;
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  timestamp: string;
  resolved: boolean;
}

const AdminPanel: React.FC = () => {
  const [currentTab, setCurrentTab] = useState(0);
  const [users, setUsers] = useState<User[]>([]);
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetric[]>([]);
  const [securityEvents, setSecurityEvents] = useState<SecurityEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [selectedItem, setSelectedItem] = useState<any>(null);

  // Mock data - replace with actual API calls
  useEffect(() => {
    const loadData = async () => {
      try {
        // Simulate API calls
        setUsers([
          {
            id: '1',
            username: 'admin',
            email: 'admin@company.com',
            role: 'System Admin',
            status: 'active',
            lastLogin: '2024-01-15T10:30:00Z',
            permissions: ['read:all', 'write:all', 'admin:all']
          },
          {
            id: '2',
            username: 'researcher1',
            email: 'researcher@company.com',
            role: 'Researcher',
            status: 'active',
            lastLogin: '2024-01-15T08:15:00Z',
            permissions: ['read:devices', 'write:training']
          }
        ]);

        setOrganizations([
          {
            id: 'org1',
            name: 'Research Institute A',
            deviceCount: 25,
            userCount: 8,
            status: 'active',
            plan: 'Enterprise',
            createdAt: '2023-12-01T00:00:00Z'
          },
          {
            id: 'org2',
            name: 'University B',
            deviceCount: 12,
            userCount: 5,
            status: 'active',
            plan: 'Academic',
            createdAt: '2024-01-01T00:00:00Z'
          }
        ]);

        setSystemMetrics([
          {
            name: 'API Response Time',
            value: '145ms',
            status: 'healthy',
            description: 'Average API response time'
          },
          {
            name: 'Database Connections',
            value: '45/100',
            status: 'healthy',
            description: 'Active database connections'
          },
          {
            name: 'Memory Usage',
            value: '78%',
            status: 'warning',
            description: 'System memory utilization'
          },
          {
            name: 'Active Devices',
            value: '37/50',
            status: 'healthy',
            description: 'Currently connected devices'
          }
        ]);

        setSecurityEvents([
          {
            id: '1',
            type: 'Failed Login',
            severity: 'medium',
            description: 'Multiple failed login attempts from IP 192.168.1.100',
            timestamp: '2024-01-15T10:25:00Z',
            resolved: false
          },
          {
            id: '2',
            type: 'Key Rotation',
            severity: 'low',
            description: 'Quantum key rotation completed successfully',
            timestamp: '2024-01-15T06:00:00Z',
            resolved: true
          }
        ]);

        setLoading(false);
      } catch (err) {
        setError('Failed to load admin data');
        setLoading(false);
      }
    };

    loadData();
  }, []);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'active':
        return 'success';
      case 'warning':
        return 'warning';
      case 'error':
      case 'critical':
        return 'error';
      default:
        return 'default';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low':
        return 'info';
      case 'medium':
        return 'warning';
      case 'high':
        return 'error';
      case 'critical':
        return 'error';
      default:
        return 'default';
    }
  };

  if (loading) {
    return (
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Typography>Loading admin panel...</Typography>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        QFLARE Admin Panel
      </Typography>

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={currentTab} onChange={handleTabChange} aria-label="admin tabs">
          <Tab label="Overview" icon={<SettingsIcon />} />
          <Tab label="Users" icon={<UsersIcon />} />
          <Tab label="Organizations" icon={<OrganizationIcon />} />
          <Tab label="Security" icon={<SecurityIcon />} />
          <Tab label="System Health" icon={<MonitoringIcon />} />
          <Tab label="Reports" icon={<ReportsIcon />} />
          <Tab label="Maintenance" icon={<UpdateIcon />} />
        </Tabs>
      </Box>

      {/* Overview Tab */}
      <TabPanel value={currentTab} index={0}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6} lg={3}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Total Users
                </Typography>
                <Typography variant="h4">
                  {users.length}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6} lg={3}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Organizations
                </Typography>
                <Typography variant="h4">
                  {organizations.length}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6} lg={3}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Active Devices
                </Typography>
                <Typography variant="h4">
                  37
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={6} lg={3}>
            <Card>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  Training Rounds
                </Typography>
                <Typography variant="h4">
                  256
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                System Metrics
              </Typography>
              <Grid container spacing={2}>
                {systemMetrics.map((metric, index) => (
                  <Grid item xs={12} sm={6} md={3} key={index}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h6">
                        {metric.value}
                      </Typography>
                      <Typography color="textSecondary" variant="body2">
                        {metric.name}
                      </Typography>
                      <Chip
                        size="small"
                        label={metric.status}
                        color={getStatusColor(metric.status) as any}
                      />
                    </Box>
                  </Grid>
                ))}
              </Grid>
            </Paper>
          </Grid>

          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Recent Security Events
              </Typography>
              <List>
                {securityEvents.slice(0, 3).map((event) => (
                  <ListItem key={event.id}>
                    <ListItemIcon>
                      {event.resolved ? (
                        <CheckIcon color="success" />
                      ) : (
                        <WarningIcon color="warning" />
                      )}
                    </ListItemIcon>
                    <ListItemText
                      primary={event.type}
                      secondary={`${event.description} - ${new Date(event.timestamp).toLocaleString()}`}
                    />
                    <Chip
                      size="small"
                      label={event.severity}
                      color={getSeverityColor(event.severity) as any}
                    />
                  </ListItem>
                ))}
              </List>
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Users Tab */}
      <TabPanel value={currentTab} index={1}>
        <Box sx={{ mb: 2 }}>
          <Button variant="contained" onClick={() => setDialogOpen(true)}>
            Add User
          </Button>
        </Box>
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Username</TableCell>
                <TableCell>Email</TableCell>
                <TableCell>Role</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Last Login</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {users.map((user) => (
                <TableRow key={user.id}>
                  <TableCell>{user.username}</TableCell>
                  <TableCell>{user.email}</TableCell>
                  <TableCell>{user.role}</TableCell>
                  <TableCell>
                    <Chip
                      label={user.status}
                      color={getStatusColor(user.status) as any}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    {new Date(user.lastLogin).toLocaleString()}
                  </TableCell>
                  <TableCell>
                    <Button size="small" onClick={() => {
                      setSelectedItem(user);
                      setDialogOpen(true);
                    }}>
                      Edit
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </TabPanel>

      {/* Organizations Tab */}
      <TabPanel value={currentTab} index={2}>
        <Box sx={{ mb: 2 }}>
          <Button variant="contained" onClick={() => setDialogOpen(true)}>
            Add Organization
          </Button>
        </Box>
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Devices</TableCell>
                <TableCell>Users</TableCell>
                <TableCell>Plan</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Created</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {organizations.map((org) => (
                <TableRow key={org.id}>
                  <TableCell>{org.name}</TableCell>
                  <TableCell>{org.deviceCount}</TableCell>
                  <TableCell>{org.userCount}</TableCell>
                  <TableCell>{org.plan}</TableCell>
                  <TableCell>
                    <Chip
                      label={org.status}
                      color={getStatusColor(org.status) as any}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    {new Date(org.createdAt).toLocaleDateString()}
                  </TableCell>
                  <TableCell>
                    <Button size="small" onClick={() => {
                      setSelectedItem(org);
                      setDialogOpen(true);
                    }}>
                      Edit
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </TabPanel>

      {/* Security Tab */}
      <TabPanel value={currentTab} index={3}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Quantum Key Management
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="textSecondary">
                  Last Rotation: 6 hours ago
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Next Rotation: In 18 hours
                </Typography>
              </Box>
              <Button
                variant="outlined"
                startIcon={<KeyIcon />}
                sx={{ mr: 1 }}
              >
                Force Rotation
              </Button>
              <Button
                variant="outlined"
                startIcon={<DownloadIcon />}
              >
                Export Keys
              </Button>
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Security Scans
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="textSecondary">
                  Last Scan: 2 hours ago
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Status: All Clear
                </Typography>
              </Box>
              <Button
                variant="outlined"
                startIcon={<SecurityIcon />}
                sx={{ mr: 1 }}
              >
                Run Scan
              </Button>
              <Button
                variant="outlined"
                startIcon={<DownloadIcon />}
              >
                Download Report
              </Button>
            </Paper>
          </Grid>

          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Security Events
              </Typography>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Type</TableCell>
                      <TableCell>Severity</TableCell>
                      <TableCell>Description</TableCell>
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
                        <TableCell>{event.description}</TableCell>
                        <TableCell>
                          {new Date(event.timestamp).toLocaleString()}
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={event.resolved ? 'Resolved' : 'Open'}
                            color={event.resolved ? 'success' : 'warning'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          {!event.resolved && (
                            <Button size="small">
                              Resolve
                            </Button>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>

      {/* System Health Tab */}
      <TabPanel value={currentTab} index={4}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="h6">
                  System Health Metrics
                </Typography>
                <IconButton onClick={() => window.location.reload()}>
                  <RefreshIcon />
                </IconButton>
              </Box>
              <Grid container spacing={2}>
                {systemMetrics.map((metric, index) => (
                  <Grid item xs={12} sm={6} key={index}>
                    <Card variant="outlined">
                      <CardContent>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                          <Typography variant="h6" sx={{ flexGrow: 1 }}>
                            {metric.name}
                          </Typography>
                          {metric.status === 'healthy' && <CheckIcon color="success" />}
                          {metric.status === 'warning' && <WarningIcon color="warning" />}
                          {metric.status === 'error' && <ErrorIcon color="error" />}
                        </Box>
                        <Typography variant="h4" color="primary">
                          {metric.value}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          {metric.description}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </Paper>
          </Grid>

          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              <List>
                <ListItem>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<RefreshIcon />}
                    onClick={() => alert('Restarting services...')}
                  >
                    Restart Services
                  </Button>
                </ListItem>
                <ListItem>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<BackupIcon />}
                    onClick={() => alert('Creating backup...')}
                  >
                    Create Backup
                  </Button>
                </ListItem>
                <ListItem>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<DatabaseIcon />}
                    onClick={() => alert('Checking database...')}
                  >
                    Check Database
                  </Button>
                </ListItem>
                <ListItem>
                  <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<SecurityIcon />}
                    onClick={() => alert('Running security scan...')}
                  >
                    Security Scan
                  </Button>
                </ListItem>
              </List>
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Reports Tab */}
      <TabPanel value={currentTab} index={5}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                System Reports
              </Typography>
              <List>
                <ListItem>
                  <ListItemText primary="Performance Report" secondary="Daily system performance metrics" />
                  <Button startIcon={<DownloadIcon />}>Download</Button>
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemText primary="Security Audit" secondary="Weekly security assessment" />
                  <Button startIcon={<DownloadIcon />}>Download</Button>
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemText primary="User Activity" secondary="Monthly user activity summary" />
                  <Button startIcon={<DownloadIcon />}>Download</Button>
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemText primary="Compliance Report" secondary="Quarterly compliance status" />
                  <Button startIcon={<DownloadIcon />}>Download</Button>
                </ListItem>
              </List>
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Custom Reports
              </Typography>
              <Box sx={{ mb: 2 }}>
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Report Type</InputLabel>
                  <Select value="" label="Report Type">
                    <MenuItem value="devices">Device Analytics</MenuItem>
                    <MenuItem value="training">Training Performance</MenuItem>
                    <MenuItem value="security">Security Events</MenuItem>
                    <MenuItem value="usage">Resource Usage</MenuItem>
                  </Select>
                </FormControl>
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Time Period</InputLabel>
                  <Select value="" label="Time Period">
                    <MenuItem value="day">Last 24 Hours</MenuItem>
                    <MenuItem value="week">Last Week</MenuItem>
                    <MenuItem value="month">Last Month</MenuItem>
                    <MenuItem value="quarter">Last Quarter</MenuItem>
                  </Select>
                </FormControl>
                <Button variant="contained" fullWidth>
                  Generate Report
                </Button>
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Maintenance Tab */}
      <TabPanel value={currentTab} index={6}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                System Maintenance
              </Typography>
              <List>
                <ListItem>
                  <ListItemIcon>
                    <UpdateIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Update System" 
                    secondary="Apply latest security patches and updates"
                  />
                  <Button variant="outlined">Update</Button>
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemIcon>
                    <BackupIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Backup Data" 
                    secondary="Create full system backup"
                  />
                  <Button variant="outlined">Backup</Button>
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemIcon>
                    <DatabaseIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Database Maintenance" 
                    secondary="Optimize database performance"
                  />
                  <Button variant="outlined">Optimize</Button>
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemIcon>
                    <SecurityIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Security Scan" 
                    secondary="Run comprehensive security assessment"
                  />
                  <Button variant="outlined">Scan</Button>
                </ListItem>
              </List>
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Configuration
              </Typography>
              <Box sx={{ mb: 2 }}>
                <FormControlLabel
                  control={<Switch defaultChecked />}
                  label="Auto Updates"
                />
              </Box>
              <Box sx={{ mb: 2 }}>
                <FormControlLabel
                  control={<Switch defaultChecked />}
                  label="Security Monitoring"
                />
              </Box>
              <Box sx={{ mb: 2 }}>
                <FormControlLabel
                  control={<Switch />}
                  label="Maintenance Mode"
                />
              </Box>
              <Box sx={{ mb: 2 }}>
                <FormControlLabel
                  control={<Switch defaultChecked />}
                  label="Automatic Backups"
                />
              </Box>
              <Button variant="contained" sx={{ mt: 2 }}>
                Save Configuration
              </Button>
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Edit Dialog */}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          {selectedItem ? 'Edit Item' : 'Add New Item'}
        </DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Name"
            type="text"
            fullWidth
            variant="outlined"
            defaultValue={selectedItem?.name || selectedItem?.username || ''}
          />
          <TextField
            margin="dense"
            label="Email"
            type="email"
            fullWidth
            variant="outlined"
            defaultValue={selectedItem?.email || ''}
          />
          {selectedItem?.role && (
            <FormControl fullWidth sx={{ mt: 2 }}>
              <InputLabel>Role</InputLabel>
              <Select value={selectedItem.role} label="Role">
                <MenuItem value="System Admin">System Admin</MenuItem>
                <MenuItem value="Researcher">Researcher</MenuItem>
                <MenuItem value="Device Manager">Device Manager</MenuItem>
                <MenuItem value="Viewer">Viewer</MenuItem>
              </Select>
            </FormControl>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={() => setDialogOpen(false)}>
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default AdminPanel;