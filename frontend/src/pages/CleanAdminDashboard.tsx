import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Alert,
  Tabs,
  Tab,
  Badge,
  List,
  ListItem,
  ListItemText,
  LinearProgress,
} from '@mui/material';
import {
  Check as CheckIcon,
  Close as CloseIcon,
  Refresh as RefreshIcon,
  Key as KeyIcon,
  People as PeopleIcon,
  Security as SecurityIcon,
  Monitor as MonitorIcon,
  Download as DownloadIcon,
  Visibility as ViewIcon,
} from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';

interface User {
  id: string;
  username: string;
  email: string;
  name: string;
  role: string;
  is_verified: boolean;
  created_at: string;
}

interface SystemStats {
  totalUsers: number;
  pendingUsers: number;
  activeUsers: number;
  totalKeys: number;
}

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

const CleanAdminDashboard: React.FC = () => {
  const { user } = useAuth();
  const [users, setUsers] = useState<User[]>([]);
  const [stats, setStats] = useState<SystemStats>({
    totalUsers: 0,
    pendingUsers: 0,
    activeUsers: 0,
    totalKeys: 0,
  });
  const [loading, setLoading] = useState(true);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [tabValue, setTabValue] = useState(0);
  const [adminNotes, setAdminNotes] = useState('');

  useEffect(() => {
    if (user?.role === 'admin') {
      loadAdminData();
    }
  }, [user]);

  const loadAdminData = async () => {
    try {
      const token = localStorage.getItem('qflare-token');
      
      // Load users
      const usersResponse = await fetch('http://localhost:8002/admin/users', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      // Load stats
      const statsResponse = await fetch('http://localhost:8002/admin/stats', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (usersResponse.ok) {
        const usersData = await usersResponse.json();
        setUsers(usersData.users || []);
      }

      if (statsResponse.ok) {
        const statsData = await statsResponse.json();
        setStats(statsData);
      }
    } catch (error) {
      console.error('Error loading admin data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleApproveUser = async (userId: string) => {
    try {
      const token = localStorage.getItem('qflare-token');
      const response = await fetch(`http://localhost:8002/admin/users/${userId}/approve`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          notes: adminNotes,
        }),
      });

      if (response.ok) {
        await loadAdminData(); // Refresh data
        setDialogOpen(false);
        setAdminNotes('');
        setSelectedUser(null);
      }
    } catch (error) {
      console.error('Error approving user:', error);
    }
  };

  const handleRejectUser = async (userId: string) => {
    try {
      const token = localStorage.getItem('qflare-token');
      const response = await fetch(`http://localhost:8002/admin/users/${userId}/reject`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          notes: adminNotes,
        }),
      });

      if (response.ok) {
        await loadAdminData(); // Refresh data
        setDialogOpen(false);
        setAdminNotes('');
        setSelectedUser(null);
      }
    } catch (error) {
      console.error('Error rejecting user:', error);
    }
  };

  const handleGenerateKeys = async (userId: string) => {
    try {
      const token = localStorage.getItem('qflare-token');
      const response = await fetch(`http://localhost:8002/admin/users/${userId}/generate-keys`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        await loadAdminData(); // Refresh data
        alert('Keys generated successfully!');
      }
    } catch (error) {
      console.error('Error generating keys:', error);
    }
  };

  const openApprovalDialog = (user: User) => {
    setSelectedUser(user);
    setDialogOpen(true);
  };

  if (user?.role !== 'admin') {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">
          Access denied. Admin privileges required.
        </Alert>
      </Box>
    );
  }

  if (loading) {
    return (
      <Box sx={{ width: '100%', mt: 2 }}>
        <LinearProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Admin Header */}
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            Admin Dashboard
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Manage users, approve registrations, and monitor system security.
          </Typography>
        </Box>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={loadAdminData}
        >
          Refresh
        </Button>
      </Box>

      {/* Statistics Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <PeopleIcon sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
              <Typography variant="h4" component="div">
                {stats.totalUsers}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Users
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Badge badgeContent={stats.pendingUsers} color="warning">
                <SecurityIcon sx={{ fontSize: 40, color: 'warning.main', mb: 1 }} />
              </Badge>
              <Typography variant="h4" component="div">
                {stats.pendingUsers}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Pending Approval
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <CheckIcon sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
              <Typography variant="h4" component="div">
                {stats.activeUsers}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Active Users
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <KeyIcon sx={{ fontSize: 40, color: 'info.main', mb: 1 }} />
              <Typography variant="h4" component="div">
                {stats.totalKeys}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Keys Generated
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Main Content Tabs */}
      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={(e, newValue) => setTabValue(newValue)}>
            <Tab label={`Pending Approvals (${stats.pendingUsers})`} />
            <Tab label="All Users" />
            <Tab label="System Monitor" />
          </Tabs>
        </Box>

        {/* Pending Approvals Tab */}
        <TabPanel value={tabValue} index={0}>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>User</TableCell>
                  <TableCell>Email</TableCell>
                  <TableCell>Registration Date</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {users.filter(u => !u.is_verified).map((user) => (
                  <TableRow key={user.id}>
                    <TableCell>
                      <Box>
                        <Typography variant="subtitle2">{user.name || user.username}</Typography>
                        <Typography variant="caption" color="text.secondary">
                          @{user.username}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>{user.email}</TableCell>
                    <TableCell>
                      {new Date(user.created_at).toLocaleDateString()}
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <IconButton
                          size="small"
                          color="success"
                          onClick={() => openApprovalDialog(user)}
                        >
                          <CheckIcon />
                        </IconButton>
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => openApprovalDialog(user)}
                        >
                          <CloseIcon />
                        </IconButton>
                        <IconButton size="small" color="primary">
                          <ViewIcon />
                        </IconButton>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          {users.filter(u => !u.is_verified).length === 0 && (
            <Alert severity="info" sx={{ mt: 2 }}>
              No pending user approvals.
            </Alert>
          )}
        </TabPanel>

        {/* All Users Tab */}
        <TabPanel value={tabValue} index={1}>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>User</TableCell>
                  <TableCell>Email</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Keys</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {users.map((user) => (
                  <TableRow key={user.id}>
                    <TableCell>
                      <Box>
                        <Typography variant="subtitle2">{user.name || user.username}</Typography>
                        <Chip
                          label={user.role}
                          size="small"
                          color={user.role === 'admin' ? 'primary' : 'default'}
                        />
                      </Box>
                    </TableCell>
                    <TableCell>{user.email}</TableCell>
                    <TableCell>
                      <Chip
                        label={user.is_verified ? 'Verified' : 'Pending'}
                        size="small"
                        color={user.is_verified ? 'success' : 'warning'}
                      />
                    </TableCell>
                    <TableCell>
                      <Button
                        size="small"
                        variant="outlined"
                        startIcon={<KeyIcon />}
                        onClick={() => handleGenerateKeys(user.id)}
                        disabled={!user.is_verified}
                      >
                        Generate
                      </Button>
                    </TableCell>
                    <TableCell>
                      <IconButton size="small" color="primary">
                        <ViewIcon />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>

        {/* System Monitor Tab */}
        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  System Health
                </Typography>
                <List>
                  <ListItem>
                    <ListItemText
                      primary="Database Connection"
                      secondary="Connected"
                    />
                    <Chip label="Online" color="success" size="small" />
                  </ListItem>
                  <ListItem>
                    <ListItemText
                      primary="Key Management Service"
                      secondary="Operational"
                    />
                    <Chip label="Active" color="success" size="small" />
                  </ListItem>
                  <ListItem>
                    <ListItemText
                      primary="Authentication Service"
                      secondary="Running"
                    />
                    <Chip label="Online" color="success" size="small" />
                  </ListItem>
                </List>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Security Logs
                </Typography>
                <Alert severity="info" sx={{ mb: 2 }}>
                  Recent security events and system activities.
                </Alert>
                <List dense>
                  <ListItem>
                    <ListItemText
                      primary="User registration: john@example.com"
                      secondary="2 minutes ago"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText
                      primary="Keys generated for user ID: 123"
                      secondary="15 minutes ago"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText
                      primary="Admin login from IP: 192.168.1.100"
                      secondary="1 hour ago"
                    />
                  </ListItem>
                </List>
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>
      </Card>

      {/* Approval Dialog */}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          User Approval - {selectedUser?.name || selectedUser?.username}
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Email: {selectedUser?.email}
          </Typography>
          <TextField
            fullWidth
            multiline
            rows={3}
            label="Admin Notes (optional)"
            value={adminNotes}
            onChange={(e) => setAdminNotes(e.target.value)}
            placeholder="Add any notes about this approval/rejection..."
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={() => selectedUser && handleRejectUser(selectedUser.id)}
            color="error"
            startIcon={<CloseIcon />}
          >
            Reject
          </Button>
          <Button
            onClick={() => selectedUser && handleApproveUser(selectedUser.id)}
            color="success"
            startIcon={<CheckIcon />}
            variant="contained"
          >
            Approve
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default CleanAdminDashboard;