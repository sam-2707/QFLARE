import React, { useState, useEffect } from 'react';
import { Box, Typography, Paper, Grid, Button, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Chip, Card, CardContent, IconButton, Dialog, DialogTitle, DialogContent, DialogActions, Alert } from '@mui/material';
import { useAuth } from '../contexts/AuthContext';
import { People, Security, Settings, Dashboard as DashboardIcon, CheckCircle, Pending, Block, Delete, VpnKey } from '@mui/icons-material';

interface PendingUser {
  id: string;
  email: string;
  username: string;
  full_name?: string;
  created_at: string;
}

interface User {
  id: string;
  username: string;
  email: string;
  full_name: string | null;
  role: string;
  is_active: boolean;
  is_verified: boolean;
  has_keys: boolean;
  created_at: string | null;
  last_login: string | null;
}

interface DashboardStats {
  totalUsers: number;
  activeUsers: number;
  pendingUsers: number;
  adminCount: number;
  userCount: number;
  recentRegistrations: number;
  totalKeys: number;
  todayLogins: number;
}

const AdminDashboard: React.FC = () => {
  const { user, logout } = useAuth();
  const [pendingUsers, setPendingUsers] = useState<PendingUser[]>([]);
  const [allUsers, setAllUsers] = useState<User[]>([]);
  const [stats, setStats] = useState<DashboardStats>({
    totalUsers: 0,
    activeUsers: 0,
    pendingUsers: 0,
    adminCount: 0,
    userCount: 0,
    recentRegistrations: 0,
    totalKeys: 0,
    todayLogins: 0
  });
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [userToDelete, setUserToDelete] = useState<User | null>(null);

  useEffect(() => {
    fetchPendingUsers();
    fetchStats();
    fetchAllUsers();
  }, []);

  const fetchPendingUsers = async () => {
    try {
      const token = localStorage.getItem('qflare-token');
      const response = await fetch('http://localhost:8002/api/admin/pending-users', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      if (response.ok) {
        const data = await response.json();
        setPendingUsers(data.pending_users || []);
      }
    } catch (error) {
      console.error('Error fetching pending users:', error);
    }
  };

  const fetchStats = async () => {
    try {
      const token = localStorage.getItem('qflare-token');
      const response = await fetch('http://localhost:8002/api/admin/dashboard-stats', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Error fetching stats:', error);
      // Fallback to mock stats
      setStats({
        totalUsers: 2,
        activeUsers: 2,
        pendingUsers: 0,
        adminCount: 1,
        userCount: 1,
        recentRegistrations: 0,
        totalKeys: 0,
        todayLogins: 0
      });
    }
  };

  const fetchAllUsers = async () => {
    try {
      const token = localStorage.getItem('qflare-token');
      const response = await fetch('http://localhost:8002/api/admin/all-users', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      if (response.ok) {
        const data = await response.json();
        setAllUsers(data.users || []);
      }
    } catch (error) {
      console.error('Error fetching all users:', error);
    }
  };

  const approveUser = async (userId: string) => {
    try {
      const token = localStorage.getItem('qflare-token');
      const response = await fetch(`http://localhost:8002/api/admin/approve-user/${userId}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      if (response.ok) {
        setSuccess('User approved successfully');
        fetchPendingUsers();
        fetchStats();
        fetchAllUsers();
        setTimeout(() => setSuccess(''), 3000);
      } else {
        setError('Failed to approve user');
      }
    } catch (error) {
      console.error('Error approving user:', error);
      setError('Error approving user');
    }
  };

  const deleteUser = async () => {
    if (!userToDelete) return;

    try {
      const token = localStorage.getItem('qflare-token');
      const response = await fetch(`http://localhost:8002/api/admin/user/${userToDelete.id}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        setSuccess('User deleted successfully');
        fetchAllUsers();
        fetchStats();
        setDeleteDialogOpen(false);
        setUserToDelete(null);
        setTimeout(() => setSuccess(''), 3000);
      } else {
        setError('Failed to delete user');
      }
    } catch (err) {
      console.error('Error deleting user:', err);
      setError('Error deleting user');
    }
  };

  const handleDeleteClick = (userToDelete: User) => {
    setUserToDelete(userToDelete);
    setDeleteDialogOpen(true);
  };

  return (
    <Box sx={{ p: 4, minHeight: '100vh', backgroundColor: '#ffffff' }}>
      {/* Header */}
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '2px solid #000', pb: 2 }}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom sx={{ color: '#000000', fontFamily: 'monospace', fontWeight: 700 }}>
            QFLARE ADMIN DASHBOARD
          </Typography>
          <Typography variant="body1" sx={{ color: '#666', fontFamily: 'monospace' }}>
            Welcome, {user?.username || user?.email} | Role: ADMINISTRATOR
          </Typography>
        </Box>
        <Button 
          variant="outlined" 
          onClick={logout}
          sx={{ 
            color: '#000', 
            borderColor: '#000',
            borderRadius: 0,
            fontFamily: 'monospace',
            fontWeight: 600,
            '&:hover': {
              borderColor: '#000',
              backgroundColor: '#f5f5f5'
            }
          }}
        >
          LOGOUT
        </Button>
      </Box>

      {/* Alerts */}
      {error && <Alert severity="error" sx={{ mb: 2, borderRadius: 0 }}>{error}</Alert>}
      {success && <Alert severity="success" sx={{ mb: 2, borderRadius: 0 }}>{success}</Alert>}

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={3}>
          <Card sx={{ border: '2px solid #000', borderRadius: 0, bgcolor: '#e6f2ff' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <People sx={{ fontSize: 40, color: '#000', mr: 2 }} />
                <Box>
                  <Typography variant="h4" sx={{ fontFamily: 'monospace', fontWeight: 700 }}>
                    {stats.totalUsers}
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', color: '#666' }}>
                    Total Users
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card sx={{ border: '2px solid #000', borderRadius: 0, bgcolor: '#e6ffe6' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <CheckCircle sx={{ fontSize: 40, color: '#2e7d32', mr: 2 }} />
                <Box>
                  <Typography variant="h4" sx={{ fontFamily: 'monospace', fontWeight: 700 }}>
                    {stats.activeUsers}
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', color: '#666' }}>
                    Active Users
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card sx={{ border: '2px solid #000', borderRadius: 0, bgcolor: '#fff2e6' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Pending sx={{ fontSize: 40, color: '#ed6c02', mr: 2 }} />
                <Box>
                  <Typography variant="h4" sx={{ fontFamily: 'monospace', fontWeight: 700 }}>
                    {stats.pendingUsers}
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', color: '#666' }}>
                    Pending Approvals
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card sx={{ border: '2px solid #000', borderRadius: 0, bgcolor: '#ffe6f2' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <VpnKey sx={{ fontSize: 40, color: '#1976d2', mr: 2 }} />
                <Box>
                  <Typography variant="h4" sx={{ fontFamily: 'monospace', fontWeight: 700 }}>
                    {stats.totalKeys}
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', color: '#666' }}>
                    Total Keys
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Admin Actions */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, backgroundColor: '#000', color: '#fff', borderRadius: 0, border: '2px solid #000' }}>
            <Security sx={{ fontSize: 40, mb: 2 }} />
            <Typography variant="h6" gutterBottom sx={{ fontFamily: 'monospace', fontWeight: 600 }}>
              SECURITY MANAGEMENT
            </Typography>
            <Typography variant="body2" sx={{ fontFamily: 'monospace', mb: 2 }}>
              {stats.adminCount} admins, {stats.userCount} regular users
            </Typography>
            <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
              Post-quantum keys: {stats.totalKeys}
            </Typography>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, backgroundColor: '#fff', color: '#000', borderRadius: 0, border: '2px solid #000' }}>
            <People sx={{ fontSize: 40, mb: 2 }} />
            <Typography variant="h6" gutterBottom sx={{ fontFamily: 'monospace', fontWeight: 600 }}>
              USER MANAGEMENT
            </Typography>
            <Typography variant="body2" sx={{ fontFamily: 'monospace', mb: 2 }}>
              Recent registrations: {stats.recentRegistrations}
            </Typography>
            <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
              Active sessions: {stats.todayLogins}
            </Typography>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, backgroundColor: '#fff', color: '#000', borderRadius: 0, border: '2px solid #000' }}>
            <Settings sx={{ fontSize: 40, mb: 2 }} />
            <Typography variant="h6" gutterBottom sx={{ fontFamily: 'monospace', fontWeight: 600 }}>
              SYSTEM STATUS
            </Typography>
            <Typography variant="body2" sx={{ fontFamily: 'monospace', mb: 2 }}>
              All systems operational
            </Typography>
            <Chip 
              label="ONLINE" 
              sx={{ 
                backgroundColor: '#e6ffe6', 
                color: '#2e7d32',
                fontFamily: 'monospace',
                border: '1px solid #2e7d32',
                borderRadius: 0,
                fontWeight: 600
              }} 
            />
          </Paper>
        </Grid>
      </Grid>

      {/* Pending Users Table */}
      {pendingUsers.length > 0 && (
        <Paper sx={{ p: 3, backgroundColor: '#fff', borderRadius: 0, border: '2px solid #000', mb: 4 }}>
          <Typography variant="h6" gutterBottom sx={{ fontFamily: 'monospace', fontWeight: 600, mb: 2 }}>
            PENDING USER APPROVALS ({pendingUsers.length})
          </Typography>
          <TableContainer>
            <Table sx={{ fontFamily: 'monospace' }}>
              <TableHead>
                <TableRow sx={{ backgroundColor: '#000' }}>
                  <TableCell sx={{ color: '#fff', fontFamily: 'monospace', fontWeight: 600 }}>USERNAME</TableCell>
                  <TableCell sx={{ color: '#fff', fontFamily: 'monospace', fontWeight: 600 }}>EMAIL</TableCell>
                  <TableCell sx={{ color: '#fff', fontFamily: 'monospace', fontWeight: 600 }}>FULL NAME</TableCell>
                  <TableCell sx={{ color: '#fff', fontFamily: 'monospace', fontWeight: 600 }}>REGISTERED</TableCell>
                  <TableCell sx={{ color: '#fff', fontFamily: 'monospace', fontWeight: 600 }}>STATUS</TableCell>
                  <TableCell sx={{ color: '#fff', fontFamily: 'monospace', fontWeight: 600 }}>ACTIONS</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {pendingUsers.map((pendingUser) => (
                  <TableRow key={pendingUser.id} hover>
                    <TableCell sx={{ fontFamily: 'monospace' }}>{pendingUser.username}</TableCell>
                    <TableCell sx={{ fontFamily: 'monospace' }}>{pendingUser.email}</TableCell>
                    <TableCell sx={{ fontFamily: 'monospace' }}>{pendingUser.full_name || 'N/A'}</TableCell>
                    <TableCell sx={{ fontFamily: 'monospace' }}>
                      {new Date(pendingUser.created_at).toLocaleDateString()}
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label="PENDING" 
                        sx={{ 
                          backgroundColor: '#fff3cd', 
                          color: '#000',
                          fontFamily: 'monospace',
                          border: '1px solid #000',
                          borderRadius: 0
                        }} 
                      />
                    </TableCell>
                    <TableCell>
                      <Button
                        variant="contained"
                        size="small"
                        onClick={() => approveUser(pendingUser.id)}
                        sx={{
                          backgroundColor: '#2e7d32',
                          color: '#fff',
                          fontFamily: 'monospace',
                          borderRadius: 0,
                          mr: 1,
                          '&:hover': {
                            backgroundColor: '#1b5e20'
                          }
                        }}
                      >
                        APPROVE
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}

      {/* All Users Table */}
      <Paper sx={{ p: 3, backgroundColor: '#fff', borderRadius: 0, border: '2px solid #000' }}>
        <Typography variant="h6" gutterBottom sx={{ fontFamily: 'monospace', fontWeight: 600, mb: 2 }}>
          ALL USERS ({allUsers.length})
        </Typography>
        <TableContainer>
          <Table sx={{ fontFamily: 'monospace' }}>
            <TableHead>
              <TableRow sx={{ backgroundColor: '#f0f0f0' }}>
                <TableCell sx={{ fontFamily: 'monospace', fontWeight: 600 }}>USERNAME</TableCell>
                <TableCell sx={{ fontFamily: 'monospace', fontWeight: 600 }}>EMAIL</TableCell>
                <TableCell sx={{ fontFamily: 'monospace', fontWeight: 600 }}>ROLE</TableCell>
                <TableCell sx={{ fontFamily: 'monospace', fontWeight: 600 }}>STATUS</TableCell>
                <TableCell sx={{ fontFamily: 'monospace', fontWeight: 600 }}>KEYS</TableCell>
                <TableCell sx={{ fontFamily: 'monospace', fontWeight: 600 }}>LAST LOGIN</TableCell>
                <TableCell sx={{ fontFamily: 'monospace', fontWeight: 600 }}>ACTIONS</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {allUsers.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} align="center" sx={{ fontFamily: 'monospace', py: 4 }}>
                    No users found
                  </TableCell>
                </TableRow>
              ) : (
                allUsers.map((u) => (
                  <TableRow key={u.id} hover>
                    <TableCell sx={{ fontFamily: 'monospace' }}>{u.username}</TableCell>
                    <TableCell sx={{ fontFamily: 'monospace' }}>{u.email}</TableCell>
                    <TableCell>
                      <Chip
                        label={u.role.toUpperCase()}
                        size="small"
                        sx={{
                          bgcolor: u.role === 'admin' ? '#000' : '#fff',
                          color: u.role === 'admin' ? '#fff' : '#000',
                          border: '1px solid #000',
                          borderRadius: 0,
                          fontFamily: 'monospace',
                          fontWeight: 600
                        }}
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={u.is_active ? 'ACTIVE' : 'INACTIVE'}
                        size="small"
                        sx={{
                          bgcolor: u.is_active ? '#e6ffe6' : '#ffe6e6',
                          color: '#000',
                          border: '1px solid #000',
                          borderRadius: 0,
                          fontFamily: 'monospace'
                        }}
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={u.has_keys ? 'YES' : 'NO'}
                        size="small"
                        sx={{
                          bgcolor: u.has_keys ? '#e6f2ff' : '#f0f0f0',
                          color: '#000',
                          border: '1px solid #000',
                          borderRadius: 0,
                          fontFamily: 'monospace'
                        }}
                      />
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'monospace' }}>
                      {u.last_login ? new Date(u.last_login).toLocaleDateString() : 'Never'}
                    </TableCell>
                    <TableCell>
                      <IconButton
                        size="small"
                        onClick={() => handleDeleteClick(u)}
                        disabled={u.role === 'admin' && u.id === user?.id}
                        sx={{ color: '#000' }}
                      >
                        <Delete />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
        PaperProps={{ sx: { border: '2px solid #000', borderRadius: 0 } }}
      >
        <DialogTitle sx={{ fontFamily: 'monospace', fontWeight: 700 }}>
          CONFIRM DELETE
        </DialogTitle>
        <DialogContent>
          <Typography sx={{ fontFamily: 'monospace' }}>
            Are you sure you want to delete user <strong>{userToDelete?.username}</strong>?
            This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => setDeleteDialogOpen(false)}
            sx={{
              color: '#000',
              border: '1px solid #000',
              borderRadius: 0,
              fontFamily: 'monospace'
            }}
          >
            CANCEL
          </Button>
          <Button
            onClick={deleteUser}
            variant="contained"
            sx={{
              bgcolor: '#000',
              color: '#fff',
              borderRadius: 0,
              fontFamily: 'monospace',
              '&:hover': { bgcolor: '#333' }
            }}
          >
            DELETE
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default AdminDashboard;
