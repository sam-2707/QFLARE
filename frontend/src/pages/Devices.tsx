import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Button,
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
  Alert,
  Tooltip,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Avatar,
  Stack,
  Fab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Computer as ComputerIcon,
  CloudQueue as CloudIcon,
  Security as SecurityIcon,
  Speed as SpeedIcon,
  Storage as StorageIcon,
  NetworkCheck as NetworkIcon,
  ExpandMore as ExpandMoreIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

// Types
interface Node {
  id: string;
  nodeId: string;
  name: string;
  status: 'online' | 'offline' | 'training' | 'idle' | 'error' | 'maintenance';
  ipAddress: string;
  port: number;
  version: string;
  computePower: number;
  bandwidth: number;
  reliabilityScore: number;
  datasetSize: number;
  dataQuality: number;
  isTrusted: boolean;
  riskScore: number;
  lastSeen: string;
  registeredAt: string;
  capabilities: {
    gpu: boolean;
    encryption: boolean;
    differentialPrivacy: boolean;
    byzantineRobust: boolean;
  };
  metrics: {
    cpuUsage: number;
    memoryUsage: number;
    diskUsage: number;
    networkLatency: number;
  };
}

// Styled Components
const StatusChip = styled(Chip)<{ statuscolor: string }>(({ theme, statuscolor }) => ({
  backgroundColor: statuscolor,
  color: theme.palette.getContrastText(statuscolor),
  fontWeight: 'bold',
}));

const MetricCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  background: `linear-gradient(135deg, ${theme.palette.primary.main}15 0%, ${theme.palette.secondary.main}05 100%)`,
}));

const NodeAvatar = styled(Avatar)<{ statuscolor: string }>(({ statuscolor }) => ({
  backgroundColor: statuscolor,
  width: 48,
  height: 48,
}));

// Status configuration
const statusConfig = {
  online: { color: '#4caf50', label: 'Online' },
  offline: { color: '#f44336', label: 'Offline' },
  training: { color: '#2196f3', label: 'Training' },
  idle: { color: '#ff9800', label: 'Idle' },
  error: { color: '#e91e63', label: 'Error' },
  maintenance: { color: '#9c27b0', label: 'Maintenance' }
};

const Devices: React.FC = () => {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [actionDialogOpen, setActionDialogOpen] = useState(false);
  const [actionType, setActionType] = useState<'start' | 'pause' | 'stop' | 'maintain'>('start');
  const [formData, setFormData] = useState({
    name: '',
    ipAddress: '',
    port: 8080,
    capabilities: {
      gpu: false,
      encryption: true,
      differentialPrivacy: true,
      byzantineRobust: true
    }
  });
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  // Mock data - replace with actual API calls
  useEffect(() => {
    const mockNodes: Node[] = [
      {
        id: '1',
        nodeId: 'node_001',
        name: 'Edge Node Alpha',
        status: 'online',
        ipAddress: '192.168.1.100',
        port: 8080,
        version: '2.1.0',
        computePower: 85,
        bandwidth: 1000,
        reliabilityScore: 0.95,
        datasetSize: 50000,
        dataQuality: 0.92,
        isTrusted: true,
        riskScore: 0.15,
        lastSeen: '2024-01-15T10:30:00Z',
        registeredAt: '2024-01-01T08:00:00Z',
        capabilities: {
          gpu: true,
          encryption: true,
          differentialPrivacy: true,
          byzantineRobust: true
        },
        metrics: {
          cpuUsage: 45,
          memoryUsage: 62,
          diskUsage: 38,
          networkLatency: 12
        }
      },
      {
        id: '2',
        nodeId: 'node_002',
        name: 'Mobile Client Beta',
        status: 'training',
        ipAddress: '192.168.1.101',
        port: 8080,
        version: '2.0.5',
        computePower: 35,
        bandwidth: 100,
        reliabilityScore: 0.78,
        datasetSize: 15000,
        dataQuality: 0.88,
        isTrusted: true,
        riskScore: 0.32,
        lastSeen: '2024-01-15T10:28:00Z',
        registeredAt: '2024-01-05T14:20:00Z',
        capabilities: {
          gpu: false,
          encryption: true,
          differentialPrivacy: true,
          byzantineRobust: false
        },
        metrics: {
          cpuUsage: 78,
          memoryUsage: 85,
          diskUsage: 65,
          networkLatency: 45
        }
      },
      {
        id: '3',
        nodeId: 'node_003',
        name: 'Cloud Instance Gamma',
        status: 'idle',
        ipAddress: '10.0.1.50',
        port: 8080,
        version: '2.1.0',
        computePower: 120,
        bandwidth: 10000,
        reliabilityScore: 0.99,
        datasetSize: 200000,
        dataQuality: 0.96,
        isTrusted: true,
        riskScore: 0.05,
        lastSeen: '2024-01-15T10:32:00Z',
        registeredAt: '2023-12-15T09:15:00Z',
        capabilities: {
          gpu: true,
          encryption: true,
          differentialPrivacy: true,
          byzantineRobust: true
        },
        metrics: {
          cpuUsage: 15,
          memoryUsage: 28,
          diskUsage: 42,
          networkLatency: 8
        }
      },
      {
        id: '4',
        nodeId: 'node_004',
        name: 'Edge Device Delta',
        status: 'error',
        ipAddress: '192.168.1.102',
        port: 8080,
        version: '1.9.8',
        computePower: 25,
        bandwidth: 50,
        reliabilityScore: 0.45,
        datasetSize: 8000,
        dataQuality: 0.72,
        isTrusted: false,
        riskScore: 0.68,
        lastSeen: '2024-01-15T09:45:00Z',
        registeredAt: '2024-01-10T16:30:00Z',
        capabilities: {
          gpu: false,
          encryption: false,
          differentialPrivacy: false,
          byzantineRobust: false
        },
        metrics: {
          cpuUsage: 95,
          memoryUsage: 92,
          diskUsage: 88,
          networkLatency: 150
        }
      }
    ];
    
    setTimeout(() => {
      setNodes(mockNodes);
      setLoading(false);
    }, 1000);
  }, []);

  const handleAddNode = () => {
    setSelectedNode(null);
    setFormData({
      name: '',
      ipAddress: '',
      port: 8080,
      capabilities: {
        gpu: false,
        encryption: true,
        differentialPrivacy: true,
        byzantineRobust: true
      }
    });
    setDialogOpen(true);
  };

  const handleEditNode = (node: Node) => {
    setSelectedNode(node);
    setFormData({
      name: node.name,
      ipAddress: node.ipAddress,
      port: node.port,
      capabilities: node.capabilities
    });
    setDialogOpen(true);
  };

  const handleNodeAction = (node: Node, action: 'start' | 'pause' | 'stop' | 'maintain') => {
    setSelectedNode(node);
    setActionType(action);
    setActionDialogOpen(true);
  };

  const executeNodeAction = () => {
    if (!selectedNode) return;
    
    // Mock action execution
    const updatedNodes = nodes.map(node => {
      if (node.id === selectedNode.id) {
        let newStatus = node.status;
        switch (actionType) {
          case 'start':
            newStatus = 'training';
            break;
          case 'pause':
            newStatus = 'idle';
            break;
          case 'stop':
            newStatus = 'offline';
            break;
          case 'maintain':
            newStatus = 'maintenance';
            break;
        }
        return { ...node, status: newStatus as Node['status'] };
      }
      return node;
    });
    
    setNodes(updatedNodes);
    setActionDialogOpen(false);
  };

  const handleSaveNode = () => {
    // Mock save functionality
    if (selectedNode) {
      // Update existing node
      const updatedNodes = nodes.map(node =>
        node.id === selectedNode.id
          ? { ...node, ...formData, lastSeen: new Date().toISOString() }
          : node
      );
      setNodes(updatedNodes);
    } else {
      // Add new node
      const newNode: Node = {
        id: Date.now().toString(),
        nodeId: `node_${Date.now()}`,
        ...formData,
        status: 'offline',
        version: '2.1.0',
        computePower: 50,
        bandwidth: 100,
        reliabilityScore: 0.8,
        datasetSize: 10000,
        dataQuality: 0.85,
        isTrusted: false,
        riskScore: 0.3,
        lastSeen: new Date().toISOString(),
        registeredAt: new Date().toISOString(),
        metrics: {
          cpuUsage: 0,
          memoryUsage: 0,
          diskUsage: 0,
          networkLatency: 0
        }
      };
      setNodes([...nodes, newNode]);
    }
    setDialogOpen(false);
  };

  const handleDeleteNode = (nodeId: string) => {
    setNodes(nodes.filter(node => node.id !== nodeId));
  };

  const refreshNodes = () => {
    setLoading(true);
    setTimeout(() => {
      // Mock refresh - update last seen times
      const updatedNodes = nodes.map(node => ({
        ...node,
        lastSeen: new Date().toISOString(),
        metrics: {
          ...node.metrics,
          cpuUsage: Math.random() * 100,
          memoryUsage: Math.random() * 100,
          networkLatency: Math.random() * 200
        }
      }));
      setNodes(updatedNodes);
      setLoading(false);
    }, 500);
  };

  const filteredNodes = nodes.filter(node => {
    const matchesSearch = node.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         node.nodeId.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         node.ipAddress.includes(searchTerm);
    
    const matchesStatus = statusFilter === 'all' || node.status === statusFilter;
    
    return matchesSearch && matchesStatus;
  });

  const getStatusCounts = () => {
    return Object.keys(statusConfig).reduce((acc, status) => {
      acc[status] = nodes.filter(node => node.status === status).length;
      return acc;
    }, {} as Record<string, number>);
  };

  const statusCounts = getStatusCounts();

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 'bold' }}>
          Device Management
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={refreshNodes}
            disabled={loading}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={handleAddNode}
          >
            Add Node
          </Button>
        </Box>
      </Box>

      {/* Status Overview */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {Object.entries(statusConfig).map(([status, config]) => (
          <Grid item xs={12} sm={6} md={2} key={status}>
            <MetricCard>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Avatar sx={{ bgcolor: config.color, width: 32, height: 32 }}>
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                      {statusCounts[status] || 0}
                    </Typography>
                  </Avatar>
                  <Typography variant="body2" color="text.secondary">
                    {config.label}
                  </Typography>
                </Box>
              </CardContent>
            </MetricCard>
          </Grid>
        ))}
      </Grid>

      {/* Filters */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Search nodes..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </Grid>
          <Grid item xs={12} md={2}>
            <FormControl fullWidth>
              <InputLabel>Status</InputLabel>
              <Select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                label="Status"
              >
                <MenuItem value="all">All</MenuItem>
                {Object.entries(statusConfig).map(([status, config]) => (
                  <MenuItem key={status} value={status}>
                    {config.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      </Paper>

      {/* Nodes Table */}
      {loading ? (
        <Paper sx={{ p: 3 }}>
          <LinearProgress />
          <Typography sx={{ mt: 2, textAlign: 'center' }}>Loading nodes...</Typography>
        </Paper>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Node</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Address</TableCell>
                <TableCell>Performance</TableCell>
                <TableCell>Dataset</TableCell>
                <TableCell>Security</TableCell>
                <TableCell align="center">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredNodes.map((node) => (
                <React.Fragment key={node.id}>
                  <TableRow hover>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <NodeAvatar statuscolor={statusConfig[node.status].color}>
                          <ComputerIcon />
                        </NodeAvatar>
                        <Box>
                          <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                            {node.name}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {node.nodeId} • v{node.version}
                          </Typography>
                        </Box>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <StatusChip
                        statuscolor={statusConfig[node.status].color}
                        label={statusConfig[node.status].label}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">{node.ipAddress}:{node.port}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Last seen: {new Date(node.lastSeen).toLocaleTimeString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ minWidth: 120 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                          <SpeedIcon fontSize="small" color="primary" />
                          <Typography variant="caption">CPU: {node.metrics.cpuUsage}%</Typography>
                        </Box>
                        <LinearProgress
                          variant="determinate"
                          value={node.metrics.cpuUsage}
                          sx={{ height: 4 }}
                        />
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {(node.datasetSize / 1000).toFixed(0)}K samples
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Quality: {(node.dataQuality * 100).toFixed(0)}%
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Chip
                          label={node.isTrusted ? 'Trusted' : 'Untrusted'}
                          size="small"
                          color={node.isTrusted ? 'success' : 'warning'}
                        />
                        <Typography variant="caption" color="text.secondary">
                          Risk: {(node.riskScore * 100).toFixed(0)}%
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell align="center">
                      <Stack direction="row" spacing={1}>
                        {node.status === 'offline' || node.status === 'idle' ? (
                          <Tooltip title="Start Training">
                            <IconButton
                              size="small"
                              onClick={() => handleNodeAction(node, 'start')}
                            >
                              <PlayIcon />
                            </IconButton>
                          </Tooltip>
                        ) : node.status === 'training' ? (
                          <Tooltip title="Pause Training">
                            <IconButton
                              size="small"
                              onClick={() => handleNodeAction(node, 'pause')}
                            >
                              <PauseIcon />
                            </IconButton>
                          </Tooltip>
                        ) : null}
                        
                        <Tooltip title="Edit Node">
                          <IconButton
                            size="small"
                            onClick={() => handleEditNode(node)}
                          >
                            <EditIcon />
                          </IconButton>
                        </Tooltip>
                        
                        <Tooltip title="Delete Node">
                          <IconButton
                            size="small"
                            onClick={() => handleDeleteNode(node.id)}
                            color="error"
                          >
                            <DeleteIcon />
                          </IconButton>
                        </Tooltip>
                      </Stack>
                    </TableCell>
                  </TableRow>
                </React.Fragment>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {filteredNodes.length === 0 && !loading && (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <ComputerIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="text.secondary">
            No nodes found
          </Typography>
          <Typography color="text.secondary" sx={{ mb: 2 }}>
            {searchTerm || statusFilter !== 'all'
              ? 'Try adjusting your filters or search terms'
              : 'Add your first federated learning node to get started'}
          </Typography>
          <Button variant="contained" startIcon={<AddIcon />} onClick={handleAddNode}>
            Add Node
          </Button>
        </Paper>
      )}

      {/* Add/Edit Node Dialog */}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          {selectedNode ? 'Edit Node' : 'Add New Node'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Node Name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="IP Address"
                value={formData.ipAddress}
                onChange={(e) => setFormData({ ...formData, ipAddress: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} md={2}>
              <TextField
                fullWidth
                label="Port"
                type="number"
                value={formData.port}
                onChange={(e) => setFormData({ ...formData, port: parseInt(e.target.value) })}
              />
            </Grid>
            
            <Grid item xs={12}>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Capabilities
              </Typography>
              <Grid container spacing={1}>
                {Object.entries(formData.capabilities).map(([capability, enabled]) => (
                  <Grid item key={capability}>
                    <Chip
                      label={capability.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                      color={enabled ? 'primary' : 'default'}
                      onClick={() => setFormData({
                        ...formData,
                        capabilities: {
                          ...formData.capabilities,
                          [capability]: !enabled
                        }
                      })}
                      clickable
                    />
                  </Grid>
                ))}
              </Grid>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={handleSaveNode}>
            {selectedNode ? 'Update' : 'Add'} Node
          </Button>
        </DialogActions>
      </Dialog>

      {/* Node Action Dialog */}
      <Dialog open={actionDialogOpen} onClose={() => setActionDialogOpen(false)}>
        <DialogTitle>
          Confirm Action
        </DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to {actionType} node "{selectedNode?.name}"?
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setActionDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={executeNodeAction} color="primary">
            Confirm
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Devices;