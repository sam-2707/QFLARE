/**
 * Federated Learning Node Visualization Component
 * Shows real-time status of all participating edge nodes
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  Tooltip,
  LinearProgress,
  IconButton,
  Badge,
  Alert,
} from '@mui/material';
import {
  Computer,
  CheckCircle,
  Warning,
  Error as ErrorIcon,
  Refresh,
  SignalCellularAlt,
  Speed,
  Memory,
} from '@mui/icons-material';
import { authenticatedFetch } from '../utils/api';

interface EdgeNode {
  node_id: string;
  node_name: string;
  status: 'active' | 'training' | 'idle' | 'offline' | 'error';
  last_seen: string;
  total_samples: number;
  local_accuracy: number;
  local_loss: number;
  contribution_weight: number;
  rounds_participated: number;
  data_distribution: string;
  computing_power: number;
  network_latency: number;
  is_byzantine: boolean;
}

interface FederatedNodesProps {
  trainingId?: string;
  autoRefresh?: boolean;
}

const FederatedNodes: React.FC<FederatedNodesProps> = ({ 
  trainingId, 
  autoRefresh = true 
}) => {
  const [nodes, setNodes] = useState<EdgeNode[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchNodes();
    
    if (autoRefresh) {
      const interval = setInterval(fetchNodes, 5000);
      return () => clearInterval(interval);
    }
  }, [trainingId, autoRefresh]);

  const fetchNodes = async () => {
    try {
      const endpoint = trainingId 
        ? `/api/training/${trainingId}/nodes`
        : '/api/training/nodes';
      
      const response = await authenticatedFetch(endpoint, { timeout: 10000 });
      
      if (response.ok) {
        const data = await response.json();
        setNodes(data.nodes || []);
      }
    } catch (error) {
      console.error('Failed to fetch nodes:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
      case 'training':
        return '#4caf50';
      case 'idle':
        return '#ff9800';
      case 'offline':
        return '#9e9e9e';
      case 'error':
        return '#f44336';
      default:
        return '#000000';
    }
  };

  const getStatusIcon = (node: EdgeNode) => {
    if (node.is_byzantine) {
      return <ErrorIcon sx={{ color: '#f44336' }} />;
    }
    
    switch (node.status) {
      case 'active':
      case 'training':
        return <CheckCircle sx={{ color: '#4caf50' }} />;
      case 'idle':
        return <Warning sx={{ color: '#ff9800' }} />;
      case 'offline':
        return <Computer sx={{ color: '#9e9e9e' }} />;
      default:
        return <ErrorIcon sx={{ color: '#f44336' }} />;
    }
  };

  if (loading) {
    return (
      <Box>
        <Typography variant="h6" sx={{ mb: 2 }}>
          🖥️ Federated Learning Nodes
        </Typography>
        <Grid container spacing={2}>
          {[1, 2, 3].map((i) => (
            <Grid item xs={12} md={4} key={i}>
              <Card>
                <CardContent>
                  <LinearProgress />
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600, fontFamily: 'Montserrat, sans-serif' }}>
          🖥️ Federated Learning Nodes ({nodes.length})
        </Typography>
        <IconButton onClick={fetchNodes} size="small">
          <Refresh />
        </IconButton>
      </Box>

      <Grid container spacing={2}>
        {nodes.map((node) => (
          <Grid item xs={12} sm={6} md={4} key={node.node_id}>
            <Card
              sx={{
                borderRadius: '12px',
                border: `2px solid ${getStatusColor(node.status)}`,
                position: 'relative',
                '&:hover': {
                  boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                  transform: 'translateY(-2px)',
                  transition: 'all 0.3s ease',
                },
              }}
            >
              <CardContent>
                {/* Node Header */}
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {getStatusIcon(node)}
                    <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                      {node.node_name}
                    </Typography>
                  </Box>
                  <Chip
                    label={node.status.toUpperCase()}
                    size="small"
                    sx={{
                      bgcolor: getStatusColor(node.status),
                      color: '#fff',
                      fontWeight: 600,
                      fontFamily: 'Montserrat, sans-serif',
                    }}
                  />
                </Box>

                {/* Byzantine Warning */}
                {node.is_byzantine && (
                  <Alert severity="error" sx={{ mb: 2, py: 0 }}>
                    ⚠️ Byzantine behavior detected
                  </Alert>
                )}

                {/* Node Metrics */}
                <Grid container spacing={1} sx={{ mb: 2 }}>
                  <Grid item xs={6}>
                    <Tooltip title="Local Accuracy">
                      <Box>
                        <Typography variant="caption" color="textSecondary">
                          Accuracy
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          {(node.local_accuracy * 100).toFixed(2)}%
                        </Typography>
                      </Box>
                    </Tooltip>
                  </Grid>
                  <Grid item xs={6}>
                    <Tooltip title="Local Loss">
                      <Box>
                        <Typography variant="caption" color="textSecondary">
                          Loss
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          {node.local_loss.toFixed(4)}
                        </Typography>
                      </Box>
                    </Tooltip>
                  </Grid>
                  <Grid item xs={6}>
                    <Tooltip title="Training Samples">
                      <Box>
                        <Typography variant="caption" color="textSecondary">
                          Samples
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          {node.total_samples.toLocaleString()}
                        </Typography>
                      </Box>
                    </Tooltip>
                  </Grid>
                  <Grid item xs={6}>
                    <Tooltip title="Rounds Participated">
                      <Box>
                        <Typography variant="caption" color="textSecondary">
                          Rounds
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          {node.rounds_participated}
                        </Typography>
                      </Box>
                    </Tooltip>
                  </Grid>
                </Grid>

                {/* Contribution Weight */}
                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                    <Typography variant="caption" color="textSecondary">
                      Contribution Weight
                    </Typography>
                    <Typography variant="caption" sx={{ fontWeight: 600 }}>
                      {(node.contribution_weight * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={node.contribution_weight * 100}
                    sx={{
                      height: 6,
                      borderRadius: 3,
                      bgcolor: '#e0e0e0',
                      '& .MuiLinearProgress-bar': {
                        bgcolor: getStatusColor(node.status),
                      },
                    }}
                  />
                </Box>

                {/* Performance Indicators */}
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  <Tooltip title={`Network Latency: ${node.network_latency}ms`}>
                    <Chip
                      icon={<SignalCellularAlt />}
                      label={`${node.network_latency}ms`}
                      size="small"
                      variant="outlined"
                    />
                  </Tooltip>
                  <Tooltip title={`Computing Power: ${node.computing_power}%`}>
                    <Chip
                      icon={<Speed />}
                      label={`${node.computing_power}%`}
                      size="small"
                      variant="outlined"
                    />
                  </Tooltip>
                  <Tooltip title={`Data Distribution: ${node.data_distribution}`}>
                    <Chip
                      icon={<Memory />}
                      label={node.data_distribution}
                      size="small"
                      variant="outlined"
                    />
                  </Tooltip>
                </Box>

                {/* Last Seen */}
                <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mt: 1 }}>
                  Last seen: {new Date(node.last_seen).toLocaleTimeString()}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {nodes.length === 0 && (
        <Card sx={{ textAlign: 'center', py: 4 }}>
          <Computer sx={{ fontSize: 64, color: '#bdbdbd', mb: 2 }} />
          <Typography variant="body1" color="textSecondary">
            No edge nodes available. Start a training session to see participating nodes.
          </Typography>
        </Card>
      )}
    </Box>
  );
};

export default FederatedNodes;
