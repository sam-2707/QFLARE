/**
 * Model Convergence Visualization
 * Shows accuracy/loss trends across all nodes and global model
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  ToggleButtonGroup,
  ToggleButton,
  IconButton,
  FormControlLabel,
  Switch,
} from '@mui/material';
import { Refresh, Timeline } from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { authenticatedFetch } from '../utils/api';

interface ConvergenceData {
  round: number;
  global_accuracy: number;
  global_loss: number;
  nodes: {
    [nodeId: string]: {
      accuracy: number;
      loss: number;
    };
  };
}

interface ModelConvergenceProps {
  trainingId: string;
  autoRefresh?: boolean;
}

const ModelConvergence: React.FC<ModelConvergenceProps> = ({ 
  trainingId, 
  autoRefresh = true 
}) => {
  const [data, setData] = useState<ConvergenceData[]>([]);
  const [metric, setMetric] = useState<'accuracy' | 'loss'>('accuracy');
  const [showNodes, setShowNodes] = useState(true);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchConvergence();
    
    if (autoRefresh) {
      const interval = setInterval(fetchConvergence, 5000);
      return () => clearInterval(interval);
    }
  }, [trainingId, autoRefresh]);

  const fetchConvergence = async () => {
    try {
      const response = await authenticatedFetch(
        `/api/training/${trainingId}/convergence`,
        { timeout: 10000 }
      );
      
      if (response.ok) {
        const result = await response.json();
        // Sort by round number to ensure proper ordering
        const sortedHistory = (result.history || []).sort((a: ConvergenceData, b: ConvergenceData) => a.round - b.round);
        setData(sortedHistory);
      }
    } catch (error) {
      console.error('Failed to fetch convergence data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleMetricChange = (
    event: React.MouseEvent<HTMLElement>,
    newMetric: 'accuracy' | 'loss' | null
  ) => {
    if (newMetric !== null) {
      setMetric(newMetric);
    }
  };

  // Transform data for chart
  const chartData = data.map((round) => {
    const point: any = {
      round: round.round,
      global: metric === 'accuracy' 
        ? round.global_accuracy * 100 
        : round.global_loss,
    };

    if (showNodes) {
      Object.entries(round.nodes).forEach(([nodeId, metrics]) => {
        point[nodeId] = metric === 'accuracy' 
          ? metrics.accuracy * 100 
          : metrics.loss;
      });
    }

    return point;
  });

  // Colors for nodes
  const nodeColors = [
    '#4caf50', '#2196f3', '#ff9800', '#9c27b0', '#f44336',
    '#00bcd4', '#ffeb3b', '#795548', '#607d8b', '#e91e63',
  ];

  // Get all node IDs
  const nodeIds = data.length > 0 
    ? Object.keys(data[0].nodes) 
    : [];

  if (loading) {
    return (
      <Card sx={{ borderRadius: '12px' }}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>
            📈 Model Convergence
          </Typography>
          <Box sx={{ height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Typography color="textSecondary">Loading convergence data...</Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ borderRadius: '12px' }}>
      <CardContent>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Timeline />
            <Typography variant="h6" sx={{ fontWeight: 600, fontFamily: 'Montserrat, sans-serif' }}>
              Model Convergence
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <FormControlLabel
              control={
                <Switch
                  checked={showNodes}
                  onChange={(e) => setShowNodes(e.target.checked)}
                  size="small"
                />
              }
              label="Show Nodes"
              sx={{ mr: 2 }}
            />
            <ToggleButtonGroup
              value={metric}
              exclusive
              onChange={handleMetricChange}
              size="small"
            >
              <ToggleButton value="accuracy">Accuracy</ToggleButton>
              <ToggleButton value="loss">Loss</ToggleButton>
            </ToggleButtonGroup>
            <IconButton onClick={fetchConvergence} size="small">
              <Refresh />
            </IconButton>
          </Box>
        </Box>

        {/* Chart */}
        {data.length > 0 ? (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis
                dataKey="round"
                label={{ value: 'Training Round', position: 'insideBottom', offset: -5 }}
                tick={{ fontSize: 12 }}
              />
              <YAxis
                label={{
                  value: metric === 'accuracy' ? 'Accuracy (%)' : 'Loss',
                  angle: -90,
                  position: 'insideLeft',
                }}
                tick={{ fontSize: 12 }}
                domain={metric === 'accuracy' ? [0, 100] : ['auto', 'auto']}
              />
              <RechartsTooltip
                contentStyle={{
                  backgroundColor: '#fff',
                  border: '1px solid #e0e0e0',
                  borderRadius: '8px',
                  fontFamily: 'Montserrat, sans-serif',
                }}
                formatter={(value: any) => {
                  if (metric === 'accuracy') {
                    return `${parseFloat(value).toFixed(2)}%`;
                  }
                  return parseFloat(value).toFixed(4);
                }}
              />
              <Legend />
              
              {/* Global model line */}
              <Line
                type="monotone"
                dataKey="global"
                stroke="#000000"
                strokeWidth={3}
                name="Global Model"
                dot={{ r: 4 }}
                activeDot={{ r: 6 }}
              />

              {/* Node lines */}
              {showNodes && nodeIds.map((nodeId, index) => (
                <Line
                  key={nodeId}
                  type="monotone"
                  dataKey={nodeId}
                  stroke={nodeColors[index % nodeColors.length]}
                  strokeWidth={2}
                  name={`Node ${nodeId.substring(0, 8)}`}
                  dot={{ r: 2 }}
                  strokeDasharray="5 5"
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <Box sx={{ height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Box sx={{ textAlign: 'center' }}>
              <Timeline sx={{ fontSize: 64, color: '#bdbdbd', mb: 2 }} />
              <Typography variant="body1" color="textSecondary">
                No convergence data available yet
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Start training to see model convergence
              </Typography>
            </Box>
          </Box>
        )}

        {/* Statistics */}
        {data.length > 0 && (
          <Box sx={{ mt: 2, display: 'flex', gap: 3, flexWrap: 'wrap' }}>
            <Box>
              <Typography variant="caption" color="textSecondary">
                Total Rounds
              </Typography>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                {data.length}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="textSecondary">
                Current {metric === 'accuracy' ? 'Accuracy' : 'Loss'}
              </Typography>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                {metric === 'accuracy'
                  ? `${(data[data.length - 1].global_accuracy * 100).toFixed(2)}%`
                  : data[data.length - 1].global_loss.toFixed(4)}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="textSecondary">
                Best {metric === 'accuracy' ? 'Accuracy' : 'Loss'}
              </Typography>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                {metric === 'accuracy'
                  ? `${(Math.max(...data.map((d) => d.global_accuracy)) * 100).toFixed(2)}%`
                  : Math.min(...data.map((d) => d.global_loss)).toFixed(4)}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="textSecondary">
                Participating Nodes
              </Typography>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                {nodeIds.length}
              </Typography>
            </Box>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default ModelConvergence;
