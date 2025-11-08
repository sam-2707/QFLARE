/**
 * FL Configuration Panel
 * Advanced federated learning configuration for training sessions
 */

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Grid,
  Slider,
  Chip,
  Tooltip,
  IconButton,
  Collapse,
  Alert,
  FormControlLabel,
  Switch,
} from '@mui/material';
import {
  Settings,
  ExpandMore,
  Info,
  Save,
  RestartAlt,
} from '@mui/icons-material';

interface FLConfig {
  // Model Configuration
  model_type: 'CNN' | 'ResNet' | 'MobileNet' | 'Custom';
  num_classes: number;
  input_shape: [number, number, number];
  
  // Training Parameters
  num_rounds: number;
  local_epochs: number;
  batch_size: number;
  learning_rate: number;
  
  // Federated Parameters
  min_nodes: number;
  aggregation_method: 'FedAvg' | 'FedProx' | 'FedYogi' | 'FedAdam';
  client_selection: 'random' | 'round_robin' | 'performance_based';
  participation_rate: number;
  
  // Privacy Settings
  enable_dp: boolean;
  epsilon: number;
  delta: number;
  clip_norm: number;
  
  // Byzantine Defense
  enable_byzantine: boolean;
  detection_method: 'krum' | 'trimmed_mean' | 'median' | 'multi_krum';
  byzantine_threshold: number;
  
  // Data Distribution
  data_distribution: 'iid' | 'non_iid' | 'skewed';
  alpha: number; // Dirichlet distribution parameter
  
  // Advanced
  enable_compression: boolean;
  compression_ratio: number;
  enable_secure_aggregation: boolean;
}

interface FLConfigPanelProps {
  onSave?: (config: FLConfig) => void;
  defaultConfig?: Partial<FLConfig>;
}

const FLConfigPanel: React.FC<FLConfigPanelProps> = ({ 
  onSave, 
  defaultConfig = {} 
}) => {
  const [expanded, setExpanded] = useState(false);
  const [config, setConfig] = useState<FLConfig>({
    model_type: 'CNN',
    num_classes: 10,
    input_shape: [28, 28, 1],
    num_rounds: 10,
    local_epochs: 5,
    batch_size: 32,
    learning_rate: 0.01,
    min_nodes: 3,
    aggregation_method: 'FedAvg',
    client_selection: 'random',
    participation_rate: 1.0,
    enable_dp: true,
    epsilon: 1.0,
    delta: 1e-5,
    clip_norm: 1.0,
    enable_byzantine: true,
    detection_method: 'krum',
    byzantine_threshold: 0.3,
    data_distribution: 'non_iid',
    alpha: 0.5,
    enable_compression: false,
    compression_ratio: 0.1,
    enable_secure_aggregation: true,
    ...defaultConfig,
  });

  const [saved, setSaved] = useState(false);

  const handleChange = (field: keyof FLConfig, value: any) => {
    setConfig({ ...config, [field]: value });
    setSaved(false);
  };

  const handleSave = () => {
    if (onSave) {
      onSave(config);
    }
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  const handleReset = () => {
    setConfig({
      model_type: 'CNN',
      num_classes: 10,
      input_shape: [28, 28, 1],
      num_rounds: 10,
      local_epochs: 5,
      batch_size: 32,
      learning_rate: 0.01,
      min_nodes: 3,
      aggregation_method: 'FedAvg',
      client_selection: 'random',
      participation_rate: 1.0,
      enable_dp: true,
      epsilon: 1.0,
      delta: 1e-5,
      clip_norm: 1.0,
      enable_byzantine: true,
      detection_method: 'krum',
      byzantine_threshold: 0.3,
      data_distribution: 'non_iid',
      alpha: 0.5,
      enable_compression: false,
      compression_ratio: 0.1,
      enable_secure_aggregation: true,
    });
    setSaved(false);
  };

  return (
    <Card sx={{ borderRadius: '12px' }}>
      <CardContent>
        {/* Header */}
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            mb: 2,
            cursor: 'pointer',
          }}
          onClick={() => setExpanded(!expanded)}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Settings />
            <Typography variant="h6" sx={{ fontWeight: 600, fontFamily: 'Montserrat, sans-serif' }}>
              FL Configuration
            </Typography>
            <Chip
              label={`${config.aggregation_method} • ${config.data_distribution}`}
              size="small"
              sx={{ bgcolor: '#f5f5f5' }}
            />
          </Box>
          <IconButton
            sx={{
              transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.3s',
            }}
          >
            <ExpandMore />
          </IconButton>
        </Box>

        <Collapse in={expanded}>
          {saved && (
            <Alert severity="success" sx={{ mb: 2 }}>
              Configuration saved successfully!
            </Alert>
          )}

          <Grid container spacing={3}>
            {/* Model Configuration */}
            <Grid item xs={12}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                🧠 Model Configuration
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth size="small">
                <InputLabel>Model Type</InputLabel>
                <Select
                  value={config.model_type}
                  label="Model Type"
                  onChange={(e) => handleChange('model_type', e.target.value)}
                >
                  <MenuItem value="CNN">Convolutional Neural Network</MenuItem>
                  <MenuItem value="ResNet">ResNet</MenuItem>
                  <MenuItem value="MobileNet">MobileNet</MenuItem>
                  <MenuItem value="Custom">Custom Architecture</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                size="small"
                label="Number of Classes"
                type="number"
                value={config.num_classes}
                onChange={(e) => handleChange('num_classes', parseInt(e.target.value))}
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                size="small"
                label="Learning Rate"
                type="number"
                value={config.learning_rate}
                onChange={(e) => handleChange('learning_rate', parseFloat(e.target.value))}
                inputProps={{ step: 0.001 }}
              />
            </Grid>

            {/* Training Parameters */}
            <Grid item xs={12}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                🎯 Training Parameters
              </Typography>
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                size="small"
                label="Number of Rounds"
                type="number"
                value={config.num_rounds}
                onChange={(e) => handleChange('num_rounds', parseInt(e.target.value))}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                size="small"
                label="Local Epochs"
                type="number"
                value={config.local_epochs}
                onChange={(e) => handleChange('local_epochs', parseInt(e.target.value))}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                size="small"
                label="Batch Size"
                type="number"
                value={config.batch_size}
                onChange={(e) => handleChange('batch_size', parseInt(e.target.value))}
              />
            </Grid>

            {/* Federated Parameters */}
            <Grid item xs={12}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                🌐 Federated Learning
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth size="small">
                <InputLabel>Aggregation Method</InputLabel>
                <Select
                  value={config.aggregation_method}
                  label="Aggregation Method"
                  onChange={(e) => handleChange('aggregation_method', e.target.value)}
                >
                  <MenuItem value="FedAvg">FedAvg (Standard)</MenuItem>
                  <MenuItem value="FedProx">FedProx (Proximal)</MenuItem>
                  <MenuItem value="FedYogi">FedYogi (Adaptive)</MenuItem>
                  <MenuItem value="FedAdam">FedAdam (Momentum)</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth size="small">
                <InputLabel>Client Selection</InputLabel>
                <Select
                  value={config.client_selection}
                  label="Client Selection"
                  onChange={(e) => handleChange('client_selection', e.target.value)}
                >
                  <MenuItem value="random">Random Selection</MenuItem>
                  <MenuItem value="round_robin">Round Robin</MenuItem>
                  <MenuItem value="performance_based">Performance-Based</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ px: 1 }}>
                <Typography variant="caption" color="textSecondary" gutterBottom>
                  Minimum Nodes: {config.min_nodes}
                </Typography>
                <Slider
                  value={config.min_nodes}
                  onChange={(e, value) => handleChange('min_nodes', value)}
                  min={1}
                  max={20}
                  marks
                  valueLabelDisplay="auto"
                />
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ px: 1 }}>
                <Typography variant="caption" color="textSecondary" gutterBottom>
                  Participation Rate: {(config.participation_rate * 100).toFixed(0)}%
                </Typography>
                <Slider
                  value={config.participation_rate}
                  onChange={(e, value) => handleChange('participation_rate', value)}
                  min={0.1}
                  max={1.0}
                  step={0.1}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
                />
              </Box>
            </Grid>

            {/* Privacy Settings */}
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                  🔒 Differential Privacy
                </Typography>
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.enable_dp}
                      onChange={(e) => handleChange('enable_dp', e.target.checked)}
                      size="small"
                    />
                  }
                  label=""
                />
              </Box>
            </Grid>
            {config.enable_dp && (
              <>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Epsilon (ε)"
                    type="number"
                    value={config.epsilon}
                    onChange={(e) => handleChange('epsilon', parseFloat(e.target.value))}
                    inputProps={{ step: 0.1 }}
                  />
                </Grid>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Delta (δ)"
                    type="number"
                    value={config.delta}
                    onChange={(e) => handleChange('delta', parseFloat(e.target.value))}
                    inputProps={{ step: 0.00001 }}
                  />
                </Grid>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Clip Norm"
                    type="number"
                    value={config.clip_norm}
                    onChange={(e) => handleChange('clip_norm', parseFloat(e.target.value))}
                    inputProps={{ step: 0.1 }}
                  />
                </Grid>
              </>
            )}

            {/* Byzantine Defense */}
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                  🛡️ Byzantine Defense
                </Typography>
                <FormControlLabel
                  control={
                    <Switch
                      checked={config.enable_byzantine}
                      onChange={(e) => handleChange('enable_byzantine', e.target.checked)}
                      size="small"
                    />
                  }
                  label=""
                />
              </Box>
            </Grid>
            {config.enable_byzantine && (
              <>
                <Grid item xs={12} md={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Detection Method</InputLabel>
                    <Select
                      value={config.detection_method}
                      label="Detection Method"
                      onChange={(e) => handleChange('detection_method', e.target.value)}
                    >
                      <MenuItem value="krum">Krum</MenuItem>
                      <MenuItem value="multi_krum">Multi-Krum</MenuItem>
                      <MenuItem value="trimmed_mean">Trimmed Mean</MenuItem>
                      <MenuItem value="median">Median</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Box sx={{ px: 1 }}>
                    <Typography variant="caption" color="textSecondary" gutterBottom>
                      Byzantine Threshold: {(config.byzantine_threshold * 100).toFixed(0)}%
                    </Typography>
                    <Slider
                      value={config.byzantine_threshold}
                      onChange={(e, value) => handleChange('byzantine_threshold', value)}
                      min={0.1}
                      max={0.5}
                      step={0.05}
                      valueLabelDisplay="auto"
                      valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
                    />
                  </Box>
                </Grid>
              </>
            )}

            {/* Data Distribution */}
            <Grid item xs={12}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                📊 Data Distribution
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth size="small">
                <InputLabel>Distribution Type</InputLabel>
                <Select
                  value={config.data_distribution}
                  label="Distribution Type"
                  onChange={(e) => handleChange('data_distribution', e.target.value)}
                >
                  <MenuItem value="iid">IID (Balanced)</MenuItem>
                  <MenuItem value="non_iid">Non-IID (Heterogeneous)</MenuItem>
                  <MenuItem value="skewed">Skewed (Imbalanced)</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            {config.data_distribution === 'non_iid' && (
              <Grid item xs={12} md={6}>
                <Box sx={{ px: 1 }}>
                  <Typography variant="caption" color="textSecondary" gutterBottom>
                    Dirichlet Alpha: {config.alpha}
                  </Typography>
                  <Slider
                    value={config.alpha}
                    onChange={(e, value) => handleChange('alpha', value)}
                    min={0.1}
                    max={10.0}
                    step={0.1}
                    valueLabelDisplay="auto"
                  />
                </Box>
              </Grid>
            )}

            {/* Advanced Settings */}
            <Grid item xs={12}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                ⚙️ Advanced Settings
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.enable_compression}
                    onChange={(e) => handleChange('enable_compression', e.target.checked)}
                  />
                }
                label="Enable Model Compression"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.enable_secure_aggregation}
                    onChange={(e) => handleChange('enable_secure_aggregation', e.target.checked)}
                  />
                }
                label="Enable Secure Aggregation"
              />
            </Grid>

            {/* Action Buttons */}
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
                <Button
                  variant="outlined"
                  startIcon={<RestartAlt />}
                  onClick={handleReset}
                  sx={{
                    borderColor: '#000',
                    color: '#000',
                    '&:hover': { borderColor: '#000', bgcolor: '#f5f5f5' },
                  }}
                >
                  Reset to Default
                </Button>
                <Button
                  variant="contained"
                  startIcon={<Save />}
                  onClick={handleSave}
                  sx={{
                    bgcolor: '#000',
                    '&:hover': { bgcolor: '#333' },
                  }}
                >
                  Save Configuration
                </Button>
              </Box>
            </Grid>
          </Grid>
        </Collapse>
      </CardContent>
    </Card>
  );
};

export default FLConfigPanel;
