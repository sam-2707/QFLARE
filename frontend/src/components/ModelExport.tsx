/**
 * Model Export & Reports Component
 * Download trained models, training reports, and metrics
 */

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  CircularProgress,
} from '@mui/material';
import {
  Download,
  Description,
  Assessment,
  InsertChart,
  TableChart,
  CheckCircle,
  ModelTraining,
} from '@mui/icons-material';
import { authenticatedFetch } from '../utils/api';

interface ModelExportProps {
  trainingId: string;
}

const ModelExport: React.FC<ModelExportProps> = ({ trainingId }) => {
  const [format, setFormat] = useState<'h5' | 'onnx' | 'tflite' | 'pytorch'>('h5');
  const [reportFormat, setReportFormat] = useState<'pdf' | 'csv' | 'json'>('pdf');
  const [downloading, setDownloading] = useState<string | null>(null);

  const handleDownloadModel = async () => {
    setDownloading('model');
    try {
      const response = await authenticatedFetch(
        `/api/training/${trainingId}/export/model?format=${format}`,
        { timeout: 30000 }
      );
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `qflare_model_${trainingId}.${format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error('Failed to download model:', error);
    } finally {
      setDownloading(null);
    }
  };

  const handleDownloadReport = async () => {
    setDownloading('report');
    try {
      const response = await authenticatedFetch(
        `/api/training/${trainingId}/export/report?format=${reportFormat}`,
        { timeout: 30000 }
      );
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `qflare_report_${trainingId}.${reportFormat}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error('Failed to download report:', error);
    } finally {
      setDownloading(null);
    }
  };

  const handleDownloadMetrics = async () => {
    setDownloading('metrics');
    try {
      const response = await authenticatedFetch(
        `/api/training/${trainingId}/export/metrics`,
        { timeout: 20000 }
      );
      
      if (response.ok) {
        const data = await response.json();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `qflare_metrics_${trainingId}.json`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error('Failed to download metrics:', error);
    } finally {
      setDownloading(null);
    }
  };

  const handleDownloadLogs = async () => {
    setDownloading('logs');
    try {
      const response = await authenticatedFetch(
        `/api/training/${trainingId}/export/logs`,
        { timeout: 20000 }
      );
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `qflare_logs_${trainingId}.txt`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error('Failed to download logs:', error);
    } finally {
      setDownloading(null);
    }
  };

  return (
    <Card sx={{ borderRadius: '12px' }}>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, fontFamily: 'Montserrat, sans-serif', mb: 2 }}>
          📥 Export & Reports
        </Typography>

        <Grid container spacing={3}>
          {/* Model Export */}
          <Grid item xs={12} md={6}>
            <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: '8px' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <ModelTraining />
                <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                  Trained Model
                </Typography>
              </Box>
              <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                <InputLabel>Model Format</InputLabel>
                <Select
                  value={format}
                  label="Model Format"
                  onChange={(e) => setFormat(e.target.value as any)}
                >
                  <MenuItem value="h5">Keras H5 (.h5)</MenuItem>
                  <MenuItem value="onnx">ONNX (.onnx)</MenuItem>
                  <MenuItem value="tflite">TensorFlow Lite (.tflite)</MenuItem>
                  <MenuItem value="pytorch">PyTorch (.pth)</MenuItem>
                </Select>
              </FormControl>
              <Button
                fullWidth
                variant="contained"
                startIcon={downloading === 'model' ? <CircularProgress size={16} /> : <Download />}
                onClick={handleDownloadModel}
                disabled={downloading !== null}
                sx={{ bgcolor: '#000', '&:hover': { bgcolor: '#333' } }}
              >
                {downloading === 'model' ? 'Downloading...' : 'Download Model'}
              </Button>
              <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mt: 1 }}>
                Download the trained federated model weights
              </Typography>
            </Box>
          </Grid>

          {/* Training Report */}
          <Grid item xs={12} md={6}>
            <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: '8px' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <Assessment />
                <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                  Training Report
                </Typography>
              </Box>
              <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                <InputLabel>Report Format</InputLabel>
                <Select
                  value={reportFormat}
                  label="Report Format"
                  onChange={(e) => setReportFormat(e.target.value as any)}
                >
                  <MenuItem value="pdf">PDF Document (.pdf)</MenuItem>
                  <MenuItem value="csv">CSV Spreadsheet (.csv)</MenuItem>
                  <MenuItem value="json">JSON Data (.json)</MenuItem>
                </Select>
              </FormControl>
              <Button
                fullWidth
                variant="contained"
                startIcon={downloading === 'report' ? <CircularProgress size={16} /> : <Description />}
                onClick={handleDownloadReport}
                disabled={downloading !== null}
                sx={{ bgcolor: '#000', '&:hover': { bgcolor: '#333' } }}
              >
                {downloading === 'report' ? 'Generating...' : 'Download Report'}
              </Button>
              <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mt: 1 }}>
                Comprehensive training session report with metrics
              </Typography>
            </Box>
          </Grid>

          {/* Quick Export Options */}
          <Grid item xs={12}>
            <Divider sx={{ my: 1 }} />
            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2 }}>
              Quick Export
            </Typography>
            <List dense>
              <ListItem
                button
                onClick={handleDownloadMetrics}
                disabled={downloading !== null}
              >
                <ListItemIcon>
                  {downloading === 'metrics' ? <CircularProgress size={20} /> : <InsertChart />}
                </ListItemIcon>
                <ListItemText
                  primary="Raw Metrics Data"
                  secondary="JSON file with all training metrics and convergence data"
                />
                {downloading !== 'metrics' && <Download />}
              </ListItem>

              <ListItem
                button
                onClick={handleDownloadLogs}
                disabled={downloading !== null}
              >
                <ListItemIcon>
                  {downloading === 'logs' ? <CircularProgress size={20} /> : <Description />}
                </ListItemIcon>
                <ListItemText
                  primary="Training Logs"
                  secondary="Complete log file with timestamps and events"
                />
                {downloading !== 'logs' && <Download />}
              </ListItem>
            </List>
          </Grid>

          {/* Report Contents Preview */}
          <Grid item xs={12}>
            <Divider sx={{ my: 1 }} />
            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2 }}>
              Report Includes:
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Chip icon={<CheckCircle />} label="Model Architecture" size="small" />
              <Chip icon={<CheckCircle />} label="Training Configuration" size="small" />
              <Chip icon={<CheckCircle />} label="Node Participation" size="small" />
              <Chip icon={<CheckCircle />} label="Convergence Charts" size="small" />
              <Chip icon={<CheckCircle />} label="Privacy Metrics" size="small" />
              <Chip icon={<CheckCircle />} label="Security Events" size="small" />
              <Chip icon={<CheckCircle />} label="Performance Benchmarks" size="small" />
              <Chip icon={<CheckCircle />} label="Data Distribution Analysis" size="small" />
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default ModelExport;
