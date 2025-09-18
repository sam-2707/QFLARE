import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Stepper,
  Step,
  StepLabel,
  Alert,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormHelperText,
  Chip,
  CircularProgress,
  Divider,
} from '@mui/material';
import {
  QrCode,
  Download,
  Security,
  Check,
  DevicesOther,
} from '@mui/icons-material';
import QRCode from 'qrcode.react';
import { useForm, Controller } from 'react-hook-form';
import toast from 'react-hot-toast';

interface DeviceFormData {
  deviceName: string;
  deviceType: string;
  organization: string;
  location: string;
  description: string;
}

const steps = ['Device Information', 'Generate Credentials', 'Configuration Download'];

const deviceTypes = [
  'IoT Sensor',
  'Edge Computer',
  'Mobile Device',
  'Server',
  'Embedded System',
  'Medical Device',
  'Automotive ECU',
  'Industrial Controller',
];

const DeviceRegistration: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [registrationData, setRegistrationData] = useState<any>(null);
  
  const { control, handleSubmit, formState: { errors }, watch } = useForm<DeviceFormData>({
    defaultValues: {
      deviceName: '',
      deviceType: '',
      organization: '',
      location: '',
      description: '',
    },
  });

  const watchedValues = watch();

  const onSubmit = async (data: DeviceFormData) => {
    setLoading(true);
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const mockRegistration = {
        deviceId: `QFLARE-${Date.now().toString(36).toUpperCase()}`,
        apiKey: `qflare_${Math.random().toString(36).substring(2)}`,
        certificateUrl: '/api/certificates/download',
        configUrl: '/api/config/download',
        serverUrl: 'https://qflare.company.com:8443',
        quantumKey: 'CRYSTALS-Kyber-1024',
        ...data,
      };
      
      setRegistrationData(mockRegistration);
      setActiveStep(1);
      toast.success('Device registered successfully!');
    } catch (error) {
      toast.error('Registration failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleNext = () => {
    if (activeStep === 0) {
      handleSubmit(onSubmit)();
    } else if (activeStep < steps.length - 1) {
      setActiveStep(activeStep + 1);
    }
  };

  const handleBack = () => {
    setActiveStep(activeStep - 1);
  };

  const downloadConfig = () => {
    const config = {
      deviceId: registrationData.deviceId,
      serverUrl: registrationData.serverUrl,
      apiKey: registrationData.apiKey,
      quantumProtocol: 'CRYSTALS-Kyber-1024',
      settings: {
        heartbeatInterval: 30000,
        maxRetries: 3,
        timeout: 10000,
        encryption: {
          algorithm: 'AES-256-GCM',
          keyExchange: 'CRYSTALS-Kyber-1024',
        },
      },
    };

    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `qflare-config-${registrationData.deviceId}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    toast.success('Configuration downloaded successfully!');
  };

  const renderStepContent = () => {
    switch (activeStep) {
      case 0:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Controller
                name="deviceName"
                control={control}
                rules={{ required: 'Device name is required' }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    fullWidth
                    label="Device Name"
                    error={!!errors.deviceName}
                    helperText={errors.deviceName?.message}
                    placeholder="e.g., MedTech-Sensor-001"
                  />
                )}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Controller
                name="deviceType"
                control={control}
                rules={{ required: 'Device type is required' }}
                render={({ field }) => (
                  <FormControl fullWidth error={!!errors.deviceType}>
                    <InputLabel>Device Type</InputLabel>
                    <Select {...field} label="Device Type">
                      {deviceTypes.map((type) => (
                        <MenuItem key={type} value={type}>
                          {type}
                        </MenuItem>
                      ))}
                    </Select>
                    <FormHelperText>{errors.deviceType?.message}</FormHelperText>
                  </FormControl>
                )}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <Controller
                name="organization"
                control={control}
                rules={{ required: 'Organization is required' }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    fullWidth
                    label="Organization"
                    error={!!errors.organization}
                    helperText={errors.organization?.message}
                    placeholder="e.g., Healthcare Corp"
                  />
                )}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <Controller
                name="location"
                control={control}
                rules={{ required: 'Location is required' }}
                render={({ field }) => (
                  <TextField
                    {...field}
                    fullWidth
                    label="Location"
                    error={!!errors.location}
                    helperText={errors.location?.message}
                    placeholder="e.g., New York, NY"
                  />
                )}
              />
            </Grid>

            <Grid item xs={12}>
              <Controller
                name="description"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    fullWidth
                    multiline
                    rows={3}
                    label="Description (Optional)"
                    placeholder="Brief description of the device and its purpose"
                  />
                )}
              />
            </Grid>
          </Grid>
        );

      case 1:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="h6" gutterBottom>
                  QR Code Configuration
                </Typography>
                <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
                  <QRCode
                    value={JSON.stringify({
                      deviceId: registrationData.deviceId,
                      serverUrl: registrationData.serverUrl,
                      apiKey: registrationData.apiKey,
                    })}
                    size={200}
                  />
                </Box>
                <Typography variant="caption" color="text.secondary">
                  Scan with your device to auto-configure
                </Typography>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Device Credentials
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Box>
                    <Typography variant="subtitle2" color="text.secondary">
                      Device ID
                    </Typography>
                    <Chip 
                      label={registrationData.deviceId} 
                      variant="outlined" 
                      size="small"
                      sx={{ fontFamily: 'monospace' }}
                    />
                  </Box>
                  
                  <Box>
                    <Typography variant="subtitle2" color="text.secondary">
                      API Key
                    </Typography>
                    <Chip 
                      label={`${registrationData.apiKey.substring(0, 20)}...`} 
                      variant="outlined" 
                      size="small"
                      sx={{ fontFamily: 'monospace' }}
                    />
                  </Box>

                  <Box>
                    <Typography variant="subtitle2" color="text.secondary">
                      Quantum Protocol
                    </Typography>
                    <Chip 
                      label="CRYSTALS-Kyber-1024" 
                      color="success"
                      size="small"
                      icon={<Security />}
                    />
                  </Box>
                </Box>
              </Card>
            </Grid>

            <Grid item xs={12}>
              <Alert severity="success" icon={<Check />}>
                <strong>Registration Complete!</strong> Your device has been successfully registered 
                with quantum-safe encryption. Proceed to download the configuration file.
              </Alert>
            </Grid>
          </Grid>
        );

      case 2:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card sx={{ p: 3, textAlign: 'center' }}>
                <DevicesOther sx={{ fontSize: 64, color: 'primary.main', mb: 2 }} />
                <Typography variant="h5" gutterBottom>
                  Device Ready for Deployment
                </Typography>
                <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                  Download the configuration file and install it on your device to complete the setup.
                </Typography>
                
                <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mb: 3 }}>
                  <Button
                    variant="contained"
                    startIcon={<Download />}
                    onClick={downloadConfig}
                    size="large"
                  >
                    Download Configuration
                  </Button>
                </Box>

                <Divider sx={{ my: 3 }} />

                <Typography variant="h6" gutterBottom>
                  Next Steps:
                </Typography>
                <Box sx={{ textAlign: 'left', maxWidth: 500, mx: 'auto' }}>
                  <ol>
                    <li>Download the configuration file above</li>
                    <li>Transfer the file to your device</li>
                    <li>Install the QFLARE client software</li>
                    <li>Load the configuration file</li>
                    <li>Start the device - it will automatically connect</li>
                  </ol>
                </Box>
              </Card>
            </Grid>
          </Grid>
        );

      default:
        return null;
    }
  };

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
        Device Registration
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
            {steps.map((label) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>

          {renderStepContent()}

          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
            <Button
              onClick={handleBack}
              disabled={activeStep === 0}
              variant="outlined"
            >
              Back
            </Button>
            
            <Box sx={{ display: 'flex', gap: 2 }}>
              {activeStep === steps.length - 1 ? (
                <Button
                  variant="contained"
                  onClick={() => {
                    setActiveStep(0);
                    setRegistrationData(null);
                    toast.success('Ready for next device registration');
                  }}
                >
                  Register Another Device
                </Button>
              ) : (
                <Button
                  onClick={handleNext}
                  variant="contained"
                  disabled={loading || (activeStep === 0 && !watchedValues.deviceName)}
                >
                  {loading ? (
                    <CircularProgress size={20} sx={{ mr: 1 }} />
                  ) : null}
                  {activeStep === 0 ? 'Register Device' : 'Next'}
                </Button>
              )}
            </Box>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default DeviceRegistration;