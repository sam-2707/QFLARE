import React, { useState } from 'react';
import {
  Box,
  Button,
  TextField,
  Typography,
  Paper,
  Stepper,
  Step,
  StepLabel,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  Grid,
  Chip,
  Container,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { Security, PhoneAndroid, CheckCircle, Verified } from '@mui/icons-material';
import { toast } from 'react-toastify';
import apiService from '../services/apiService';

const steps = ['Device Information', 'Verify OTP', 'Registration Complete'];

const deviceTypes = [
  'IoT Device',
  'Mobile Device',
  'Desktop Computer',
  'Server',
  'Edge Device',
  'Embedded System',
];

const useCases = [
  'Healthcare Analytics',
  'Financial Services',
  'Manufacturing IoT',
  'Smart City Infrastructure',
  'Autonomous Vehicles',
  'Supply Chain Management',
  'Energy Management',
  'Other',
];

const SecureRegistration: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [form, setForm] = useState({
    device_id: '',
    device_type: '',
    organization: '',
    contact_email: '',
    phone_number: '',
    use_case: '',
  });
  const [otp, setOtp] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleRegister = async () => {
    setLoading(true);
    setError('');
    setSuccess('');
    try {
      await apiService.registerDevice({ ...form, key_exchange_method: 'qr_otp', phone_number: form.phone_number });
      setActiveStep(1);
      toast.success('Registration initiated! OTP sent to your phone.');
    } catch (err: any) {
      const errorMsg = err.message || 'Registration failed';
      setError(errorMsg);
      toast.error(errorMsg);
    }
    setLoading(false);
  };

  const handleVerify = async () => {
    setLoading(true);
    setError('');
    try {
      await apiService.verifyOtp({ device_id: form.device_id, otp });
      setActiveStep(2);
      setSuccess('Device registered and verified successfully!');
      toast.success('🎉 Device verified and enrolled in QFLARE network!');
    } catch (err: any) {
      const errorMsg = err.message || 'OTP verification failed';
      setError(errorMsg);
      toast.error(errorMsg);
    }
    setLoading(false);
  };

  return (
    <Container maxWidth="md">
      {/* Header */}
      <Paper
        sx={{
          background: 'linear-gradient(135deg, #1e40af 0%, #3b82f6 100%)',
          color: 'white',
          p: 4,
          mb: 4,
          borderRadius: 3,
          textAlign: 'center',
        }}
      >
        <Security sx={{ fontSize: 60, mb: 2 }} />
        <Typography variant="h3" fontWeight={700} mb={1}>
          Secure Device Registration
        </Typography>
        <Typography variant="h6" sx={{ opacity: 0.9 }}>
          Quantum-safe enrollment with multi-factor authentication
        </Typography>
        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1, mt: 2 }}>
          <Chip label="Post-Quantum Crypto" size="small" sx={{ backgroundColor: 'rgba(255,255,255,0.2)', color: 'white' }} />
          <Chip label="MITM Protection" size="small" sx={{ backgroundColor: 'rgba(255,255,255,0.2)', color: 'white' }} />
          <Chip label="SMS OTP" size="small" sx={{ backgroundColor: 'rgba(255,255,255,0.2)', color: 'white' }} />
        </Box>
      </Paper>

      <Card sx={{ p: 4 }}>
        <Stepper activeStep={activeStep} alternativeLabel sx={{ mb: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {activeStep === 0 && (
          <Box component="form" autoComplete="off" onSubmit={(e) => { e.preventDefault(); handleRegister(); }}>
            <Typography variant="h5" fontWeight={600} mb={3}>
              Device Information
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Device ID"
                  name="device_id"
                  value={form.device_id}
                  onChange={handleChange}
                  fullWidth
                  required
                  helperText="Unique identifier for your device"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth required>
                  <InputLabel>Device Type</InputLabel>
                  <Select
                    name="device_type"
                    value={form.device_type}
                    onChange={(e) => setForm({ ...form, device_type: e.target.value })}
                    label="Device Type"
                  >
                    {deviceTypes.map((type) => (
                      <MenuItem key={type} value={type}>
                        {type}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Organization"
                  name="organization"
                  value={form.organization}
                  onChange={handleChange}
                  fullWidth
                  required
                  helperText="Your company or organization name"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Contact Email"
                  name="contact_email"
                  value={form.contact_email}
                  onChange={handleChange}
                  fullWidth
                  required
                  type="email"
                  helperText="Primary contact email address"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Phone Number"
                  name="phone_number"
                  value={form.phone_number}
                  onChange={handleChange}
                  fullWidth
                  required
                  type="tel"
                  helperText="Mobile number for OTP delivery"
                  InputProps={{
                    startAdornment: <PhoneAndroid sx={{ mr: 1, color: 'action.active' }} />,
                  }}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth required>
                  <InputLabel>Use Case</InputLabel>
                  <Select
                    name="use_case"
                    value={form.use_case}
                    onChange={(e) => setForm({ ...form, use_case: e.target.value })}
                    label="Use Case"
                  >
                    {useCases.map((useCase) => (
                      <MenuItem key={useCase} value={useCase}>
                        {useCase}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
            {error && <Alert severity="error" sx={{ mt: 3 }}>{error}</Alert>}
            <Button
              type="submit"
              variant="contained"
              size="large"
              fullWidth
              sx={{ mt: 4, py: 1.5 }}
              disabled={loading}
            >
              {loading ? <CircularProgress size={24} color="inherit" /> : 'Register Device & Send OTP'}
            </Button>
          </Box>
        )}

        {activeStep === 1 && (
          <Box component="form" autoComplete="off" onSubmit={(e) => { e.preventDefault(); handleVerify(); }}>
            <Box sx={{ textAlign: 'center', mb: 4 }}>
              <PhoneAndroid sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
              <Typography variant="h5" fontWeight={600} mb={2}>
                Verify Your Phone Number
              </Typography>
              <Typography variant="body1" color="text.secondary" mb={2}>
                We've sent a 6-digit verification code to
              </Typography>
              <Chip label={form.phone_number} color="primary" variant="outlined" />
            </Box>
            <Box sx={{ maxWidth: 300, mx: 'auto' }}>
              <TextField
                label="Enter OTP"
                value={otp}
                onChange={(e) => setOtp(e.target.value)}
                fullWidth
                required
                inputProps={{ maxLength: 6, style: { textAlign: 'center', fontSize: '1.5rem', letterSpacing: '0.5rem' } }}
                helperText="Enter the 6-digit code sent to your phone"
              />
              {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
              <Button
                type="submit"
                variant="contained"
                size="large"
                fullWidth
                sx={{ mt: 3, py: 1.5 }}
                disabled={loading || otp.length !== 6}
              >
                {loading ? <CircularProgress size={24} color="inherit" /> : 'Verify OTP'}
              </Button>
            </Box>
          </Box>
        )}

        {activeStep === 2 && (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <CheckCircle sx={{ fontSize: 80, color: 'success.main', mb: 3 }} />
            <Typography variant="h4" fontWeight={700} mb={2} color="success.main">
              Registration Successful!
            </Typography>
            <Typography variant="h6" color="text.secondary" mb={4}>
              Your device is now enrolled in the QFLARE network with quantum-safe encryption.
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1, mb: 4 }}>
              <Chip label="Quantum Keys Active" color="success" icon={<Verified />} />
              <Chip label="CRYSTALS-Kyber-1024" color="primary" />
              <Chip label="CRYSTALS-Dilithium-2" color="secondary" />
            </Box>
            <Button
              variant="contained"
              size="large"
              onClick={() => window.location.href = '/devices'}
              sx={{ px: 4 }}
            >
              View Device Dashboard
            </Button>
          </Box>
        )}
      </Card>
    </Container>
  );
};

export default SecureRegistration;
