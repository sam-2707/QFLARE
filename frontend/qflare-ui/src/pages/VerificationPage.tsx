import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  TextField,
  Typography,
  Paper,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  Container,
  Chip,
} from '@mui/material';
import { Security, CheckCircle, PhoneAndroid } from '@mui/icons-material';
import { toast } from 'react-toastify';
import apiService from '../services/apiService';

const VerificationPage: React.FC = () => {
  const { deviceId } = useParams<{ deviceId: string }>();
  const navigate = useNavigate();
  const [otp, setOtp] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [verified, setVerified] = useState(false);

  const handleVerify = async () => {
    if (!deviceId || !otp) return;
    
    setLoading(true);
    setError('');
    try {
      await apiService.verifyOtp({ device_id: deviceId, otp });
      setVerified(true);
      toast.success('🎉 Device verified successfully!');
      setTimeout(() => navigate('/devices'), 2000);
    } catch (err: any) {
      const errorMsg = err.message || 'OTP verification failed';
      setError(errorMsg);
      toast.error(errorMsg);
    }
    setLoading(false);
  };

  if (verified) {
    return (
      <Container maxWidth="sm">
        <Card sx={{ p: 6, textAlign: 'center' }}>
          <CheckCircle sx={{ fontSize: 80, color: 'success.main', mb: 3 }} />
          <Typography variant="h4" fontWeight={700} mb={2} color="success.main">
            Verification Complete!
          </Typography>
          <Typography variant="body1" color="text.secondary" mb={4}>
            Device {deviceId} has been successfully verified and enrolled.
          </Typography>
          <Button variant="contained" size="large" onClick={() => navigate('/devices')}>
            Go to Dashboard
          </Button>
        </Card>
      </Container>
    );
  }

  return (
    <Container maxWidth="sm">
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
        <Typography variant="h4" fontWeight={700} mb={1}>
          Device Verification
        </Typography>
        <Typography variant="h6" sx={{ opacity: 0.9 }}>
          Complete your secure enrollment
        </Typography>
      </Paper>

      <Card sx={{ p: 4 }}>
        <Box sx={{ textAlign: 'center', mb: 4 }}>
          <PhoneAndroid sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
          <Typography variant="h5" fontWeight={600} mb={2}>
            Enter Verification Code
          </Typography>
          <Typography variant="body1" color="text.secondary" mb={2}>
            Enter the OTP sent to your registered phone number
          </Typography>
          <Chip label={`Device: ${deviceId}`} color="primary" variant="outlined" />
        </Box>

        <Box component="form" autoComplete="off" onSubmit={(e) => { e.preventDefault(); handleVerify(); }}>
          <TextField
            label="Enter OTP"
            value={otp}
            onChange={(e) => setOtp(e.target.value)}
            fullWidth
            required
            inputProps={{ 
              maxLength: 6, 
              style: { textAlign: 'center', fontSize: '1.5rem', letterSpacing: '0.5rem' } 
            }}
            helperText="Enter the 6-digit code"
            sx={{ mb: 3 }}
          />
          
          {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}
          
          <Button
            type="submit"
            variant="contained"
            size="large"
            fullWidth
            disabled={loading || otp.length !== 6}
            sx={{ py: 1.5 }}
          >
            {loading ? <CircularProgress size={24} color="inherit" /> : 'Verify Device'}
          </Button>
        </Box>
      </Card>
    </Container>
  );
};

export default VerificationPage;
