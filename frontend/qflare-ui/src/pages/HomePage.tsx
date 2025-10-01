import React from 'react';
import {
  Box,
  Typography,
  Container,
  Grid,
  Card,
  CardContent,
  Button,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
  LinearProgress,
} from '@mui/material';
import {
  Security,
  Speed,
  Shield,
  DeviceHub,
  CheckCircle,
  TrendingUp,
  Lock,
  Computer,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const HomePage: React.FC = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: <Security sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: 'Post-Quantum Cryptography',
      description: 'CRYSTALS-Kyber-1024 and Dilithium-2 provide quantum-resistant security.',
    },
    {
      icon: <Shield sx={{ fontSize: 40, color: 'secondary.main' }} />,
      title: 'Secure Key Exchange',
      description: 'Multiple MITM-resistant methods for secure key distribution.',
    },
    {
      icon: <DeviceHub sx={{ fontSize: 40, color: 'success.main' }} />,
      title: 'Federated Learning',
      description: 'Distributed machine learning without compromising data privacy.',
    },
    {
      icon: <Speed sx={{ fontSize: 40, color: 'warning.main' }} />,
      title: 'High Performance',
      description: 'Optimized for enterprise-scale deployment and real-time processing.',
    },
  ];

  const securityMetrics = [
    { label: 'Quantum Resistance Level', value: 95, color: 'primary' },
    { label: 'Key Exchange Security', value: 98, color: 'secondary' },
    { label: 'Encryption Strength', value: 100, color: 'success' },
    { label: 'Network Security', value: 92, color: 'warning' },
  ];

  const benefits = [
    'Military-grade quantum-safe encryption',
    'Zero-knowledge federated learning',
    'MITM attack prevention protocols',
    'Real-time OTP delivery via SMS',
    'Enterprise-ready scalability',
    'Compliance with future quantum threats',
  ];

  return (
    <Box>
      {/* Hero Section */}
      <Paper
        sx={{
          background: 'linear-gradient(135deg, #1e40af 0%, #3b82f6 100%)',
          color: 'white',
          p: 6,
          mb: 6,
          borderRadius: 4,
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'url("data:image/svg+xml,%3Csvg width="40" height="40" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="rgba(255,255,255,0.1)" fill-opacity="0.1"%3E%3Cpath d="M20 20c0 4.4-3.6 8-8 8s-8-3.6-8-8 3.6-8 8-8 8 3.6 8 8zm0-20c4.4 0 8 3.6 8 8s-3.6 8-8 8-8-3.6-8-8 3.6-8 8-8z"/%3E%3C/g%3E%3C/svg%3E")',
          }}
        />
        <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={8}>
              <Typography variant="h1" sx={{ mb: 2, fontSize: { xs: '2.5rem', md: '3.5rem' } }}>
                Quantum-Safe Federated Learning
              </Typography>
              <Typography variant="h5" sx={{ mb: 4, opacity: 0.9, fontWeight: 400 }}>
                Secure, scalable, and future-proof distributed machine learning with
                post-quantum cryptography protection.
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                <Button
                  variant="contained"
                  size="large"
                  onClick={() => navigate('/secure-register')}
                  sx={{
                    backgroundColor: 'rgba(255,255,255,0.2)',
                    color: 'white',
                    '&:hover': {
                      backgroundColor: 'rgba(255,255,255,0.3)',
                    },
                  }}
                >
                  Get Started
                </Button>
                <Button
                  variant="outlined"
                  size="large"
                  onClick={() => navigate('/devices')}
                  sx={{
                    borderColor: 'rgba(255,255,255,0.5)',
                    color: 'white',
                    '&:hover': {
                      borderColor: 'white',
                      backgroundColor: 'rgba(255,255,255,0.1)',
                    },
                  }}
                >
                  View Devices
                </Button>
                <Button
                  variant="outlined"
                  size="large"
                  onClick={() => navigate('/fl')}
                  sx={{
                    borderColor: 'rgba(255,255,255,0.5)',
                    color: 'white',
                    '&:hover': {
                      borderColor: 'white',
                      backgroundColor: 'rgba(255,255,255,0.1)',
                    },
                  }}
                >
                  FL Dashboard
                </Button>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Computer sx={{ fontSize: 120, opacity: 0.8 }} />
              </Box>
            </Grid>
          </Grid>
        </Container>
      </Paper>

      {/* Features Section */}
      <Container maxWidth="lg" sx={{ mb: 6 }}>
        <Typography variant="h3" align="center" sx={{ mb: 1 }}>
          Enterprise-Grade Security
        </Typography>
        <Typography variant="body1" align="center" color="text.secondary" sx={{ mb: 4 }}>
          Built for the quantum computing era with military-grade encryption
        </Typography>
        <Grid container spacing={4}>
          {features.map((feature, index) => (
            <Grid item xs={12} md={6} lg={3} key={index}>
              <Card
                sx={{
                  height: '100%',
                  textAlign: 'center',
                  p: 3,
                  transition: 'transform 0.3s ease-in-out',
                  '&:hover': {
                    transform: 'translateY(-8px)',
                  },
                }}
              >
                <CardContent>
                  <Box sx={{ mb: 2 }}>{feature.icon}</Box>
                  <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                    {feature.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {feature.description}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Container>

      {/* Security Metrics */}
      <Container maxWidth="lg" sx={{ mb: 6 }}>
        <Grid container spacing={4}>
          <Grid item xs={12} md={6}>
            <Card sx={{ height: '100%', p: 3 }}>
              <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
                Security Metrics
              </Typography>
              {securityMetrics.map((metric, index) => (
                <Box key={index} sx={{ mb: 3 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" fontWeight={600}>
                      {metric.label}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {metric.value}%
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={metric.value}
                    color={metric.color as any}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                </Box>
              ))}
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card sx={{ height: '100%', p: 3 }}>
              <Typography variant="h5" sx={{ mb: 3, fontWeight: 600 }}>
                Key Benefits
              </Typography>
              <List>
                {benefits.map((benefit, index) => (
                  <ListItem key={index} sx={{ px: 0 }}>
                    <ListItemIcon sx={{ minWidth: 36 }}>
                      <CheckCircle color="success" />
                    </ListItemIcon>
                    <ListItemText
                      primary={benefit}
                      primaryTypographyProps={{ variant: 'body2' }}
                    />
                  </ListItem>
                ))}
              </List>
            </Card>
          </Grid>
        </Grid>
      </Container>

      {/* Status Cards */}
      <Container maxWidth="lg">
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Card sx={{ p: 3, textAlign: 'center' }}>
              <Lock sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
              <Typography variant="h4" color="primary.main" fontWeight={700}>
                256-bit
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Quantum-Safe Encryption
              </Typography>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card sx={{ p: 3, textAlign: 'center' }}>
              <TrendingUp sx={{ fontSize: 48, color: 'success.main', mb: 2 }} />
              <Typography variant="h4" color="success.main" fontWeight={700}>
                99.9%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                System Uptime
              </Typography>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card sx={{ p: 3, textAlign: 'center' }}>
              <DeviceHub sx={{ fontSize: 48, color: 'secondary.main', mb: 2 }} />
              <Typography variant="h4" color="secondary.main" fontWeight={700}>
                Enterprise
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Ready Scale
              </Typography>
            </Card>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default HomePage;
