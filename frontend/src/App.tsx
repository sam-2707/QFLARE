import React, { useState } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Box, CssBaseline } from '@mui/material';
import { Helmet } from 'react-helmet-async';

// Authentication
import { AuthProvider, useAuth } from './contexts/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import Login from './components/Login';

// Layout Components
import Sidebar from './components/Layout/Sidebar';
import Header from './components/Layout/Header';

// Page Components
import Dashboard from './pages/Dashboard';
import Devices from './pages/Devices';
import FederatedLearning from './pages/FederatedLearning';
import Security from './pages/Security';
import Monitoring from './pages/Monitoring';
import Settings from './pages/Settings';
import DeviceRegistration from './pages/DeviceRegistration';

// Constants
const DRAWER_WIDTH = 280;

function AppContent() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const { isAuthenticated, user } = useAuth();

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  if (!isAuthenticated) {
    return <Login />;
  }

  return (
    <>
      <Helmet>
        <title>QFLARE Dashboard - Quantum Federated Learning Platform</title>
        <meta name="description" content="QFLARE - Advanced quantum-resistant federated learning platform with real-time monitoring and device management" />
        <meta name="keywords" content="quantum computing, federated learning, machine learning, cryptography, post-quantum" />
      </Helmet>
      
      <Box sx={{ display: 'flex' }}>
        <CssBaseline />
        
        <Header 
          drawerWidth={DRAWER_WIDTH}
          onDrawerToggle={handleDrawerToggle}
        />
        
        <Sidebar 
          drawerWidth={DRAWER_WIDTH}
          mobileOpen={mobileOpen}
          onDrawerToggle={handleDrawerToggle}
        />
        
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            p: 3,
            width: { sm: `calc(100% - ${DRAWER_WIDTH}px)` },
            mt: '64px',
            minHeight: 'calc(100vh - 64px)',
            backgroundColor: '#f5f5f5',
          }}
        >
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            
            {/* Common routes for both admin and user */}
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/federated-learning" element={<FederatedLearning />} />
            
            {/* Admin-only routes */}
            <Route 
              path="/devices" 
              element={
                <ProtectedRoute requiredRole="admin">
                  <Devices />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/devices/register" 
              element={
                <ProtectedRoute requiredRole="admin">
                  <DeviceRegistration />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/security" 
              element={
                <ProtectedRoute requiredRole="admin">
                  <Security />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/monitoring" 
              element={
                <ProtectedRoute requiredRole="admin">
                  <Monitoring />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/settings" 
              element={
                <ProtectedRoute requiredRole="admin">
                  <Settings />
                </ProtectedRoute>
              } 
            />
            
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </Box>
      </Box>
    </>
  );
}

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;