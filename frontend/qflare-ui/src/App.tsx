import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

import theme from './theme';
import Layout from './components/Layout/Layout';
import HomePage from './pages/HomePage';
import SecureRegistration from './pages/SecureRegistration';
import VerificationPage from './pages/VerificationPage';
import DevicesPage from './pages/DevicesPage';
import AdminDashboard from './pages/AdminDashboard';
import FederatedLearningPage from './pages/FederatedLearningPage';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/secure-register" element={<SecureRegistration />} />
            <Route path="/verify/:deviceId" element={<VerificationPage />} />
            <Route path="/devices" element={<DevicesPage />} />
            <Route path="/admin" element={<AdminDashboard />} />
            <Route path="/fl" element={<FederatedLearningPage />} />
          </Routes>
        </Layout>
      </Router>
      <ToastContainer
        position="top-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="light"
      />
    </ThemeProvider>
  );
}

export default App;