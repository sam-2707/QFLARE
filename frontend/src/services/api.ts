// API service for making backend requests
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8001/api';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to include auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('qflare-token');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid, redirect to login
      localStorage.removeItem('qflare-token');
      localStorage.removeItem('qflare-user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API endpoints
export const api = {
  // Authentication
  auth: {
    login: (credentials: { username: string; password: string }) =>
      apiClient.post('/auth/login', credentials),
    logout: () => apiClient.post('/auth/logout'),
    validateToken: () => apiClient.get('/auth/validate-token'),
  },

  // System info
  system: {
    getInfo: () => apiClient.get('/system/info'),
    getHealth: () => apiClient.get('../health'),
  },

  // Metrics
  metrics: {
    getRealtime: () => apiClient.get('/metrics'),
    getHistory: () => apiClient.get('/metrics/history'),
  },

  // Clients
  clients: {
    getAll: () => apiClient.get('/clients'),
    getById: (id: string) => apiClient.get(`/clients/${id}`),
    disconnect: (id: string) => apiClient.post(`/clients/${id}/disconnect`),
  },

  // Training
  training: {
    getStatus: () => apiClient.get('/training/status'),
    start: () => apiClient.post('/training/start'),
    stop: () => apiClient.post('/training/stop'),
    getRounds: () => apiClient.get('/training/rounds'),
  },

  // Configuration
  config: {
    get: () => apiClient.get('/config'),
    update: (config: any) => apiClient.put('/config', config),
  },

  // Security
  security: {
    getScans: () => apiClient.get('/security/scans'),
    startScan: () => apiClient.post('/security/scan'),
  },
};

export default apiClient;