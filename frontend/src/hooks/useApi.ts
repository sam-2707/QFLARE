import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';

// Custom hook for real-time metrics
export function useRealTimeMetrics(interval: number = 5000) {
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = useCallback(async () => {
    try {
      const response = await api.metrics.getRealtime();
      setMetrics(response.data);
      setError(null);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch metrics');
      console.error('Metrics fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMetrics();
    const intervalId = setInterval(fetchMetrics, interval);
    return () => clearInterval(intervalId);
  }, [fetchMetrics, interval]);

  return { metrics, loading, error, refetch: fetchMetrics };
}

// Custom hook for system info
export function useSystemInfo() {
  const [systemInfo, setSystemInfo] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSystemInfo = async () => {
      try {
        const response = await api.system.getInfo();
        setSystemInfo(response.data);
        setError(null);
      } catch (err: any) {
        setError(err.message || 'Failed to fetch system info');
        console.error('System info fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchSystemInfo();
  }, []);

  return { systemInfo, loading, error };
}

// Custom hook for training status
export function useTrainingStatus() {
  const [trainingStatus, setTrainingStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await api.training.getStatus();
      setTrainingStatus(response.data);
      setError(null);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch training status');
      console.error('Training status fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const intervalId = setInterval(fetchStatus, 3000);
    return () => clearInterval(intervalId);
  }, [fetchStatus]);

  return { trainingStatus, loading, error, refetch: fetchStatus };
}

// Custom hook for clients
export function useClients() {
  const [clients, setClients] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchClients = useCallback(async () => {
    try {
      const response = await api.clients.getAll();
      // Backend returns {clients: [...]} so we need to extract the clients array
      const clientsData = response.data.clients || response.data;
      setClients(Array.isArray(clientsData) ? clientsData : []);
      setError(null);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch clients');
      console.error('Clients fetch error:', err);
      // Set empty array on error
      setClients([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchClients();
    const intervalId = setInterval(fetchClients, 10000);
    return () => clearInterval(intervalId);
  }, [fetchClients]);

  return { clients, loading, error, refetch: fetchClients };
}

// WebSocket hook for real-time updates
export function useWebSocket(url: string) {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [connectionStatus, setConnectionStatus] = useState<'Connecting' | 'Open' | 'Closing' | 'Closed'>('Closed');

  useEffect(() => {
    const ws = new WebSocket(url);
    setSocket(ws);
    setConnectionStatus('Connecting');

    ws.onopen = () => {
      setConnectionStatus('Open');
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setLastMessage(data);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onclose = () => {
      setConnectionStatus('Closed');
      console.log('WebSocket disconnected');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      ws.close();
    };
  }, [url]);

  const sendMessage = useCallback((message: any) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    }
  }, [socket]);

  return { socket, lastMessage, connectionStatus, sendMessage };
}