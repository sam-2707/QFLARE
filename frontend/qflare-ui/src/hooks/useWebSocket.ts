import { useState, useEffect, useCallback, useRef } from 'react';

export interface WebSocketMessage {
  event: string;
  data: any;
  timestamp: string;
}

export interface UseWebSocketOptions {
  url: string;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
  onMessage?: (message: WebSocketMessage) => void;
}

export interface UseWebSocketReturn {
  isConnected: boolean;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  lastMessage: WebSocketMessage | null;
  sendMessage: (message: any) => void;
  reconnect: () => void;
  disconnect: () => void;
}

export const useWebSocket = (options: UseWebSocketOptions): UseWebSocketReturn => {
  const {
    url,
    reconnectInterval = 3000,
    maxReconnectAttempts = 5,
    onOpen,
    onClose,
    onError,
    onMessage
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  
  const websocketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const shouldReconnectRef = useRef(true);

  const connect = useCallback(() => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setConnectionStatus('connecting');
    
    try {
      // Convert http/https URL to ws/wss
      const wsUrl = 'ws://localhost:8000/ws';
      websocketRef.current = new WebSocket(wsUrl);

      websocketRef.current.onopen = () => {
        setIsConnected(true);
        setConnectionStatus('connected');
        reconnectAttemptsRef.current = 0;
        
        console.log(`✅ WebSocket connected to ${wsUrl}`);
        onOpenRef.current?.();
      };

      websocketRef.current.onclose = (event) => {
        setIsConnected(false);
        setConnectionStatus('disconnected');
        
        console.log(`❌ WebSocket disconnected from ${wsUrl}. Code: ${event.code}, Reason: ${event.reason}`);
        onCloseRef.current?.();

        // Attempt to reconnect if enabled
        if (shouldReconnectRef.current && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1;
          console.log(`🔄 Attempting to reconnect (${reconnectAttemptsRef.current}/${maxReconnectAttempts})...`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };

      websocketRef.current.onerror = (error) => {
        setConnectionStatus('error');
        console.error('WebSocket error:', error);
        onErrorRef.current?.(error);
      };

      websocketRef.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          console.log(`📨 WebSocket message received:`, message.event, message.data);
          setLastMessage(message);
          onMessageRef.current?.(message);
        } catch (error) {
          console.error('❌ Error parsing WebSocket message:', error);
        }
      };

    } catch (error) {
      setConnectionStatus('error');
      console.error('Error creating WebSocket connection:', error);
    }
  }, [url, reconnectInterval, maxReconnectAttempts]);

  // Stable references for callbacks
  const onOpenRef = useRef(onOpen);
  const onCloseRef = useRef(onClose);
  const onErrorRef = useRef(onError);
  const onMessageRef = useRef(onMessage);

  useEffect(() => {
    onOpenRef.current = onOpen;
    onCloseRef.current = onClose;
    onErrorRef.current = onError;
    onMessageRef.current = onMessage;
  });

  const sendMessage = useCallback((message: any) => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      try {
        websocketRef.current.send(JSON.stringify(message));
      } catch (error) {
        console.error('Error sending WebSocket message:', error);
      }
    } else {
      console.warn('WebSocket is not connected. Message not sent:', message);
    }
  }, []);

  const reconnect = useCallback(() => {
    reconnectAttemptsRef.current = 0;
    shouldReconnectRef.current = true;
    connect();
  }, [connect]);

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
    }
    
    setIsConnected(false);
    setConnectionStatus('disconnected');
  }, []);

  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    isConnected,
    connectionStatus,
    lastMessage,
    sendMessage,
    reconnect,
    disconnect
  };
};

// Specialized hook for FL dashboard
export const useFLWebSocket = () => {
  const [flStatus, setFlStatus] = useState<any>(null);
  const [trainingProgress, setTrainingProgress] = useState<any>(null);
  const [modelAggregation, setModelAggregation] = useState<any>(null);
  const [deviceStatus, setDeviceStatus] = useState<any>(null);
  const [errorNotification, setErrorNotification] = useState<any>(null);

  const handleMessage = useCallback((message: WebSocketMessage) => {
    switch (message.event) {
      case 'fl_status_update':
        setFlStatus(message.data);
        break;
      case 'training_progress':
        setTrainingProgress(message.data);
        break;
      case 'model_aggregation':
        setModelAggregation(message.data);
        break;
      case 'device_status':
        setDeviceStatus(message.data);
        break;
      case 'error_notification':
        setErrorNotification(message.data);
        break;
      case 'recent_events':
        // Handle recent events for newly connected clients
        console.log('Received recent events:', message.data);
        break;
      case 'heartbeat':
        // Handle heartbeat to keep connection alive
        console.log('💓 Heartbeat received:', message.data.timestamp);
        break;
      case 'connection_established':
        // Handle initial connection confirmation
        console.log('🔗 Connection established:', message.data.message);
        break;
      default:
        console.log('Unhandled WebSocket event:', message.event, message.data);
    }
  }, []);

  const websocket = useWebSocket({
    url: 'ws://localhost:8003/api/ws/dashboard',
    onMessage: handleMessage,
    onOpen: () => {
      console.log('FL Dashboard WebSocket connected');
    },
    onClose: () => {
      console.log('FL Dashboard WebSocket disconnected');
    },
    onError: (error) => {
      console.error('FL Dashboard WebSocket error:', error);
    }
  });

  // Send subscription message on connect
  useEffect(() => {
    if (websocket.isConnected) {
      websocket.sendMessage({
        type: 'subscribe',
        events: ['fl_status_update', 'training_progress', 'model_aggregation', 'device_status', 'error_notification']
      });
    }
  }, [websocket.isConnected, websocket.sendMessage]);

  return {
    ...websocket,
    flStatus,
    trainingProgress,
    modelAggregation,
    deviceStatus,
    errorNotification,
    clearError: () => setErrorNotification(null)
  };
};