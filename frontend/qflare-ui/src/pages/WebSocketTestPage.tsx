import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Container, 
  Card, 
  CardContent, 
  Button, 
  Chip,
  Grid,
  Alert,
  LinearProgress
} from '@mui/material';
import { 
  Wifi, 
  WifiOff, 
  PlayArrow, 
  Refresh 
} from '@mui/icons-material';

// Simple WebSocket hook for testing
const useSimpleWebSocket = (url: string) => {
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<any[]>([]);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [ws, setWs] = useState<WebSocket | null>(null);

  useEffect(() => {
    const websocket = new WebSocket(url);
    
    websocket.onopen = () => {
      setIsConnected(true);
      console.log('WebSocket connected');
    };
    
    websocket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        setLastMessage(message);
        setMessages(prev => [...prev.slice(-9), message]); // Keep last 10 messages
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
    
    websocket.onclose = () => {
      setIsConnected(false);
      console.log('WebSocket disconnected');
    };
    
    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    setWs(websocket);
    
    return () => {
      websocket.close();
    };
  }, [url]);

  const sendMessage = (message: any) => {
    if (ws && isConnected) {
      ws.send(JSON.stringify(message));
    }
  };

  return { isConnected, messages, lastMessage, sendMessage };
};

const WebSocketTestPage: React.FC = () => {
  const [flStatus, setFlStatus] = useState<any>(null);
  const [trainingInProgress, setTrainingInProgress] = useState(false);
  
  // Connect to dashboard WebSocket
  const { isConnected, messages, lastMessage, sendMessage } = useSimpleWebSocket(
    'ws://localhost:8080/api/ws/dashboard'
  );

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      switch (lastMessage.event) {
        case 'fl_status_update':
          setFlStatus(lastMessage.data);
          break;
        case 'training_progress':
          console.log('Training progress:', lastMessage.data);
          break;
        case 'model_aggregation':
          console.log('Model aggregation:', lastMessage.data);
          setTrainingInProgress(false);
          break;
        default:
          console.log('WebSocket message:', lastMessage);
      }
    }
  }, [lastMessage]);

  const startRealTraining = async () => {
    setTrainingInProgress(true);
    
    try {
      const response = await fetch('http://localhost:8080/api/fl/run_real_training', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
          target_participants: '3',
          local_epochs: '3',
          learning_rate: '0.01',
          batch_size: '64'
        })
      });
      
      const result = await response.json();
      console.log('Training result:', result);
      
      if (!result.success) {
        setTrainingInProgress(false);
      }
    } catch (error) {
      console.error('Error starting training:', error);
      setTrainingInProgress(false);
    }
  };

  const fetchFLStatus = async () => {
    try {
      const response = await fetch('http://localhost:8080/api/fl/status');
      const data = await response.json();
      if (data.success) {
        setFlStatus(data.fl_status);
      }
    } catch (error) {
      console.error('Error fetching FL status:', error);
    }
  };

  // Subscribe to events when connected
  useEffect(() => {
    if (isConnected) {
      sendMessage({
        type: 'subscribe',
        events: ['fl_status_update', 'training_progress', 'model_aggregation']
      });
      
      // Fetch initial status
      fetchFLStatus();
    }
  }, [isConnected, sendMessage]);

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          WebSocket FL Dashboard
        </Typography>
        
        <Chip
          icon={isConnected ? <Wifi /> : <WifiOff />}
          label={isConnected ? "Connected" : "Disconnected"}
          color={isConnected ? "success" : "error"}
          variant="outlined"
        />
      </Box>

      <Grid container spacing={3}>
        {/* Connection Status */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Connection Status
              </Typography>
              <Typography variant="body2" color="textSecondary">
                WebSocket: {isConnected ? "Connected" : "Disconnected"}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Messages received: {messages.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* FL Status */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                FL Status
              </Typography>
              {flStatus ? (
                <>
                  <Typography variant="body2">
                    Round: {flStatus.current_round}/{flStatus.total_rounds}
                  </Typography>
                  <Typography variant="body2">
                    Status: <Chip label={flStatus.status} size="small" />
                  </Typography>
                  <Typography variant="body2">
                    Active devices: {flStatus.active_devices}
                  </Typography>
                </>
              ) : (
                <Typography variant="body2" color="textSecondary">
                  No FL status available
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Controls */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Controls
              </Typography>
              <Box display="flex" flexDirection="column" gap={2}>
                <Button
                  variant="contained"
                  startIcon={<PlayArrow />}
                  onClick={startRealTraining}
                  disabled={trainingInProgress || !isConnected}
                  fullWidth
                >
                  {trainingInProgress ? "Training..." : "Start Real FL Round"}
                </Button>
                
                <Button
                  variant="outlined"
                  startIcon={<Refresh />}
                  onClick={fetchFLStatus}
                  disabled={!isConnected}
                  fullWidth
                >
                  Refresh Status
                </Button>
              </Box>
              
              {trainingInProgress && (
                <Box mt={2}>
                  <LinearProgress />
                  <Typography variant="caption" color="textSecondary">
                    Real ML training in progress...
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Messages */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent WebSocket Messages
              </Typography>
              
              {messages.length === 0 ? (
                <Typography variant="body2" color="textSecondary">
                  No messages received yet
                </Typography>
              ) : (
                <Box maxHeight={300} overflow="auto">
                  {messages.slice().reverse().map((message, index) => (
                    <Alert key={index} severity="info" sx={{ mb: 1 }}>
                      <Typography variant="caption" component="div">
                        {message.timestamp}
                      </Typography>
                      <Typography variant="body2">
                        <strong>{message.event}</strong>: {JSON.stringify(message.data, null, 2)}
                      </Typography>
                    </Alert>
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default WebSocketTestPage;