import React, { useState, useEffect } from 'react';
import {
  IconButton,
  Badge,
  Menu,
  MenuItem,
  Typography,
  Box,
  Divider,
  Avatar,
  Chip,
  Button,
  Snackbar,
  Alert,
  Tooltip,
} from '@mui/material';
import {
  Notifications,
  CheckCircle,
  Warning,
  Info,
  Close,
  CloudOff,
} from '@mui/icons-material';
import { authenticatedFetch, isAuthenticated, parseErrorResponse } from '../utils/api';

interface Notification {
  id: string;
  title: string;
  message: string;
  type: 'success' | 'warning' | 'info' | 'error';
  read: boolean;
  timestamp: string;
}

const NotificationBell: React.FC = () => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [error, setError] = useState<string>('');
  const [showError, setShowError] = useState(false);
  const [isOnline, setIsOnline] = useState(navigator.onLine);

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      fetchNotifications();
    };
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  useEffect(() => {
    fetchNotifications();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchNotifications, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchNotifications = async () => {
    // Don't fetch if not authenticated
    if (!isAuthenticated()) {
      console.log('Not authenticated, skipping notification fetch');
      setNotifications([]);
      setUnreadCount(0);
      return;
    }

    try {
      const response = await authenticatedFetch('/api/notifications', {
        timeout: 10000, // 10 second timeout for notifications
      });
      
      if (!response.ok) {
        const errorMsg = await parseErrorResponse(response);
        throw new Error(errorMsg);
      }
      
      const data = await response.json();
      setNotifications(data.notifications || []);
      setUnreadCount(data.unread || 0);
      setError('');
    } catch (error: any) {
      console.error('Failed to fetch notifications:', error);
      // Set empty arrays on error to prevent undefined issues
      setNotifications([]);
      setUnreadCount(0);
      
      // Only show error if user is authenticated (avoid spam on login page)
      if (isAuthenticated()) {
        setError(error.message || 'Failed to load notifications.');
        setShowError(true);
      }
    }
  };

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const markAsRead = async (notificationId: string) => {
    // Optimistic update - update UI immediately
    const previousNotifications = [...notifications];
    const previousUnreadCount = unreadCount;
    
    setNotifications(notifications.map(n => 
      n.id === notificationId ? { ...n, read: true } : n
    ));
    setUnreadCount(Math.max(0, unreadCount - 1));
    
    try {
      const response = await authenticatedFetch(`/api/notifications/${notificationId}/read`, {
        method: 'POST',
        timeout: 5000,
      });
      
      if (!response.ok) {
        throw new Error('Failed to mark as read');
      }
    } catch (error) {
      console.error('Failed to mark notification as read:', error);
      // Rollback on error
      setNotifications(previousNotifications);
      setUnreadCount(previousUnreadCount);
      setError('Failed to mark notification as read.');
      setShowError(true);
    }
  };

  const markAllAsRead = async () => {
    // Optimistic update
    const previousNotifications = [...notifications];
    const previousUnreadCount = unreadCount;
    
    setNotifications(notifications.map(n => ({ ...n, read: true })));
    setUnreadCount(0);
    
    try {
      const unreadNotifications = previousNotifications.filter((n) => !n.read);
      for (const notification of unreadNotifications) {
        const response = await authenticatedFetch(`/api/notifications/${notification.id}/read`, {
          method: 'POST',
          timeout: 5000,
        });
        if (!response.ok) throw new Error('Failed to mark as read');
      }
    } catch (error) {
      console.error('Failed to mark all as read:', error);
      // Rollback on error
      setNotifications(previousNotifications);
      setUnreadCount(previousUnreadCount);
      setError('Failed to mark all as read.');
      setShowError(true);
    }
  };

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <CheckCircle sx={{ color: '#4caf50', fontSize: 24 }} />;
      case 'warning':
        return <Warning sx={{ color: '#ff9800', fontSize: 24 }} />;
      case 'error':
        return <Close sx={{ color: '#f44336', fontSize: 24 }} />;
      case 'info':
      default:
        return <Info sx={{ color: '#2196f3', fontSize: 24 }} />;
    }
  };

  const getNotificationColor = (type: string) => {
    switch (type) {
      case 'success':
        return '#4caf50';
      case 'warning':
        return '#ff9800';
      case 'error':
        return '#f44336';
      case 'info':
      default:
        return '#2196f3';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <>
      <Tooltip title={isOnline ? "Notifications" : "Offline - Notifications unavailable"}>
        <IconButton
          color="inherit"
          onClick={handleClick}
          disabled={!isOnline}
          sx={{
            '&:hover': {
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
            },
          }}
        >
          <Badge 
            badgeContent={!isOnline ? <CloudOff fontSize="small" /> : unreadCount} 
            color={!isOnline ? "default" : "error"}
          >
            <Notifications />
          </Badge>
        </IconButton>
      </Tooltip>

      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleClose}
        PaperProps={{
          sx: {
            mt: 1.5,
            width: 400,
            maxHeight: 600,
            borderRadius: '12px',
            boxShadow: '0px 8px 24px rgba(0,0,0,0.15)',
          },
        }}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
      >
        {/* Header */}
        <Box sx={{ px: 2, py: 1.5, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Notifications
          </Typography>
          {unreadCount > 0 && (
            <Button size="small" onClick={markAllAsRead} sx={{ textTransform: 'none', fontSize: '0.75rem' }}>
              Mark all as read
            </Button>
          )}
        </Box>
        <Divider />

        {/* Notifications List */}
        {!notifications || notifications.length === 0 ? (
          <Box sx={{ px: 2, py: 4, textAlign: 'center' }}>
            <Notifications sx={{ fontSize: 48, color: '#bdbdbd', mb: 1 }} />
            <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
              No notifications yet
            </Typography>
            <Button 
              size="small" 
              variant="outlined" 
              onClick={() => fetchNotifications()}
              disabled={!isOnline}
            >
              Refresh
            </Button>
          </Box>
        ) : (
          <Box sx={{ maxHeight: 480, overflowY: 'auto' }}>
            {notifications.map((notification, index) => (
              <React.Fragment key={notification.id}>
                <MenuItem
                  onClick={() => {
                    if (!notification.read) {
                      markAsRead(notification.id);
                    }
                  }}
                  sx={{
                    px: 2,
                    py: 1.5,
                    display: 'block',
                    backgroundColor: notification.read ? 'transparent' : 'rgba(25, 118, 210, 0.08)',
                    '&:hover': {
                      backgroundColor: notification.read ? 'rgba(0, 0, 0, 0.04)' : 'rgba(25, 118, 210, 0.12)',
                    },
                  }}
                >
                  <Box sx={{ display: 'flex', gap: 1.5 }}>
                    <Avatar
                      sx={{
                        width: 40,
                        height: 40,
                        backgroundColor: `${getNotificationColor(notification.type)}15`,
                      }}
                    >
                      {getNotificationIcon(notification.type)}
                    </Avatar>
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 0.5 }}>
                        <Typography
                          variant="body2"
                          sx={{
                            fontWeight: notification.read ? 400 : 600,
                            color: notification.read ? 'text.secondary' : 'text.primary',
                          }}
                        >
                          {notification.title}
                        </Typography>
                        {!notification.read && (
                          <Box
                            sx={{
                              width: 8,
                              height: 8,
                              borderRadius: '50%',
                              backgroundColor: '#1976d2',
                              flexShrink: 0,
                              ml: 1,
                            }}
                          />
                        )}
                      </Box>
                      <Typography
                        variant="body2"
                        sx={{
                          fontSize: '0.875rem',
                          color: 'text.secondary',
                          mb: 0.5,
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical',
                          overflow: 'hidden',
                        }}
                      >
                        {notification.message}
                      </Typography>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                          {formatTimestamp(notification.timestamp)}
                        </Typography>
                        <Chip
                          label={notification.type}
                          size="small"
                          sx={{
                            height: 20,
                            fontSize: '0.65rem',
                            backgroundColor: `${getNotificationColor(notification.type)}15`,
                            color: getNotificationColor(notification.type),
                            fontWeight: 600,
                          }}
                        />
                      </Box>
                    </Box>
                  </Box>
                </MenuItem>
                {index < notifications.length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </Box>
        )}

        {/* Footer */}
        {notifications.length > 0 && (
          <>
            <Divider />
            <Box sx={{ px: 2, py: 1.5, textAlign: 'center' }}>
              <Button
                size="small"
                sx={{
                  textTransform: 'none',
                  fontWeight: 600,
                  color: '#1976d2',
                }}
                onClick={handleClose}
              >
                View All Notifications
              </Button>
            </Box>
          </>
        )}
      </Menu>

      {/* Error Snackbar */}
      <Snackbar 
        open={showError} 
        autoHideDuration={6000} 
        onClose={() => setShowError(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={() => setShowError(false)} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>
    </>
  );
};

export default NotificationBell;
