import React from 'react';
import {
  Box,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Avatar
} from '@mui/material';
import {
  ModelTraining as TrainingIcon,
  CloudUpload as UploadIcon,
  Timeline as TimelineIcon,
  Devices as DeviceIcon,
  CheckCircle as CheckIcon,
  Schedule as ScheduleIcon,
  Person as PersonIcon
} from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';

const UserDashboard = () => {
  const { user } = useAuth();

  const userStats = [
    { label: 'Training Sessions', value: '12', icon: <TrainingIcon /> },
    { label: 'Models Submitted', value: '8', icon: <UploadIcon /> },
    { label: 'Accuracy Score', value: '94.2%', icon: <TimelineIcon /> },
    { label: 'Devices Connected', value: '2', icon: <DeviceIcon /> },
  ];

  const recentSessions = [
    { id: 1, name: 'Image Classification', status: 'Completed', accuracy: '92.5%', date: '2 hours ago' },
    { id: 2, name: 'Text Processing', status: 'In Progress', accuracy: '89.1%', date: '1 day ago' },
    { id: 3, name: 'Sentiment Analysis', status: 'Completed', accuracy: '91.8%', date: '3 days ago' },
  ];

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Welcome Section */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Welcome back, {user?.name}!
        </Typography>
        <Typography variant="body1" color="textSecondary">
          Here's your federated learning activity summary
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* User Stats Cards */}
        {userStats.map((stat, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Avatar sx={{ backgroundColor: 'primary.main', mr: 2 }}>
                    {stat.icon}
                  </Avatar>
                  <Box>
                    <Typography variant="h5" component="div">
                      {stat.value}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      {stat.label}
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}

        {/* Quick Actions */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Button
                  variant="contained"
                  startIcon={<TrainingIcon />}
                  fullWidth
                  sx={{ justifyContent: 'flex-start' }}
                >
                  Start New Training Session
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<UploadIcon />}
                  fullWidth
                  sx={{ justifyContent: 'flex-start' }}
                >
                  Upload Model
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<DeviceIcon />}
                  fullWidth
                  sx={{ justifyContent: 'flex-start' }}
                >
                  View My Devices
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Training Sessions */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Training Sessions
              </Typography>
              <List>
                {recentSessions.map((session) => (
                  <ListItem key={session.id} divider>
                    <ListItemIcon>
                      {session.status === 'Completed' ? (
                        <CheckIcon color="success" />
                      ) : (
                        <ScheduleIcon color="warning" />
                      )}
                    </ListItemIcon>
                    <ListItemText
                      primary={session.name}
                      secondary={
                        <Box>
                          <Typography variant="body2" component="span">
                            Accuracy: {session.accuracy} • {session.date}
                          </Typography>
                          <Box sx={{ mt: 1 }}>
                            <Chip
                              label={session.status}
                              size="small"
                              color={session.status === 'Completed' ? 'success' : 'warning'}
                              variant="outlined"
                            />
                          </Box>
                        </Box>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Current Training Progress */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Current Training Progress
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="textSecondary">
                  Text Processing Model
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={65} 
                  sx={{ mt: 1, mb: 1 }} 
                />
                <Typography variant="body2">
                  Round 13 of 20 • 65% Complete
                </Typography>
              </Box>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="textSecondary">
                  Current Accuracy: 89.1%
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Estimated completion: 2 hours
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Personal Performance */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Your Performance
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box>
                  <Typography variant="body2" color="textSecondary">
                    Average Model Accuracy
                  </Typography>
                  <Typography variant="h5" color="primary">
                    91.5%
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="body2" color="textSecondary">
                    Contribution Score
                  </Typography>
                  <Typography variant="h5" color="success.main">
                    Excellent
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="body2" color="textSecondary">
                    Training Hours This Month
                  </Typography>
                  <Typography variant="h5">
                    42.5 hrs
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default UserDashboard;