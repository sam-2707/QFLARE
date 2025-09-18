import React, { useState, ChangeEvent } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Button,
  TextField,
  Switch,
  FormControl,
  FormControlLabel,
  InputLabel,
  Select,
  MenuItem,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Alert,
  Tabs,
  Tab,
  SelectChangeEvent
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Security as SecurityIcon,
  Notifications as NotificationsIcon,
  Person as PersonIcon,
  Save as SaveIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <Box
      role="tabpanel"
      hidden={value !== index}
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </Box>
  );
}

const Settings = () => {
  const [currentTab, setCurrentTab] = useState(0);
  const [settings, setSettings] = useState({
    // General settings
    organizationName: 'Research Institute A',
    adminEmail: 'admin@company.com',
    timezone: 'UTC',
    
    // Security settings
    sessionTimeout: 30,
    requireMFA: true,
    passwordComplexity: 'high',
    keyRotationInterval: 30,
    
    // Notification settings
    emailNotifications: true,
    securityAlerts: true,
    performanceAlerts: false,
    maintenanceNotifications: true,
    
    // Training settings
    maxTrainingRounds: 100,
    defaultBatchSize: 32,
    defaultLearningRate: 0.001,
    trainingTimeout: 3600
  });

  const [showSaveAlert, setShowSaveAlert] = useState(false);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const handleSettingChange = (key: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleSave = () => {
    // Here you would save settings to the backend
    console.log('Saving settings:', settings);
    setShowSaveAlert(true);
    setTimeout(() => setShowSaveAlert(false), 3000);
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>

      {showSaveAlert && (
        <Alert severity="success" sx={{ mb: 3 }}>
          Settings saved successfully!
        </Alert>
      )}

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={currentTab} onChange={handleTabChange}>
          <Tab label="General" icon={<SettingsIcon />} />
          <Tab label="Security" icon={<SecurityIcon />} />
          <Tab label="Notifications" icon={<NotificationsIcon />} />
          <Tab label="Training" icon={<PersonIcon />} />
        </Tabs>
      </Box>

      {/* General Settings */}
      <TabPanel value={currentTab} index={0}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Organization Settings
              </Typography>
              <TextField
                fullWidth
                label="Organization Name"
                value={settings.organizationName}
                onChange={(e: ChangeEvent<HTMLInputElement>) => handleSettingChange('organizationName', e.target.value)}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                label="Admin Email"
                type="email"
                value={settings.adminEmail}
                onChange={(e: ChangeEvent<HTMLInputElement>) => handleSettingChange('adminEmail', e.target.value)}
                sx={{ mb: 2 }}
              />
              <FormControl fullWidth>
                <InputLabel>Timezone</InputLabel>
                <Select
                  value={settings.timezone}
                  label="Timezone"
                  onChange={(e: SelectChangeEvent<string>) => handleSettingChange('timezone', e.target.value)}
                >
                  <MenuItem value="UTC">UTC</MenuItem>
                  <MenuItem value="EST">Eastern Time</MenuItem>
                  <MenuItem value="PST">Pacific Time</MenuItem>
                  <MenuItem value="CET">Central European Time</MenuItem>
                </Select>
              </FormControl>
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                System Information
              </Typography>
              <List>
                <ListItem>
                  <ListItemText
                    primary="QFLARE Version"
                    secondary="v1.2.0"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Last Updated"
                    secondary="January 15, 2024"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="License Type"
                    secondary="Enterprise"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Support Status"
                    secondary="Active"
                  />
                </ListItem>
              </List>
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Security Settings */}
      <TabPanel value={currentTab} index={1}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Authentication & Access
              </Typography>
              <TextField
                fullWidth
                label="Session Timeout (minutes)"
                type="number"
                value={settings.sessionTimeout}
                onChange={(e: ChangeEvent<HTMLInputElement>) => handleSettingChange('sessionTimeout', parseInt(e.target.value))}
                sx={{ mb: 2 }}
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.requireMFA}
                    onChange={(e: ChangeEvent<HTMLInputElement>) => handleSettingChange('requireMFA', e.target.checked)}
                  />
                }
                label="Require Multi-Factor Authentication"
                sx={{ mb: 2 }}
              />
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Password Complexity</InputLabel>
                <Select
                  value={settings.passwordComplexity}
                  label="Password Complexity"
                  onChange={(e: SelectChangeEvent<string>) => handleSettingChange('passwordComplexity', e.target.value)}
                >
                  <MenuItem value="low">Low</MenuItem>
                  <MenuItem value="medium">Medium</MenuItem>
                  <MenuItem value="high">High</MenuItem>
                </Select>
              </FormControl>
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Quantum Key Management
              </Typography>
              <TextField
                fullWidth
                label="Key Rotation Interval (days)"
                type="number"
                value={settings.keyRotationInterval}
                onChange={(e: ChangeEvent<HTMLInputElement>) => handleSettingChange('keyRotationInterval', parseInt(e.target.value))}
                sx={{ mb: 2 }}
              />
              <Button
                variant="outlined"
                startIcon={<RefreshIcon />}
                sx={{ mb: 2 }}
              >
                Force Key Rotation
              </Button>
              <Typography variant="body2" color="textSecondary">
                Next automatic rotation: In 18 hours
              </Typography>
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Notification Settings */}
      <TabPanel value={currentTab} index={2}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Email Notifications
              </Typography>
              <List>
                <ListItem>
                  <ListItemText primary="General Notifications" />
                  <ListItemSecondaryAction>
                    <Switch
                      checked={settings.emailNotifications}
                      onChange={(e: ChangeEvent<HTMLInputElement>) => handleSettingChange('emailNotifications', e.target.checked)}
                    />
                  </ListItemSecondaryAction>
                </ListItem>
                <ListItem>
                  <ListItemText primary="Security Alerts" />
                  <ListItemSecondaryAction>
                    <Switch
                      checked={settings.securityAlerts}
                      onChange={(e: ChangeEvent<HTMLInputElement>) => handleSettingChange('securityAlerts', e.target.checked)}
                    />
                  </ListItemSecondaryAction>
                </ListItem>
                <ListItem>
                  <ListItemText primary="Performance Alerts" />
                  <ListItemSecondaryAction>
                    <Switch
                      checked={settings.performanceAlerts}
                      onChange={(e: ChangeEvent<HTMLInputElement>) => handleSettingChange('performanceAlerts', e.target.checked)}
                    />
                  </ListItemSecondaryAction>
                </ListItem>
                <ListItem>
                  <ListItemText primary="Maintenance Notifications" />
                  <ListItemSecondaryAction>
                    <Switch
                      checked={settings.maintenanceNotifications}
                      onChange={(e: ChangeEvent<HTMLInputElement>) => handleSettingChange('maintenanceNotifications', e.target.checked)}
                    />
                  </ListItemSecondaryAction>
                </ListItem>
              </List>
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Alert Preferences
              </Typography>
              <TextField
                fullWidth
                label="Alert Email Address"
                type="email"
                value={settings.adminEmail}
                sx={{ mb: 2 }}
                disabled
              />
              <Typography variant="body2" color="textSecondary">
                Alerts will be sent to the admin email address configured in General settings.
              </Typography>
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Training Settings */}
      <TabPanel value={currentTab} index={3}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Training Configuration
              </Typography>
              <TextField
                fullWidth
                label="Maximum Training Rounds"
                type="number"
                value={settings.maxTrainingRounds}
                onChange={(e: ChangeEvent<HTMLInputElement>) => handleSettingChange('maxTrainingRounds', parseInt(e.target.value))}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                label="Default Batch Size"
                type="number"
                value={settings.defaultBatchSize}
                onChange={(e: ChangeEvent<HTMLInputElement>) => handleSettingChange('defaultBatchSize', parseInt(e.target.value))}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                label="Default Learning Rate"
                type="number"
                inputProps={{ step: 0.001 }}
                value={settings.defaultLearningRate}
                onChange={(e: ChangeEvent<HTMLInputElement>) => handleSettingChange('defaultLearningRate', parseFloat(e.target.value))}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                label="Training Timeout (seconds)"
                type="number"
                value={settings.trainingTimeout}
                onChange={(e: ChangeEvent<HTMLInputElement>) => handleSettingChange('trainingTimeout', parseInt(e.target.value))}
              />
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Resource Limits
              </Typography>
              <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                Configure resource limits for federated learning training.
              </Typography>
              <TextField
                fullWidth
                label="Max Concurrent Rounds"
                type="number"
                defaultValue={5}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                label="Max Devices per Round"
                type="number"
                defaultValue={100}
                sx={{ mb: 2 }}
              />
              <TextField
                fullWidth
                label="Memory Limit per Device (GB)"
                type="number"
                defaultValue={4}
              />
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Save Button */}
      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="contained"
          startIcon={<SaveIcon />}
          onClick={handleSave}
          size="large"
        >
          Save Settings
        </Button>
      </Box>
    </Container>
  );
};

export default Settings;