import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Box,
  Typography,
  Divider,
  Chip,
  Avatar,
  Button
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Devices as DevicesIcon,
  ModelTraining as MLIcon,
  Security as SecurityIcon,
  Monitor as MonitoringIcon,
  Settings as SettingsIcon,
  AddCircle as AddIcon,
  AdminPanelSettings as AdminIcon,
  Person as PersonIcon,
  Logout as LogoutIcon
} from '@mui/icons-material';
import { useAuth } from '../../contexts/AuthContext';

interface SidebarProps {
  drawerWidth: number;
  mobileOpen: boolean;
  onDrawerToggle: () => void;
}

// Admin menu items
const adminMenuItems = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
  { text: 'Devices', icon: <DevicesIcon />, path: '/devices' },
  { text: 'Register Device', icon: <AddIcon />, path: '/devices/register' },
  { text: 'Federated Learning', icon: <MLIcon />, path: '/federated-learning' },
  { text: 'Security', icon: <SecurityIcon />, path: '/security' },
  { text: 'Monitoring', icon: <MonitoringIcon />, path: '/monitoring' },
  { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
];

// User menu items
const userMenuItems = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
  { text: 'Federated Learning', icon: <MLIcon />, path: '/federated-learning' },
];

const Sidebar = ({ drawerWidth, mobileOpen, onDrawerToggle }: SidebarProps) => {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, isAdmin, logout } = useAuth();

  // Get menu items based on user role
  const menuItems = isAdmin ? adminMenuItems : userMenuItems;

  const handleNavigation = (path: string) => {
    navigate(path);
    if (mobileOpen) {
      onDrawerToggle();
    }
  };

  const drawer = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Toolbar>
        <Box sx={{ display: 'flex', flexDirection: 'column', width: '100%' }}>
          <Typography variant="h6" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
            QFLARE
          </Typography>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            Quantum Federated Learning
          </Typography>
        </Box>
      </Toolbar>
      
      <Divider />
      
      {/* User Profile Section */}
      <Box sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Avatar sx={{ 
            width: 40, 
            height: 40, 
            mr: 2,
            backgroundColor: isAdmin ? 'primary.main' : 'secondary.main'
          }}>
            {isAdmin ? <AdminIcon /> : <PersonIcon />}
          </Avatar>
          <Box sx={{ flex: 1 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
              {user?.name || 'User'}
            </Typography>
            <Chip 
              label={isAdmin ? 'Administrator' : 'User'} 
              size="small" 
              color={isAdmin ? 'primary' : 'secondary'}
              variant="outlined"
            />
          </Box>
        </Box>
      </Box>
      
      <Divider />
      
      {/* Navigation Menu */}
      <List sx={{ px: 2, py: 1, flex: 1 }}>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding sx={{ mb: 0.5 }}>
            <ListItemButton
              onClick={() => handleNavigation(item.path)}
              selected={location.pathname === item.path}
              sx={{
                borderRadius: 2,
                '&.Mui-selected': {
                  backgroundColor: 'primary.main',
                  color: 'white',
                  '&:hover': {
                    backgroundColor: 'primary.dark',
                  },
                  '& .MuiListItemIcon-root': {
                    color: 'white',
                  },
                },
                '&:hover': {
                  backgroundColor: 'primary.light',
                  color: 'white',
                  '& .MuiListItemIcon-root': {
                    color: 'white',
                  },
                },
              }}
            >
              <ListItemIcon sx={{ minWidth: 40 }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText 
                primary={item.text}
                primaryTypographyProps={{
                  fontSize: '0.875rem',
                  fontWeight: location.pathname === item.path ? 600 : 400,
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Divider sx={{ mx: 2 }} />

      {/* Logout Button */}
      <Box sx={{ p: 2 }}>
        <Button
          fullWidth
          variant="outlined"
          color="error"
          startIcon={<LogoutIcon />}
          onClick={logout}
          sx={{ 
            borderRadius: 2,
            textTransform: 'none',
            justifyContent: 'flex-start'
          }}
        >
          Sign Out
        </Button>
      </Box>

      <Divider sx={{ mx: 2 }} />

      <Box sx={{ p: 2 }}>
        <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
          System Status
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          <Chip 
            label="Quantum Safe" 
            color="success" 
            size="small" 
            variant="outlined"
          />
          <Chip 
            label="39 Devices Online" 
            color="info" 
            size="small" 
            variant="outlined"
          />
          <Chip 
            label="FL Active" 
            color="primary" 
            size="small" 
            variant="outlined"
          />
        </Box>
      </Box>
    </Box>
  );

  return (
    <Box
      component="nav"
      sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
    >
      {/* Mobile drawer */}
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={onDrawerToggle}
        ModalProps={{
          keepMounted: true, // Better mobile performance
        }}
        sx={{
          display: { xs: 'block', sm: 'none' },
          '& .MuiDrawer-paper': {
            boxSizing: 'border-box',
            width: drawerWidth,
          },
        }}
      >
        {drawer}
      </Drawer>

      {/* Desktop drawer */}
      <Drawer
        variant="permanent"
        sx={{
          display: { xs: 'none', sm: 'block' },
          '& .MuiDrawer-paper': {
            boxSizing: 'border-box',
            width: drawerWidth,
          },
        }}
        open
      >
        {drawer}
      </Drawer>
    </Box>
  );
};

export default Sidebar;