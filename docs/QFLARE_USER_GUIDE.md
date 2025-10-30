# QFLARE Application User Guide

## Overview

QFLARE (Quantum-Resistant Federated Learning Administration) is a professional web application for managing quantum-safe federated learning operations. The application provides secure authentication, real-time monitoring, and comprehensive device management capabilities.

## 🔐 Authentication System

### User Roles

The application supports two distinct user roles:

#### Administrator Access
- **Username**: `admin`
- **Password**: `admin123`
- **Capabilities**:
  - Full system dashboard with real-time metrics
  - Device management and monitoring
  - Training session control
  - Security scanning and compliance
  - System configuration management

#### Standard User Access  
- **Username**: `user`
- **Password**: `user123`
- **Capabilities**:
  - Personal training sessions
  - Model submission and tracking
  - Individual device management
  - Performance analytics
  - Session history

### Login Process

1. **Access the Application**: Navigate to `http://localhost:3000`
2. **Select User Role**: Use the tabs to switch between Administrator and User login
3. **Enter Credentials**: Input the appropriate username and password
4. **Sign In**: Click the "Sign In" button to authenticate

The application includes pre-filled demo credentials for easy access and testing.

## 📊 Administrator Dashboard Features

### Real-Time System Monitoring

- **Live Connection Status**: WebSocket-based real-time data updates
- **Device Metrics**: Active device count and connection status
- **Training Progress**: Current federated learning rounds and sessions
- **Performance Metrics**: Throughput (ops/sec) and latency measurements
- **Security Events**: Quantum-safe protocol monitoring

### Key Statistics Cards

1. **Devices Online**
   - Shows connected device count
   - Progress bar indicating active/total ratio
   - Real-time status updates

2. **Active Training Sessions**
   - Current federated learning activities
   - Round number indicators
   - Session progress tracking

3. **System Performance**
   - Operations per second (throughput)
   - Average latency measurements
   - Performance trend indicators

4. **Security Events**
   - Quantum-safe protocol status
   - Security incident monitoring
   - Compliance indicators

### Interactive Charts and Graphs

- **Performance Timeline**: Real-time throughput and latency charts
- **Device Distribution**: Category-based device analytics
- **Training Progress**: Federated learning round progression

## 👤 User Dashboard Features

### Personal Training Management

- **Training Sessions**: Track individual ML training progress
- **Model Submission**: Upload and manage personal models
- **Accuracy Tracking**: Monitor model performance metrics
- **Device Connection**: Manage connected personal devices

### Quick Actions

- **Start New Training**: Initiate federated learning sessions
- **Upload Models**: Submit trained models to the system
- **View Progress**: Access detailed training analytics
- **Device Settings**: Configure connected devices

## 🔧 Technical Features

### Backend Integration

- **RESTful API**: Professional FastAPI backend with comprehensive endpoints
- **Real-Time Updates**: WebSocket connections for live data streaming
- **Secure Authentication**: Token-based authentication with session management
- **Error Handling**: Robust error management with user-friendly messages

### Frontend Technologies

- **React TypeScript**: Modern, type-safe frontend framework
- **Material-UI**: Professional component library with consistent design
- **Real-Time Hooks**: Custom hooks for live data fetching and WebSocket management
- **Responsive Design**: Mobile-friendly interface that adapts to different screen sizes

### Security Features

- **Quantum-Safe Protocols**: Post-quantum cryptographic implementation
- **Role-Based Access**: Distinct capabilities based on user roles
- **Session Management**: Secure token storage and validation
- **Connection Security**: Encrypted communication channels

## 🚀 Getting Started

### Prerequisites

- Node.js and npm installed
- Python environment with required packages
- Access to the QFLARE backend server

### Quick Start

1. **Backend Server**: Ensure backend is running on `http://localhost:8000`
2. **Frontend Application**: Access the web interface at `http://localhost:3000`
3. **Login**: Use demo credentials (admin/admin123 or user/user123)
4. **Explore**: Navigate through the dashboard features

### Connection Status Indicators

- **Green Wi-Fi Icon**: Real-time connection active
- **Red Wi-Fi Icon**: Connection issues detected
- **Loading Indicators**: Data being fetched from backend
- **Error Messages**: Connection or authentication problems

## 📱 User Interface Guide

### Navigation

- **Tab-Based Login**: Switch between Administrator and User access
- **Sidebar Navigation**: Easy access to different application sections
- **Breadcrumb Navigation**: Track your current location in the app

### Visual Feedback

- **Loading States**: Skeleton placeholders during data fetching
- **Success Indicators**: Green checkmarks and success messages
- **Warning Alerts**: Yellow indicators for attention items
- **Error Notifications**: Red alerts for critical issues

### Data Visualization

- **Real-Time Charts**: Live updating performance graphs
- **Progress Bars**: Visual progress indicators for various metrics
- **Status Cards**: Color-coded information cards
- **Activity Lists**: Chronological activity feeds

## 🔍 Troubleshooting

### Common Issues

#### Login Problems
- **Invalid Credentials Error**: Verify username/password combination
- **Connection Error**: Check backend server status (port 8000)
- **Network Issues**: Ensure stable internet connection

#### Dashboard Issues
- **No Real-Time Data**: Check WebSocket connection status
- **Loading Forever**: Refresh the page or check backend connectivity
- **Missing Features**: Verify user role and permissions

#### Performance Issues
- **Slow Loading**: Check network connection and backend server performance
- **Frequent Disconnections**: Verify WebSocket stability
- **Outdated Data**: Use refresh button or reload the application

### Support

For technical support or questions about the QFLARE application:

1. Check the real-time connection status indicator
2. Verify backend server is running (http://localhost:8000/health)
3. Review browser console for error messages
4. Contact system administrator for access issues

## 🎯 Best Practices

### For Administrators

- Monitor connection status regularly
- Review security events frequently
- Manage device connections proactively
- Keep track of training session progress

### For Users

- Ensure devices are properly connected
- Monitor training session accuracy
- Keep personal credentials secure
- Report any unusual system behavior

## 📈 Analytics and Reporting

### Real-Time Metrics

- Live system performance data
- Instant device status updates
- Training progress notifications
- Security event alerts

### Historical Data

- Performance trend analysis
- Training session history
- Device connection logs
- Security audit trails

---

**QFLARE** - Quantum-Resistant Federated Learning Administration Platform
Version 1.0.0 | Professional Edition