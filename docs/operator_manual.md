# QFLARE Operator Training Manual

## Overview

This manual provides comprehensive training for QFLARE system operators, covering daily operations, monitoring, troubleshooting, and emergency procedures.

## Getting Started

### Access Requirements

Before operating QFLARE systems, ensure you have:

- Valid user account with appropriate permissions
- VPN access to production networks
- Multi-factor authentication configured
- Emergency contact information
- Escalation procedures documentation

### Initial System Login

1. **Access the Admin Panel**
   ```
   URL: https://qflare.company.com/admin
   Username: [your-username]
   Password: [secure-password]
   MFA Token: [from authenticator app]
   ```

2. **Verify System Status**
   - Check overall system health on Overview tab
   - Review active alerts and warnings
   - Confirm all critical services are running
   - Validate monitoring systems are operational

### Daily Operations Checklist

#### Morning Routine (Start of Shift)

- [ ] Log into admin panel and verify access
- [ ] Review overnight alerts and incidents
- [ ] Check system health metrics
- [ ] Verify backup completion status
- [ ] Review security events from last 24 hours
- [ ] Confirm all devices are online and healthy
- [ ] Check federated learning training status

#### Hourly Checks

- [ ] Monitor system performance metrics
- [ ] Review active training rounds
- [ ] Check device connectivity status
- [ ] Verify quantum key rotation schedule
- [ ] Monitor resource utilization levels

#### End of Shift Routine

- [ ] Document any issues encountered
- [ ] Complete incident reports
- [ ] Hand off open issues to next shift
- [ ] Update operation logs
- [ ] Verify backup schedules are on track

## System Monitoring

### Key Metrics to Monitor

#### Performance Metrics
- **API Response Time**: Should be < 200ms average
- **Database Connections**: Monitor for connection pool exhaustion
- **Memory Usage**: Alert if > 85% consistently
- **CPU Usage**: Alert if > 80% for extended periods
- **Disk Usage**: Alert if > 90% on any volume

#### Business Metrics
- **Active Devices**: Total and by organization
- **Training Rounds**: Success rate and duration
- **User Activity**: Login frequency and patterns
- **Data Throughput**: Volume of federated learning data

### Monitoring Tools

#### Grafana Dashboards

1. **System Overview Dashboard**
   - Access: `https://monitoring.qflare.company.com/d/system-overview`
   - Key panels: CPU, Memory, Network, Storage
   - Refresh rate: 30 seconds
   - Time range: Last 4 hours (adjustable)

2. **Application Performance Dashboard**
   - Access: `https://monitoring.qflare.company.com/d/app-performance`
   - Key panels: Response times, Error rates, Throughput
   - Alert thresholds: Response time > 1s, Error rate > 1%

3. **Federated Learning Dashboard**
   - Access: `https://monitoring.qflare.company.com/d/fl-training`
   - Key panels: Active rounds, Device participation, Model accuracy
   - Training metrics: Convergence rates, Communication overhead

#### Alert Management

**Alert Severity Levels:**
- **P0 (Critical)**: System down, data loss risk
- **P1 (High)**: Degraded performance, some features unavailable
- **P2 (Medium)**: Minor issues, monitoring required
- **P3 (Low)**: Informational, no immediate action needed

**Response Times:**
- P0: Immediate response (< 5 minutes)
- P1: Within 30 minutes
- P2: Within 2 hours
- P3: Next business day

### Common Monitoring Scenarios

#### High Memory Usage Alert

1. **Immediate Actions:**
   ```bash
   # Check memory usage by service
   kubectl top pods -n qflare-production
   
   # Identify memory-intensive processes
   kubectl exec -it <pod-name> -- ps aux --sort=-%mem | head -10
   ```

2. **Investigation Steps:**
   - Check for memory leaks in application logs
   - Review recent deployments or configuration changes
   - Analyze memory usage trends over time
   - Consider scaling up resources if legitimate increase

3. **Resolution Actions:**
   - Restart affected services if memory leak detected
   - Scale up pod resources temporarily
   - Implement permanent fix if root cause identified

#### Database Connection Issues

1. **Immediate Actions:**
   ```bash
   # Check database connectivity
   kubectl exec -it postgres-0 -n qflare-production -- pg_isready
   
   # Review connection counts
   kubectl exec -it postgres-0 -n qflare-production -- \
   psql -U qflare -c "SELECT count(*) FROM pg_stat_activity;"
   ```

2. **Investigation Steps:**
   - Check for connection pool exhaustion
   - Review database logs for errors
   - Analyze query performance for slow queries
   - Check for database locks or deadlocks

3. **Resolution Actions:**
   - Restart application pods to reset connections
   - Increase connection pool size if needed
   - Optimize slow queries
   - Scale database resources if necessary

## Device Management

### Device Registration Process

1. **Generate Enrollment Token**
   ```bash
   # Via admin panel or CLI
   python scripts/generate_token.py --organization "Org Name" --expires-hours 24
   ```

2. **Device Onboarding Steps**
   - Provide enrollment token to device owner
   - Device uses token to register with platform
   - System validates device capabilities
   - Quantum keys are generated and distributed
   - Device appears in monitoring dashboard

3. **Post-Registration Verification**
   - Confirm device appears in device list
   - Verify quantum key exchange completed
   - Test basic connectivity and functionality
   - Add device to appropriate training groups

### Device Health Monitoring

#### Key Device Metrics
- **Connectivity Status**: Online/Offline status
- **Last Heartbeat**: Recent communication timestamp
- **Training Participation**: Active in current rounds
- **Resource Utilization**: CPU, Memory, Storage usage
- **Security Status**: Certificate validity, key freshness

#### Device Troubleshooting

**Device Offline Issues:**
1. Check network connectivity from device
2. Verify firewall rules allow QFLARE traffic
3. Confirm device certificates haven't expired
4. Check device logs for connection errors
5. Restart device services if necessary

**Training Participation Issues:**
1. Verify device meets minimum requirements
2. Check device is in correct training group
3. Review device resource availability
4. Confirm no conflicting training jobs

### Device Lifecycle Management

#### Device Updates
- Monitor for available updates
- Schedule maintenance windows
- Apply updates in rolling fashion
- Verify functionality post-update
- Document any issues encountered

#### Device Decommissioning
1. Remove from active training groups
2. Revoke device certificates
3. Clear quantum keys from system
4. Update device inventory
5. Securely wipe device if returned

## Security Operations

### Quantum Key Management

#### Key Rotation Schedule
- **Automatic Rotation**: Every 30 days
- **Emergency Rotation**: On security incident
- **Manual Rotation**: Monthly verification

#### Key Rotation Process
1. **Pre-Rotation Checks**
   ```bash
   # Check current key status
   python scripts/check_keys.py --environment production
   
   # Verify all devices are online
   python scripts/device_status.py --check-connectivity
   ```

2. **Execute Rotation**
   ```bash
   # Initiate key rotation
   python scripts/rotate_keys.py --environment production --confirm
   ```

3. **Post-Rotation Verification**
   - Confirm all devices received new keys
   - Test encrypted communications
   - Verify training can continue
   - Update rotation logs

#### Security Incident Response

**Incident Classification:**
- **Level 1**: Suspected security breach
- **Level 2**: Confirmed unauthorized access
- **Level 3**: Data compromise or system compromise
- **Level 4**: Critical infrastructure attack

**Response Procedures:**

1. **Immediate Actions (All Levels)**
   - Document incident time and details
   - Notify security team and management
   - Preserve evidence and logs
   - Do not modify affected systems

2. **Level 1 Response**
   - Increase monitoring frequency
   - Review access logs for anomalies
   - Verify user account integrity
   - Monitor for escalation indicators

3. **Level 2+ Response**
   - Isolate affected systems
   - Force quantum key rotation
   - Revoke suspicious user sessions
   - Implement additional monitoring
   - Engage external security team if needed

### Security Monitoring

#### Security Event Types
- **Authentication Events**: Failed logins, suspicious patterns
- **Access Events**: Unauthorized resource access attempts
- **Network Events**: Unusual traffic patterns, port scans
- **System Events**: Privilege escalations, file modifications

#### Log Analysis
```bash
# Search for failed authentication attempts
kubectl logs -n qflare-production deployment/qflare-api | \
grep "authentication failed" | tail -20

# Check for suspicious IP addresses
kubectl logs -n qflare-production deployment/qflare-api | \
grep -E "403|401" | awk '{print $1}' | sort | uniq -c | sort -nr

# Monitor quantum key events
kubectl logs -n qflare-production deployment/qflare-api | \
grep "quantum_key" | tail -10
```

## Federated Learning Operations

### Training Round Management

#### Starting a Training Round
1. **Pre-Training Checks**
   - Verify sufficient devices are online
   - Check model repository is accessible
   - Confirm training data is available
   - Validate compute resources

2. **Round Configuration**
   ```json
   {
     "model_config": {
       "architecture": "cnn",
       "hyperparameters": {
         "learning_rate": 0.001,
         "batch_size": 32,
         "epochs": 10
       }
     },
     "participant_criteria": {
       "min_participants": 10,
       "device_requirements": {
         "min_memory_gb": 4,
         "min_cpu_cores": 2
       }
     }
   }
   ```

3. **Round Execution**
   - Initiate training via admin panel
   - Monitor device participation
   - Track training progress
   - Handle device disconnections

#### Training Monitoring

**Key Metrics:**
- **Participation Rate**: Devices actively training vs. invited
- **Progress Tracking**: Percentage completion per device
- **Model Convergence**: Accuracy improvements over rounds
- **Communication Efficiency**: Data transfer optimization

**Common Issues:**
- **Low Participation**: Check device availability and requirements
- **Slow Convergence**: Review hyperparameters and data quality
- **Device Failures**: Investigate hardware or network issues
- **Security Alerts**: Verify quantum key integrity

### Model Management

#### Model Versioning
- Track model versions across training rounds
- Maintain model performance metrics
- Store model artifacts securely
- Enable rollback to previous versions

#### Model Deployment
1. **Validation Phase**
   - Test model on validation dataset
   - Verify model performance metrics
   - Check for potential bias or issues
   - Approve for deployment

2. **Deployment Process**
   - Package model with deployment config
   - Deploy to staging environment first
   - Run integration tests
   - Deploy to production with monitoring

## Troubleshooting Guide

### Common Issues and Solutions

#### API Performance Issues

**Symptoms:**
- Slow response times (> 2 seconds)
- Timeouts from frontend applications
- High error rates in monitoring

**Diagnosis:**
```bash
# Check API pod resource usage
kubectl top pods -n qflare-production -l app=qflare-api

# Review API logs for errors
kubectl logs -n qflare-production deployment/qflare-api --tail=100

# Check database performance
kubectl exec -it postgres-0 -n qflare-production -- \
psql -U qflare -c "SELECT query, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
```

**Solutions:**
- Scale API pods horizontally
- Optimize database queries
- Implement caching where appropriate
- Review and optimize business logic

#### Database Connection Problems

**Symptoms:**
- "Connection refused" errors
- "Too many connections" errors
- Application unable to start

**Diagnosis:**
```bash
# Check database pod status
kubectl get pods -n qflare-production -l app=postgres

# Test database connectivity
kubectl exec -it postgres-0 -n qflare-production -- pg_isready

# Check connection count
kubectl exec -it postgres-0 -n qflare-production -- \
psql -U qflare -c "SELECT count(*) FROM pg_stat_activity;"
```

**Solutions:**
- Restart database pod if crashed
- Increase max_connections in PostgreSQL config
- Optimize connection pooling settings
- Kill long-running queries if necessary

#### Device Connectivity Issues

**Symptoms:**
- Devices showing as offline
- Failed training participation
- Certificate validation errors

**Diagnosis:**
```bash
# Check device status via API
curl -H "Authorization: Bearer $TOKEN" \
https://qflare.company.com/api/v1/devices

# Review device logs (if accessible)
# Check firewall rules
# Verify DNS resolution
```

**Solutions:**
- Restart device services
- Update device certificates
- Check network configuration
- Force quantum key refresh

### Emergency Procedures

#### System Outage Response

**Immediate Actions (First 5 minutes):**
1. Confirm outage scope via monitoring
2. Check infrastructure status (servers, network)
3. Notify incident response team
4. Update status page for users
5. Begin logging all actions taken

**Investigation Phase (5-30 minutes):**
1. Review recent changes or deployments
2. Check logs for error patterns
3. Verify database and service status
4. Test connectivity to external dependencies
5. Identify root cause if possible

**Recovery Phase:**
1. Implement fix based on root cause
2. Restart services in proper order
3. Verify functionality before declaring recovery
4. Update stakeholders on status
5. Begin post-incident review process

#### Data Loss Prevention

**Backup Verification:**
```bash
# Check recent backup status
python scripts/verify_backups.py --environment production --days 7

# Test backup restoration (in staging)
python scripts/restore_backup.py --environment staging --backup latest
```

**Recovery Procedures:**
1. Stop all write operations immediately
2. Assess extent of data loss
3. Restore from most recent clean backup
4. Replay transaction logs if available
5. Verify data integrity post-recovery

## Maintenance Procedures

### Scheduled Maintenance

#### Weekly Maintenance (Weekends)
- Apply security patches to operating systems
- Update container images to latest versions
- Review and cleanup old log files
- Verify backup integrity
- Test disaster recovery procedures

#### Monthly Maintenance
- Full system security scan
- Performance optimization review
- Capacity planning assessment
- Update documentation
- Review and update monitoring thresholds

#### Quarterly Maintenance
- Major version updates
- Security audit and penetration testing
- Disaster recovery testing
- Staff training updates
- Compliance review

### Maintenance Windows

#### Planning Maintenance
1. Schedule maintenance window (typically 2-4 hours)
2. Notify all stakeholders 1 week in advance
3. Prepare rollback procedures
4. Test maintenance procedures in staging
5. Prepare communication templates

#### During Maintenance
1. Enable maintenance mode
2. Drain traffic from services
3. Apply updates systematically
4. Test functionality at each step
5. Monitor for issues continuously

#### Post-Maintenance
1. Verify all services are operational
2. Run comprehensive health checks
3. Monitor performance for 24 hours
4. Document any issues encountered
5. Update maintenance procedures if needed

## Performance Optimization

### Monitoring Performance

#### Key Performance Indicators (KPIs)
- API response time: < 200ms average
- Database query time: < 100ms average
- Training round completion: < 2 hours
- Device onboarding time: < 5 minutes
- System uptime: > 99.9%

#### Performance Profiling
```bash
# Profile API performance
curl -w "@curl-format.txt" -o /dev/null -s \
https://qflare.company.com/api/v1/devices

# Database query analysis
kubectl exec -it postgres-0 -n qflare-production -- \
psql -U qflare -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"

# Resource utilization
kubectl top nodes
kubectl top pods -n qflare-production
```

### Optimization Strategies

#### Database Optimization
- Regularly update table statistics
- Implement proper indexing strategy
- Monitor and optimize slow queries
- Use connection pooling effectively
- Consider read replicas for reporting

#### Application Optimization
- Implement caching strategies
- Optimize API endpoints
- Use async processing for heavy operations
- Implement proper error handling
- Monitor memory usage patterns

#### Infrastructure Optimization
- Right-size resource allocations
- Implement auto-scaling policies
- Optimize network configurations
- Use content delivery networks
- Implement load balancing strategies

## Compliance and Auditing

### Audit Requirements

#### Regular Audits
- Security audit: Monthly
- Performance audit: Quarterly
- Compliance audit: Annually
- Data governance audit: Bi-annually

#### Audit Preparation
1. Gather required documentation
2. Prepare system access for auditors
3. Export relevant logs and metrics
4. Review compliance status
5. Prepare remediation plans

### Compliance Monitoring

#### Data Protection Compliance
- Monitor data access patterns
- Verify encryption at rest and in transit
- Track data retention policies
- Document data processing activities
- Maintain consent records

#### Security Compliance
- Regular vulnerability scans
- Patch management tracking
- Access control reviews
- Incident response documentation
- Security training records

## Emergency Contacts

### Internal Contacts
- **Operations Team Lead**: ops-lead@company.com, +1-555-0101
- **Security Team**: security@company.com, +1-555-0102
- **Engineering Manager**: eng-manager@company.com, +1-555-0103
- **CTO**: cto@company.com, +1-555-0104

### External Contacts
- **Cloud Provider Support**: [Provider-specific contact]
- **Security Incident Response**: incident-response@company.com
- **Legal Team**: legal@company.com
- **Customer Support**: support@company.com

### Escalation Matrix

| Severity | First Contact | Escalation 1 | Escalation 2 | Escalation 3 |
|----------|---------------|--------------|--------------|--------------|
| P0 | Operations Team | Engineering Manager | CTO | CEO |
| P1 | Operations Team | Engineering Manager | CTO | - |
| P2 | Operations Team | Engineering Manager | - | - |
| P3 | Operations Team | - | - | - |

## Additional Resources

### Documentation
- [API Documentation](./api_documentation.md)
- [Deployment Guide](./deployment_guide.md)
- [Security Guidelines](../security/README.md)
- [Troubleshooting Guide](../TROUBLESHOOTING.md)

### Training Materials
- QFLARE Architecture Overview
- Quantum Cryptography Fundamentals
- Federated Learning Concepts
- Kubernetes Operations
- Security Best Practices

### Tools and Utilities
- Monitoring dashboards
- Log analysis tools
- Performance profiling tools
- Security scanning tools
- Backup and recovery tools

---

**Document Version**: 1.0
**Last Updated**: January 2024
**Next Review**: April 2024

For questions or suggestions regarding this manual, contact the Operations Team at ops@company.com.