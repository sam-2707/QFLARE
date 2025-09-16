-- ================================================================
-- QFLARE Production Database Schema - PostgreSQL
-- Quantum-Safe Federated Learning with Comprehensive Security
-- ================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- ================================================================
-- ENUMS AND TYPES
-- ================================================================

-- Device types for federated learning
CREATE TYPE device_type_enum AS ENUM (
    'EDGE_NODE',
    'IOT_DEVICE', 
    'MOBILE_DEVICE',
    'SERVER_NODE',
    'GATEWAY_NODE'
);

-- Device status tracking
CREATE TYPE device_status_enum AS ENUM (
    'PENDING',
    'ACTIVE',
    'SUSPENDED',
    'DEREGISTERED',
    'COMPROMISED'
);

-- Key types for cryptographic operations
CREATE TYPE key_type_enum AS ENUM (
    'KYBER_PUBLIC',
    'KYBER_PRIVATE',
    'DILITHIUM_PUBLIC',
    'DILITHIUM_PRIVATE',
    'SESSION_KEY',
    'DERIVED_KEY',
    'BACKUP_KEY'
);

-- Key status lifecycle
CREATE TYPE key_status_enum AS ENUM (
    'ACTIVE',
    'EXPIRED',
    'REVOKED',
    'COMPROMISED',
    'PENDING_ROTATION'
);

-- Session status
CREATE TYPE session_status_enum AS ENUM (
    'PENDING',
    'ACTIVE',
    'EXPIRED',
    'TERMINATED',
    'COMPROMISED'
);

-- Federated learning round status
CREATE TYPE fl_round_status_enum AS ENUM (
    'PREPARING',
    'TRAINING',
    'AGGREGATING',
    'COMPLETED',
    'FAILED'
);

-- Audit event types
CREATE TYPE audit_event_enum AS ENUM (
    'DEVICE_REGISTRATION',
    'DEVICE_AUTHENTICATION',
    'KEY_EXCHANGE',
    'KEY_ROTATION',
    'MODEL_SUBMISSION',
    'MODEL_AGGREGATION',
    'SECURITY_VIOLATION',
    'SYSTEM_ERROR'
);

-- ================================================================
-- CORE TABLES
-- ================================================================

-- Organizations/Tenants for multi-tenancy
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    domain VARCHAR(255) NOT NULL UNIQUE,
    encryption_policy JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Device registry with comprehensive metadata
CREATE TABLE devices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    device_id VARCHAR(255) NOT NULL UNIQUE,
    device_name VARCHAR(255) NOT NULL,
    device_type device_type_enum NOT NULL,
    status device_status_enum DEFAULT 'PENDING',
    
    -- Hardware and software metadata
    hardware_info JSONB DEFAULT '{}',
    software_version VARCHAR(100),
    os_info JSONB DEFAULT '{}',
    capabilities JSONB DEFAULT '{}',
    
    -- Network and security
    ip_address INET,
    mac_address MACADDR,
    last_seen TIMESTAMP WITH TIME ZONE,
    security_level INTEGER DEFAULT 1 CHECK (security_level >= 1 AND security_level <= 5),
    
    -- Trust and reputation
    trust_score DECIMAL(3,2) DEFAULT 1.00 CHECK (trust_score >= 0 AND trust_score <= 1),
    reputation_score INTEGER DEFAULT 100 CHECK (reputation_score >= 0 AND reputation_score <= 100),
    failed_authentications INTEGER DEFAULT 0,
    
    -- Timestamps
    registered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_authenticated TIMESTAMP WITH TIME ZONE,
    last_key_rotation TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Quantum-safe cryptographic keys
CREATE TABLE cryptographic_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_id UUID REFERENCES devices(id) ON DELETE CASCADE,
    
    -- Key identification
    key_id VARCHAR(255) NOT NULL UNIQUE,
    key_type key_type_enum NOT NULL,
    algorithm VARCHAR(100) NOT NULL, -- e.g., 'Kyber1024', 'Dilithium2'
    
    -- Key material (encrypted at rest)
    public_key BYTEA,
    private_key_encrypted BYTEA, -- Encrypted with master key
    key_size INTEGER NOT NULL,
    
    -- Key lifecycle
    status key_status_enum DEFAULT 'ACTIVE',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    rotated_at TIMESTAMP WITH TIME ZONE,
    revoked_at TIMESTAMP WITH TIME ZONE,
    
    -- Security metadata
    generation_entropy BYTEA, -- Random seed used for key generation
    derivation_info JSONB DEFAULT '{}',
    usage_count INTEGER DEFAULT 0,
    max_usage_count INTEGER,
    
    -- Timestamps
    first_used TIMESTAMP WITH TIME ZONE,
    last_used TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT valid_expiry CHECK (expires_at > created_at)
);

-- Key exchange sessions with temporal mapping
CREATE TABLE key_exchange_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL UNIQUE,
    device_id UUID REFERENCES devices(id) ON DELETE CASCADE,
    
    -- Session state
    status session_status_enum DEFAULT 'PENDING',
    algorithm VARCHAR(100) NOT NULL,
    
    -- Temporal mapping for quantum resistance
    initiation_timestamp BIGINT NOT NULL, -- Unix timestamp
    completion_timestamp BIGINT,
    expiry_timestamp BIGINT NOT NULL,
    time_window INTEGER DEFAULT 300, -- seconds
    
    -- Cryptographic data
    server_public_key_id UUID REFERENCES cryptographic_keys(id),
    client_public_key BYTEA NOT NULL,
    shared_secret_hash BYTEA, -- Hash of shared secret (never store plaintext)
    derived_key_hash BYTEA,   -- Hash of derived key
    
    -- Session security
    nonce BYTEA NOT NULL,
    salt BYTEA NOT NULL,
    key_derivation_rounds INTEGER DEFAULT 1000000,
    
    -- Metadata
    ip_address INET,
    user_agent TEXT,
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Federated learning rounds and model management
CREATE TABLE fl_rounds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    
    round_number INTEGER NOT NULL,
    status fl_round_status_enum DEFAULT 'PREPARING',
    
    -- Model information
    model_architecture JSONB NOT NULL,
    global_model_hash BYTEA,
    target_participants INTEGER,
    min_participants INTEGER DEFAULT 1,
    
    -- Timing
    started_at TIMESTAMP WITH TIME ZONE,
    deadline TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Aggregation settings
    aggregation_algorithm VARCHAR(100) DEFAULT 'FedAvg',
    aggregation_weights JSONB DEFAULT '{}',
    
    -- Security
    secure_aggregation BOOLEAN DEFAULT true,
    privacy_budget DECIMAL(10,6),
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(organization_id, round_number)
);

-- Device participation in federated learning
CREATE TABLE fl_participation (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    round_id UUID REFERENCES fl_rounds(id) ON DELETE CASCADE,
    device_id UUID REFERENCES devices(id) ON DELETE CASCADE,
    session_id UUID REFERENCES key_exchange_sessions(id),
    
    -- Participation metadata
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    model_submitted_at TIMESTAMP WITH TIME ZONE,
    contribution_weight DECIMAL(5,4) DEFAULT 1.0000,
    
    -- Model update information (encrypted)
    encrypted_model_update BYTEA,
    model_update_hash BYTEA,
    gradient_norm DECIMAL(15,10),
    
    -- Quality metrics
    local_loss DECIMAL(15,10),
    local_accuracy DECIMAL(5,4),
    training_samples INTEGER,
    training_time_seconds INTEGER,
    
    -- Security validation
    signature BYTEA, -- Quantum-safe signature
    verified BOOLEAN DEFAULT false,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    UNIQUE(round_id, device_id)
);

-- Comprehensive audit log
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    device_id UUID REFERENCES devices(id) ON DELETE SET NULL,
    session_id UUID REFERENCES key_exchange_sessions(id) ON DELETE SET NULL,
    
    -- Event information
    event_type audit_event_enum NOT NULL,
    event_subtype VARCHAR(100),
    severity VARCHAR(20) DEFAULT 'INFO', -- DEBUG, INFO, WARN, ERROR, CRITICAL
    
    -- Event details
    event_data JSONB NOT NULL DEFAULT '{}',
    user_agent TEXT,
    ip_address INET,
    
    -- Security context
    threat_level INTEGER DEFAULT 0 CHECK (threat_level >= 0 AND threat_level <= 10),
    risk_score DECIMAL(3,2) DEFAULT 0.00,
    
    -- Tracing
    trace_id UUID,
    parent_event_id UUID REFERENCES audit_logs(id),
    
    -- Timestamps
    event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- System configuration and feature flags
CREATE TABLE system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    
    config_key VARCHAR(255) NOT NULL,
    config_value JSONB NOT NULL,
    config_type VARCHAR(50) DEFAULT 'string',
    
    -- Security
    encrypted BOOLEAN DEFAULT false,
    requires_restart BOOLEAN DEFAULT false,
    
    -- Metadata
    description TEXT,
    default_value JSONB,
    validation_schema JSONB,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(organization_id, config_key)
);

-- ================================================================
-- INDEXES FOR PERFORMANCE
-- ================================================================

-- Device indexes
CREATE INDEX idx_devices_org_type ON devices(organization_id, device_type);
CREATE INDEX idx_devices_status ON devices(status);
CREATE INDEX idx_devices_last_seen ON devices(last_seen);
CREATE INDEX idx_devices_trust_score ON devices(trust_score);
CREATE INDEX idx_devices_device_id ON devices(device_id);

-- Key indexes
CREATE INDEX idx_keys_device_type ON cryptographic_keys(device_id, key_type);
CREATE INDEX idx_keys_status_expires ON cryptographic_keys(status, expires_at);
CREATE INDEX idx_keys_algorithm ON cryptographic_keys(algorithm);
CREATE INDEX idx_keys_created_at ON cryptographic_keys(created_at);

-- Session indexes
CREATE INDEX idx_sessions_device_status ON key_exchange_sessions(device_id, status);
CREATE INDEX idx_sessions_expiry ON key_exchange_sessions(expiry_timestamp);
CREATE INDEX idx_sessions_timestamp ON key_exchange_sessions(initiation_timestamp);
CREATE INDEX idx_sessions_session_id ON key_exchange_sessions(session_id);

-- FL Round indexes
CREATE INDEX idx_fl_rounds_org_status ON fl_rounds(organization_id, status);
CREATE INDEX idx_fl_rounds_started ON fl_rounds(started_at);
CREATE INDEX idx_fl_participation_round_device ON fl_participation(round_id, device_id);

-- Audit log indexes
CREATE INDEX idx_audit_org_type_time ON audit_logs(organization_id, event_type, event_timestamp);
CREATE INDEX idx_audit_device_time ON audit_logs(device_id, event_timestamp);
CREATE INDEX idx_audit_severity_time ON audit_logs(severity, event_timestamp);
CREATE INDEX idx_audit_timestamp ON audit_logs(event_timestamp);

-- GIN indexes for JSONB columns
CREATE INDEX idx_devices_metadata_gin ON devices USING GIN(metadata);
CREATE INDEX idx_keys_derivation_gin ON cryptographic_keys USING GIN(derivation_info);
CREATE INDEX idx_audit_data_gin ON audit_logs USING GIN(event_data);

-- ================================================================
-- FUNCTIONS AND TRIGGERS
-- ================================================================

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_devices_updated_at BEFORE UPDATE ON devices FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_sessions_updated_at BEFORE UPDATE ON key_exchange_sessions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_fl_rounds_updated_at BEFORE UPDATE ON fl_rounds FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_system_config_updated_at BEFORE UPDATE ON system_config FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Key rotation trigger
CREATE OR REPLACE FUNCTION check_key_expiry()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.expires_at <= CURRENT_TIMESTAMP AND OLD.status = 'ACTIVE' THEN
        NEW.status = 'EXPIRED';
        
        -- Log key expiry
        INSERT INTO audit_logs (event_type, event_data, device_id)
        VALUES ('KEY_ROTATION', 
                jsonb_build_object('key_id', NEW.key_id, 'expired_at', NEW.expires_at),
                NEW.device_id);
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER key_expiry_check BEFORE UPDATE ON cryptographic_keys FOR EACH ROW EXECUTE FUNCTION check_key_expiry();

-- Session cleanup function
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    expired_count INTEGER;
BEGIN
    UPDATE key_exchange_sessions 
    SET status = 'EXPIRED' 
    WHERE status = 'ACTIVE' 
    AND expiry_timestamp < EXTRACT(EPOCH FROM CURRENT_TIMESTAMP);
    
    GET DIAGNOSTICS expired_count = ROW_COUNT;
    
    -- Log cleanup
    INSERT INTO audit_logs (event_type, event_data)
    VALUES ('SYSTEM_ERROR', 
            jsonb_build_object('action', 'session_cleanup', 'expired_sessions', expired_count));
    
    RETURN expired_count;
END;
$$ language 'plpgsql';

-- ================================================================
-- SECURITY POLICIES (ROW LEVEL SECURITY)
-- ================================================================

-- Enable RLS on sensitive tables
ALTER TABLE devices ENABLE ROW LEVEL SECURITY;
ALTER TABLE cryptographic_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE key_exchange_sessions ENABLE ROW LEVEL SECURITY;

-- Device access policy (organization-based)
CREATE POLICY device_org_policy ON devices
    FOR ALL
    TO qflare_api_role
    USING (organization_id = current_setting('app.current_org_id')::UUID);

-- Key access policy (device ownership)
CREATE POLICY key_device_policy ON cryptographic_keys
    FOR ALL
    TO qflare_api_role
    USING (device_id IN (
        SELECT id FROM devices 
        WHERE organization_id = current_setting('app.current_org_id')::UUID
    ));

-- ================================================================
-- VIEWS FOR COMMON QUERIES
-- ================================================================

-- Active device summary
CREATE VIEW active_devices_summary AS
SELECT 
    d.organization_id,
    d.device_type,
    COUNT(*) as device_count,
    AVG(d.trust_score) as avg_trust_score,
    MAX(d.last_seen) as last_activity
FROM devices d
WHERE d.status = 'ACTIVE'
GROUP BY d.organization_id, d.device_type;

-- Current sessions view
CREATE VIEW current_sessions AS
SELECT 
    s.session_id,
    s.device_id,
    d.device_name,
    s.status,
    s.algorithm,
    s.initiation_timestamp,
    s.expiry_timestamp,
    (s.expiry_timestamp - EXTRACT(EPOCH FROM CURRENT_TIMESTAMP)) as time_remaining_seconds
FROM key_exchange_sessions s
JOIN devices d ON s.device_id = d.id
WHERE s.status = 'ACTIVE' 
AND s.expiry_timestamp > EXTRACT(EPOCH FROM CURRENT_TIMESTAMP);

-- Security dashboard view
CREATE VIEW security_dashboard AS
SELECT 
    COUNT(CASE WHEN d.status = 'ACTIVE' THEN 1 END) as active_devices,
    COUNT(CASE WHEN d.status = 'SUSPENDED' THEN 1 END) as suspended_devices,
    COUNT(CASE WHEN s.status = 'ACTIVE' THEN 1 END) as active_sessions,
    COUNT(CASE WHEN k.status = 'EXPIRED' THEN 1 END) as expired_keys,
    COUNT(CASE WHEN a.severity = 'CRITICAL' AND a.event_timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour' THEN 1 END) as recent_critical_events
FROM devices d
CROSS JOIN key_exchange_sessions s
CROSS JOIN cryptographic_keys k
CROSS JOIN audit_logs a;

-- ================================================================
-- SAMPLE DATA INSERTION
-- ================================================================

-- Insert default organization
INSERT INTO organizations (name, domain, encryption_policy) VALUES 
('QFLARE Research Lab', 'qflare.research', 
 '{"algorithm": "Kyber1024", "key_rotation_hours": 24, "max_session_duration": 3600}');

-- ================================================================
-- MAINTENANCE JOBS
-- ================================================================

-- Create scheduled cleanup job (requires pg_cron extension)
-- SELECT cron.schedule('cleanup-sessions', '*/5 * * * *', 'SELECT cleanup_expired_sessions();');

COMMENT ON DATABASE qflare IS 'QFLARE Quantum-Safe Federated Learning Database';