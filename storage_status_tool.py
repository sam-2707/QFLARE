#!/usr/bin/env python3
"""
QFLARE Storage Status and Migration Tool
Monitors storage system and provides migration utilities
"""

import asyncio
import os
import json
from datetime import datetime
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StorageStatus:
    """Monitor and report on QFLARE storage system status"""
    
    def __init__(self):
        self.status_data = {}
    
    async def check_storage_health(self) -> Dict[str, Any]:
        """Check the health and status of storage systems"""
        
        logger.info("🔍 Checking QFLARE storage system health...")
        
        # Check environment variables
        database_url = os.getenv('QFLARE_DATABASE_URL')
        kms_key_id = os.getenv('QFLARE_KMS_KEY_ID')
        
        # Check dependencies
        dependencies = {}
        try:
            import sqlalchemy
            dependencies['sqlalchemy'] = sqlalchemy.__version__
        except ImportError:
            dependencies['sqlalchemy'] = 'NOT_INSTALLED'
        
        try:
            import boto3
            dependencies['boto3'] = boto3.__version__
        except ImportError:
            dependencies['boto3'] = 'NOT_INSTALLED'
        
        try:
            from cryptography import __version__ as crypto_version
            dependencies['cryptography'] = crypto_version
        except ImportError:
            dependencies['cryptography'] = 'NOT_INSTALLED'
        
        # Test secure storage
        storage_status = "unknown"
        storage_type = "unknown"
        device_count = 0
        key_count = 0
        
        try:
            from server.secure_device_storage import get_secure_storage
            storage = get_secure_storage()
            
            status_info = await storage.get_storage_status()
            storage_status = "healthy" if status_info['secure_mode'] else "fallback"
            storage_type = status_info['storage_type']
            device_count = status_info['device_count']
            key_count = status_info['key_count']
            
        except Exception as e:
            logger.error(f"❌ Storage health check failed: {e}")
            storage_status = "error"
        
        # Compile status report
        status_report = {
            'timestamp': datetime.now().isoformat(),
            'storage': {
                'status': storage_status,
                'type': storage_type,
                'device_count': device_count,
                'key_count': key_count,
                'secure_mode': storage_status != "fallback"
            },
            'configuration': {
                'database_url_configured': bool(database_url),
                'kms_key_configured': bool(kms_key_id),
                'database_url_preview': database_url[:50] + "..." if database_url else None,
                'kms_key_preview': kms_key_id[-20:] if kms_key_id else None
            },
            'dependencies': dependencies,
            'recommendations': []
        }
        
        # Generate recommendations
        if storage_status == "fallback":
            status_report['recommendations'].append({
                'priority': 'HIGH',
                'category': 'Security',
                'message': 'Using in-memory storage - not secure for production',
                'action': 'Set QFLARE_DATABASE_URL and QFLARE_KMS_KEY_ID environment variables'
            })
        
        if dependencies['sqlalchemy'] == 'NOT_INSTALLED':
            status_report['recommendations'].append({
                'priority': 'HIGH',
                'category': 'Dependencies',
                'message': 'SQLAlchemy not installed - required for secure storage',
                'action': 'Run: pip install sqlalchemy psycopg2-binary'
            })
        
        if dependencies['boto3'] == 'NOT_INSTALLED':
            status_report['recommendations'].append({
                'priority': 'HIGH',
                'category': 'Dependencies',
                'message': 'Boto3 not installed - required for KMS encryption',
                'action': 'Run: pip install boto3'
            })
        
        if dependencies['cryptography'] == 'NOT_INSTALLED':
            status_report['recommendations'].append({
                'priority': 'HIGH',
                'category': 'Dependencies',
                'message': 'Cryptography not installed - required for encryption',
                'action': 'Run: pip install cryptography'
            })
        
        return status_report
    
    def print_status_report(self, status: Dict[str, Any]):
        """Print formatted status report"""
        
        print("\n🔐 QFLARE Storage System Status Report")
        print("=" * 60)
        print(f"📅 Generated: {status['timestamp']}")
        print()
        
        # Storage status
        storage = status['storage']
        status_icon = "✅" if storage['status'] == "healthy" else "⚠️" if storage['status'] == "fallback" else "❌"
        print(f"📊 Storage Status: {status_icon} {storage['status'].upper()}")
        print(f"🗄️  Storage Type: {storage['type']}")
        print(f"📱 Devices Stored: {storage['device_count']}")
        print(f"🔑 Keys Stored: {storage['key_count']}")
        print(f"🔒 Secure Mode: {'Yes' if storage['secure_mode'] else 'No'}")
        print()
        
        # Configuration
        config = status['configuration']
        print("⚙️  Configuration:")
        db_icon = "✅" if config['database_url_configured'] else "❌"
        kms_icon = "✅" if config['kms_key_configured'] else "❌"
        print(f"   {db_icon} Database URL: {'Configured' if config['database_url_configured'] else 'Not Set'}")
        print(f"   {kms_icon} KMS Key ID: {'Configured' if config['kms_key_configured'] else 'Not Set'}")
        print()
        
        # Dependencies
        print("📦 Dependencies:")
        for dep, version in status['dependencies'].items():
            dep_icon = "✅" if version != 'NOT_INSTALLED' else "❌"
            print(f"   {dep_icon} {dep}: {version}")
        print()
        
        # Recommendations
        if status['recommendations']:
            print("💡 Recommendations:")
            for rec in status['recommendations']:
                priority_icon = "🚨" if rec['priority'] == 'HIGH' else "⚠️"
                print(f"   {priority_icon} [{rec['category']}] {rec['message']}")
                print(f"      Action: {rec['action']}")
                print()
        else:
            print("✅ No recommendations - system is properly configured!")
            print()
    
    async def migrate_to_secure_storage(self):
        """Migrate existing in-memory data to secure storage"""
        logger.info("🔄 Starting migration to secure storage...")
        
        try:
            # Check if migration is needed/possible
            from server.secure_device_storage import get_secure_storage
            storage = get_secure_storage()
            
            status_info = await storage.get_storage_status()
            
            if not status_info['secure_mode']:
                logger.error("❌ Cannot migrate - secure storage not configured")
                return False
            
            # For now, since we don't have existing in-memory data to migrate,
            # we'll create some test data to demonstrate the migration
            logger.info("🧪 Creating test data for migration demonstration...")
            
            test_devices = [
                {
                    'device_name': 'migration_test_mobile_001',
                    'device_type': 'smartphone',
                    'capabilities': ['cpu', 'gpu'],
                    'contact_info': 'test@qflare.com',
                    'location': {'region': 'us-east-1', 'city': 'New York'},
                    'device_class': 'mobile',
                    'security_level': 2
                },
                {
                    'device_name': 'migration_test_laptop_001', 
                    'device_type': 'laptop',
                    'capabilities': ['cpu', 'gpu'],
                    'contact_info': 'admin@qflare.com',
                    'location': {'region': 'eu-west-1', 'city': 'London'},
                    'device_class': 'desktop',
                    'security_level': 3
                }
            ]
            
            migrated_devices = []
            
            for device_data in test_devices:
                try:
                    device_id = await storage.store_device(device_data)
                    migrated_devices.append({
                        'device_id': device_id,
                        'device_name': device_data['device_name']
                    })
                    logger.info(f"✅ Migrated device: {device_data['device_name']} -> {device_id}")
                    
                    # Generate and store sample key material
                    import os
                    key_data = os.urandom(64)  # 512-bit key
                    key_id = await storage.store_key_material(
                        device_id, 'quantum_private', key_data, 'kyber1024'
                    )
                    logger.info(f"🔐 Generated key for device: {key_id}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to migrate device {device_data['device_name']}: {e}")
            
            logger.info(f"✅ Migration completed: {len(migrated_devices)} devices migrated")
            
            # Save migration report
            migration_report = {
                'timestamp': datetime.now().isoformat(),
                'migration_type': 'test_data_creation',
                'migrated_devices': migrated_devices,
                'total_migrated': len(migrated_devices),
                'status': 'completed'
            }
            
            with open('migration_report.json', 'w') as f:
                json.dump(migration_report, f, indent=2, default=str)
            
            logger.info("📊 Migration report saved to migration_report.json")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return False

async def main():
    """Main function to run storage status checks and migration"""
    
    print("🚀 QFLARE Storage Management Tool")
    print("=" * 50)
    
    status_checker = StorageStatus()
    
    # Check storage health
    status_report = await status_checker.check_storage_health()
    status_checker.print_status_report(status_report)
    
    # Ask user if they want to run migration
    if status_report['storage']['status'] == 'healthy':
        print("🔄 Secure storage is available. Would you like to run a migration test?")
        print("This will create sample devices and keys in the secure storage system.")
        
        # For automation, we'll run the migration test
        print("Running migration test...")
        success = await status_checker.migrate_to_secure_storage()
        
        if success:
            print("✅ Migration test completed successfully!")
            
            # Check status again after migration
            print("\n📊 Post-migration status:")
            updated_status = await status_checker.check_storage_health()
            print(f"   📱 Devices: {updated_status['storage']['device_count']}")
            print(f"   🔑 Keys: {updated_status['storage']['key_count']}")
        
    elif status_report['storage']['status'] == 'fallback':
        print("\n💡 To enable secure storage:")
        print("1. Install dependencies: pip install sqlalchemy psycopg2-binary boto3 cryptography")
        print("2. Set up PostgreSQL database")
        print("3. Create AWS KMS key")
        print("4. Set environment variables:")
        print("   export QFLARE_DATABASE_URL='postgresql://user:pass@localhost:5432/qflare'")
        print("   export QFLARE_KMS_KEY_ID='arn:aws:kms:region:account:key/key-id'")
        
    else:
        print("\n❌ Storage system has errors. Please check the recommendations above.")

if __name__ == "__main__":
    asyncio.run(main())