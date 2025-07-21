# audit_logger.py
"""
Simple audit logging logic for demo purposes.
"""
audit_log = []

def log_audit_event(event):
    audit_log.append(event)
    print(f"[AUDIT] {event}")