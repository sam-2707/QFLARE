#!/usr/bin/env python3
"""
QFLARE Key Management Architecture Visualizer
Generates visual diagrams of key storage and management flow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

class QFLAREArchitectureVisualizer:
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
    def create_key_management_diagram(self):
        """Create comprehensive key management architecture diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 15)
        ax.axis('off')
        
        # Title
        ax.text(10, 14, 'QFLARE Key Management Architecture', 
                fontsize=20, fontweight='bold', ha='center')
        ax.text(10, 13.3, 'Post-Quantum Cryptographic Security Framework', 
                fontsize=14, ha='center', style='italic')
        
        # Layer 1: Hardware Security Module (HSM)
        hsm_box = FancyBboxPatch((1, 11), 6, 2, 
                                boxstyle="round,pad=0.1", 
                                facecolor=self.colors['primary'], 
                                edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(hsm_box)
        ax.text(4, 12.2, 'Hardware Security Module', fontsize=12, fontweight='bold', 
                ha='center', va='center', color='white')
        ax.text(4, 11.7, 'FIPS 140-2 Level 3', fontsize=10, ha='center', va='center', color='white')
        ax.text(4, 11.3, '🔐 Tamper-resistant key storage', fontsize=10, ha='center', va='center', color='white')
        
        # Layer 2: Intel SGX Enclaves
        sgx_boxes = []
        for i in range(3):
            sgx_box = FancyBboxPatch((9 + i*3.5, 11), 3, 2, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=self.colors['success'], 
                                    edgecolor='black', linewidth=2, alpha=0.8)
            ax.add_patch(sgx_box)
            ax.text(10.5 + i*3.5, 12.2, f'SGX Enclave {i+1}', fontsize=10, fontweight='bold', 
                    ha='center', va='center', color='white')
            ax.text(10.5 + i*3.5, 11.7, 'Sealed Keys', fontsize=9, ha='center', va='center', color='white')
            ax.text(10.5 + i*3.5, 11.3, f'Node {i+1}', fontsize=9, ha='center', va='center', color='white')
        
        # Layer 3: Key Generation Engine
        keygen_box = FancyBboxPatch((2, 8), 5, 2, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=self.colors['secondary'], 
                                   edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(keygen_box)
        ax.text(4.5, 9.2, 'Post-Quantum Key Generator', fontsize=12, fontweight='bold', 
                ha='center', va='center', color='white')
        ax.text(4.5, 8.7, 'Kyber-1024 + Dilithium', fontsize=10, ha='center', va='center', color='white')
        ax.text(4.5, 8.3, '⚛️ Quantum-resistant algorithms', fontsize=10, ha='center', va='center', color='white')
        
        # Layer 4: Distributed Storage
        storage_boxes = []
        storage_labels = ['Primary Storage', 'Backup Storage', 'Archive Storage']
        for i, label in enumerate(storage_labels):
            storage_box = FancyBboxPatch((8.5 + i*3.5, 8), 3, 2, 
                                        boxstyle="round,pad=0.1", 
                                        facecolor=self.colors['info'], 
                                        edgecolor='black', linewidth=2, alpha=0.8)
            ax.add_patch(storage_box)
            ax.text(10 + i*3.5, 9.2, label, fontsize=10, fontweight='bold', 
                    ha='center', va='center', color='white')
            ax.text(10 + i*3.5, 8.7, 'AES-256', fontsize=9, ha='center', va='center', color='white')
            ax.text(10 + i*3.5, 8.3, f'Region {i+1}', fontsize=9, ha='center', va='center', color='white')
        
        # Layer 5: Federated Learning Nodes
        fl_nodes = []
        for i in range(4):
            node_box = FancyBboxPatch((1 + i*4.5, 5), 3.5, 1.5, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=self.colors['warning'], 
                                     edgecolor='black', linewidth=2, alpha=0.8)
            ax.add_patch(node_box)
            ax.text(2.75 + i*4.5, 5.9, f'FL Node {i+1}', fontsize=10, fontweight='bold', 
                    ha='center', va='center', color='white')
            ax.text(2.75 + i*4.5, 5.5, 'Secure Training', fontsize=9, ha='center', va='center', color='white')
            ax.text(2.75 + i*4.5, 5.1, f'🔒 Encrypted comms', fontsize=9, ha='center', va='center', color='white')
        
        # Layer 6: Security Monitoring
        monitor_box = FancyBboxPatch((6, 2), 8, 1.5, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=self.colors['danger'], 
                                    edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(monitor_box)
        ax.text(10, 2.9, 'Real-time Security Monitoring', fontsize=12, fontweight='bold', 
                ha='center', va='center', color='white')
        ax.text(10, 2.5, '🛡️ Threat Detection | 📊 Key Rotation | ⚠️ Anomaly Detection', 
                fontsize=10, ha='center', va='center', color='white')
        ax.text(10, 2.1, 'Blockchain Audit Trail | Zero-Trust Architecture', 
                fontsize=10, ha='center', va='center', color='white')
        
        # Add connection arrows
        # HSM to SGX Enclaves
        arrow1 = ConnectionPatch((7, 12), (9, 12), "data", "data",
                                arrowstyle="->", shrinkA=5, shrinkB=5, 
                                mutation_scale=20, fc=self.colors['dark'])
        ax.add_patch(arrow1)
        
        # Key Generator to HSM
        arrow2 = ConnectionPatch((4.5, 10), (4, 11), "data", "data",
                                arrowstyle="->", shrinkA=5, shrinkB=5, 
                                mutation_scale=20, fc=self.colors['dark'])
        ax.add_patch(arrow2)
        
        # Key Generator to Storage
        arrow3 = ConnectionPatch((6.5, 9), (8.5, 9), "data", "data",
                                arrowstyle="->", shrinkA=5, shrinkB=5, 
                                mutation_scale=20, fc=self.colors['dark'])
        ax.add_patch(arrow3)
        
        # Storage to FL Nodes
        for i in range(4):
            arrow = ConnectionPatch((10, 8), (2.75 + i*4.5, 6.5), "data", "data",
                                   arrowstyle="->", shrinkA=5, shrinkB=5, 
                                   mutation_scale=20, fc=self.colors['dark'])
            ax.add_patch(arrow)
        
        # FL Nodes to Monitoring
        for i in range(4):
            arrow = ConnectionPatch((2.75 + i*4.5, 5), (8 + i*1, 3.5), "data", "data",
                                   arrowstyle="->", shrinkA=5, shrinkB=5, 
                                   mutation_scale=20, fc=self.colors['dark'])
            ax.add_patch(arrow)
        
        # Add security indicators
        ax.text(1, 0.5, '🔐 Security Features:', fontsize=12, fontweight='bold')
        features = [
            '• Post-Quantum Cryptography (Kyber-1024, Dilithium)',
            '• Hardware-backed Security (HSM + SGX)',
            '• Multi-layer Key Protection',
            '• Geographic Distribution & Redundancy',
            '• Real-time Threat Monitoring',
            '• Automated Key Rotation (24h cycle)',
            '• Zero-Trust Architecture'
        ]
        
        for i, feature in enumerate(features):
            ax.text(1.5, 0.1 - i*0.3, feature, fontsize=10)
        
        plt.tight_layout()
        plt.savefig('qflare_key_architecture.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_security_comparison_chart(self):
        """Create security comparison chart vs competitors"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Security Features Comparison
        platforms = ['TensorFlow\nFederated', 'PySyft', 'FedML', 'QFLARE']
        features = ['Post-Quantum\nCrypto', 'Hardware\nEnclaves', 'Differential\nPrivacy', 
                   'Byzantine\nTolerance', 'Real-time\nMonitoring', 'Key\nRotation']
        
        # Scores (0-5 scale)
        scores = np.array([
            [0, 2, 3, 1, 2, 1],  # TensorFlow Federated
            [0, 1, 4, 2, 1, 0],  # PySyft
            [0, 0, 2, 3, 2, 1],  # FedML
            [5, 5, 5, 4, 5, 5]   # QFLARE
        ])
        
        x = np.arange(len(features))
        width = 0.2
        
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4']
        
        for i, (platform, color) in enumerate(zip(platforms, colors)):
            offset = (i - 1.5) * width
            bars = ax1.bar(x + offset, scores[i], width, label=platform, 
                          color=color, alpha=0.8, edgecolor='black')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{height}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_xlabel('Security Features', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Security Level (0-5)', fontsize=12, fontweight='bold')
        ax1.set_title('Security Features Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(features, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 6)
        
        # Overall Security Score Radar Chart
        categories = ['Quantum\nResistance', 'Privacy\nProtection', 'Attack\nResilience', 
                     'Key\nManagement', 'Monitoring\nCapability', 'Performance']
        
        # Scores out of 10
        qflare_scores = [10, 9, 9, 10, 9, 8]
        competitor_avg = [2, 6, 5, 3, 4, 7]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        qflare_scores += qflare_scores[:1]
        competitor_avg += competitor_avg[:1]
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles, qflare_scores, 'o-', linewidth=3, label='QFLARE', color='#1f77b4')
        ax2.fill(angles, qflare_scores, alpha=0.25, color='#1f77b4')
        
        ax2.plot(angles, competitor_avg, 'o-', linewidth=3, label='Competitors Avg', color='#ff7f0e')
        ax2.fill(angles, competitor_avg, alpha=0.25, color='#ff7f0e')
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories, fontsize=10)
        ax2.set_ylim(0, 10)
        ax2.set_yticks([2, 4, 6, 8, 10])
        ax2.set_title('Overall Security Assessment', fontsize=14, fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('qflare_security_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_threat_timeline(self):
        """Create quantum threat timeline visualization"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Timeline data
        years = [2024, 2026, 2028, 2030, 2032, 2035, 2040]
        quantum_threat = [10, 25, 45, 70, 85, 95, 100]
        qflare_protection = [95, 95, 95, 95, 95, 95, 95]
        traditional_security = [90, 75, 50, 25, 10, 5, 0]
        
        # Plot lines
        ax.plot(years, quantum_threat, 'r-', linewidth=4, marker='o', markersize=8, 
                label='Quantum Threat Level', markerfacecolor='darkred')
        ax.plot(years, qflare_protection, 'b-', linewidth=4, marker='s', markersize=8, 
                label='QFLARE Protection Level', markerfacecolor='darkblue')
        ax.plot(years, traditional_security, 'orange', linewidth=4, marker='^', markersize=8, 
                label='Traditional Crypto Security', markerfacecolor='darkorange')
        
        # Fill areas
        ax.fill_between(years, quantum_threat, alpha=0.3, color='red', label='Quantum Risk Zone')
        ax.fill_between(years, qflare_protection, alpha=0.3, color='blue', label='QFLARE Safe Zone')
        
        # Add annotations
        ax.annotate('NISQ Era Begins', xy=(2026, 25), xytext=(2025, 40),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=12, fontweight='bold', color='red')
        
        ax.annotate('Cryptographically\nRelevant QC', xy=(2032, 85), xytext=(2030, 100),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=12, fontweight='bold', color='red', ha='center')
        
        ax.annotate('QFLARE: Quantum-Ready\nSince Day 1', xy=(2024, 95), xytext=(2026, 85),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                   fontsize=12, fontweight='bold', color='blue', ha='center')
        
        # Styling
        ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        ax.set_ylabel('Security Level (%)', fontsize=14, fontweight='bold')
        ax.set_title('Quantum Computing Threat Timeline vs QFLARE Protection', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.set_ylim(0, 110)
        
        # Add critical dates
        critical_dates = {
            2024: "QFLARE Launch",
            2030: "NIST Migration Deadline", 
            2035: "Quantum Advantage"
        }
        
        for year, event in critical_dates.items():
            ax.axvline(x=year, color='gray', linestyle='--', alpha=0.7)
            ax.text(year, 5, event, rotation=90, fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('quantum_threat_timeline.png', dpi=300, bbox_inches='tight')
        plt.show()

def generate_all_visuals():
    """Generate all presentation visuals"""
    visualizer = QFLAREArchitectureVisualizer()
    
    print("🎨 Generating QFLARE presentation visuals...")
    
    print("📊 Creating key management architecture diagram...")
    visualizer.create_key_management_diagram()
    
    print("📈 Creating security comparison chart...")
    visualizer.create_security_comparison_chart()
    
    print("⏰ Creating quantum threat timeline...")
    visualizer.create_threat_timeline()
    
    print("✅ All visuals generated successfully!")
    print("Files created:")
    print("  - qflare_key_architecture.png")
    print("  - qflare_security_comparison.png") 
    print("  - quantum_threat_timeline.png")

if __name__ == "__main__":
    generate_all_visuals()