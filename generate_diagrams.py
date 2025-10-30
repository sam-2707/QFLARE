"""
QFLARE System Architecture Diagram Generator
Creates comprehensive block diagrams and flowcharts for the QFLARE system
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arrow
import numpy as np
from pathlib import Path

class QFLAREDiagramGenerator:
    """Generate QFLARE system architecture diagrams"""
    
    def __init__(self, output_dir="docs/diagrams"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define consistent colors
        self.colors = {
            'client': '#E8F4FD',      # Light blue
            'edge': '#FFE8CC',        # Light orange  
            'server': '#E8F8E8',      # Light green
            'crypto': '#FFE8F8',      # Light pink
            'data': '#F0F0F0',        # Light gray
            'security': '#FFE8E8',    # Light red
            'network': '#E8E8FF',     # Light purple
            'border': '#333333'       # Dark gray
        }
    
    def create_overall_architecture(self):
        """Create the main QFLARE system architecture diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Title
        ax.text(0.5, 0.95, 'QFLARE: Quantum-Safe Federated Learning Architecture', 
                ha='center', va='center', fontsize=18, fontweight='bold')
        
        # Client Devices Layer
        client_y = 0.75
        clients = [
            ('Mobile\nDevices', 0.15, client_y),
            ('IoT\nSensors', 0.35, client_y),
            ('Edge\nCompute', 0.55, client_y),
            ('Desktop\nClients', 0.75, client_y)
        ]
        
        for name, x, y in clients:
            self._draw_component(ax, name, x, y, 0.12, 0.08, self.colors['client'])
        
        # Edge Aggregation Layer
        edge_y = 0.55
        edges = [
            ('Edge Node 1\n[Kyber+Dilithium]', 0.25, edge_y),
            ('Edge Node 2\n[Secure Agg]', 0.5, edge_y),
            ('Edge Node 3\n[DP Noise]', 0.75, edge_y)
        ]
        
        for name, x, y in edges:
            self._draw_component(ax, name, x, y, 0.15, 0.1, self.colors['edge'])
        
        # Central Server Layer
        server_components = [
            ('Global Model\nAggregator', 0.3, 0.3),
            ('PQC Key\nManagement', 0.5, 0.3),
            ('Privacy Budget\nTracker', 0.7, 0.3)
        ]
        
        for name, x, y in server_components:
            self._draw_component(ax, name, x, y, 0.12, 0.08, self.colors['server'])
        
        # Security & Monitoring Layer  
        security_y = 0.1
        security_components = [
            ('Threat\nMonitoring', 0.15, security_y),
            ('Audit\nLogging', 0.35, security_y),
            ('KMS/Vault', 0.55, security_y),
            ('Performance\nMetrics', 0.75, security_y)
        ]
        
        for name, x, y in security_components:
            self._draw_component(ax, name, x, y, 0.1, 0.06, self.colors['security'])
        
        # Draw connections
        # Clients to Edge Nodes (sample connections)
        self._draw_arrow(ax, clients[0][1], client_y-0.04, edges[0][1], edge_y+0.05, 'PQC\nHandshake')
        self._draw_arrow(ax, clients[1][1], client_y-0.04, edges[1][1], edge_y+0.05, 'PQC\nHandshake')
        
        # Edge to Server
        self._draw_arrow(ax, edges[1][1], edge_y-0.05, server_components[1][1], 0.38, 'Encrypted\nAggregation')
        
        # Server to Security  
        self._draw_arrow(ax, server_components[0][1], 0.26, security_components[1][1], security_y+0.03, 'Logs &\nMetrics')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color=self.colors['client'], label='Client Layer'),
            mpatches.Patch(color=self.colors['edge'], label='Edge Aggregation'),
            mpatches.Patch(color=self.colors['server'], label='Central Server'),
            mpatches.Patch(color=self.colors['security'], label='Security & Monitoring')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.88))
        
        # Add layer labels
        ax.text(0.02, client_y, 'CLIENT\nLAYER', va='center', ha='left', fontweight='bold', rotation=90)
        ax.text(0.02, edge_y, 'EDGE\nLAYER', va='center', ha='left', fontweight='bold', rotation=90)
        ax.text(0.02, 0.3, 'SERVER\nLAYER', va='center', ha='left', fontweight='bold', rotation=90)
        ax.text(0.02, security_y, 'SECURITY\nLAYER', va='center', ha='left', fontweight='bold', rotation=90)
        
        self._finalize_plot(ax, 'qflare_overall_architecture.png')
        return fig

    def create_federated_learning_flow(self):
        """Create federated learning process flow diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        ax.text(0.5, 0.95, 'QFLARE Federated Learning Process Flow', 
                ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Define process steps
        steps = [
            ('1. Global Model\nDistribution', 0.5, 0.85, self.colors['server']),
            ('2. Local Training\n(DP Noise)', 0.15, 0.7, self.colors['client']),
            ('3. Gradient\nComputation', 0.15, 0.55, self.colors['client']),
            ('4. PQC Encryption\n(Kyber KEM)', 0.15, 0.4, self.colors['crypto']),
            ('5. Secure Upload\nto Edge', 0.5, 0.4, self.colors['network']),
            ('6. Edge Aggregation\n(Secure MPC)', 0.85, 0.55, self.colors['edge']),
            ('7. Encrypted Upload\nto Server', 0.85, 0.7, self.colors['network']),
            ('8. Global Model\nUpdate', 0.5, 0.25, self.colors['server']),
            ('9. Convergence\nCheck', 0.5, 0.1, self.colors['data'])
        ]
        
        # Draw process steps
        for i, (name, x, y, color) in enumerate(steps):
            self._draw_component(ax, name, x, y, 0.12, 0.08, color)
            
            # Add step numbers in circles
            circle = Circle((x-0.08, y+0.05), 0.02, color='red', alpha=0.7)
            ax.add_patch(circle)
            ax.text(x-0.08, y+0.05, str(i+1), ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white')
        
        # Draw flow arrows
        flows = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 0)
        ]
        
        for start_idx, end_idx in flows:
            start = steps[start_idx]
            end = steps[end_idx]
            self._draw_flow_arrow(ax, start[1], start[2], end[1], end[2])
        
        # Add decision diamond for convergence
        self._draw_diamond(ax, 0.5, 0.1, 0.08, 0.05, self.colors['data'])
        ax.text(0.5, 0.1, 'Converged?', ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Add security annotations
        ax.text(0.02, 0.5, 'SECURITY FEATURES:\n• PQC Encryption\n• Differential Privacy\n• Secure Aggregation\n• Zero-Knowledge Proofs', 
                va='top', ha='left', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['security'], alpha=0.7))
        
        self._finalize_plot(ax, 'qflare_federated_learning_flow.png')
        return fig

    def create_pqc_handshake_sequence(self):
        """Create PQC handshake sequence diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        ax.text(0.5, 0.95, 'QFLARE Post-Quantum Cryptographic Handshake', 
                ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Participants
        participants = [
            ('Client Device', 0.2, 0.9),
            ('Edge Node', 0.5, 0.9),
            ('Central Server', 0.8, 0.9)
        ]
        
        for name, x, y in participants:
            self._draw_component(ax, name, x, y, 0.12, 0.06, self.colors['client'])
            # Draw vertical timeline
            ax.plot([x, x], [0.1, 0.85], 'k--', alpha=0.3, linewidth=2)
        
        # Handshake steps with timing
        steps = [
            ('1. Client Hello\n+ Kyber Public Key'),
            ('2. Server Hello\n+ Certificate Chain'),
            ('3. Kyber Encapsulation\n+ Shared Secret'),
            ('4. Dilithium Signature\nVerification'),
            ('5. Encrypted Channel\nEstablished'),
            ('6. Forward to Server\n(if needed)'),
            ('7. Session Key\nDistribution'),
            ('8. Secure Data Transfer\nReady')
        ]
        
        for step_info in steps:
            if len(step_info) == 6:
                name, x1, y1, x2, y2, timing = step_info
                self._draw_message_arrow(ax, x1, y1, x2, y2, name, timing)
        
        # Add crypto algorithm boxes
        ax.text(0.02, 0.4, 'CRYPTOGRAPHIC\nALGORITHMS:\n\n• CRYSTALS-Kyber\n  (Key Encapsulation)\n• CRYSTALS-Dilithium\n  (Digital Signatures)\n• AES-256-GCM\n  (Symmetric Encryption)\n• SHA-3\n  (Hashing)', 
                va='top', ha='left', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['crypto'], alpha=0.7))
        
        # Performance metrics
        ax.text(0.98, 0.4, 'PERFORMANCE\nMETRICS:\n\nTotal Handshake: ~10ms\nKey Size: 1184 bytes\nSignature: 3293 bytes\nSecurity Level: NIST-3\n\nQuantum Resistance:\n✓ Lattice-based security\n✓ Post-quantum safe', 
                va='top', ha='right', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['data'], alpha=0.7))
        
        self._finalize_plot(ax, 'qflare_pqc_handshake.png')
        return fig

    def create_secure_aggregation_diagram(self):
        """Create secure aggregation protocol diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        ax.text(0.5, 0.95, 'QFLARE Secure Aggregation Protocol', 
                ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Client devices
        client_positions = [(0.1, 0.7), (0.1, 0.5), (0.1, 0.3)]
        for i, (x, y) in enumerate(client_positions):
            self._draw_component(ax, f'Client {i+1}\n[Local Model]', x, y, 0.08, 0.06, self.colors['client'])
        
        # Secure aggregation components
        agg_components = [
            ('Secret Sharing\nProtocol', 0.35, 0.7),
            ('Homomorphic\nEncryption', 0.35, 0.5),
            ('Multi-Party\nComputation', 0.35, 0.3)
        ]
        
        for name, x, y in agg_components:
            self._draw_component(ax, name, x, y, 0.1, 0.08, self.colors['crypto'])
        
        # Edge aggregator
        self._draw_component(ax, 'Edge Aggregator\n[Secure Sum]', 0.6, 0.5, 0.12, 0.1, self.colors['edge'])
        
        # Global model
        self._draw_component(ax, 'Global Model\n[Updated Weights]', 0.85, 0.5, 0.1, 0.08, self.colors['server'])
        
        # Draw data flow
        # Clients to secure aggregation
        for i, (cx, cy) in enumerate(client_positions):
            ax.arrow(cx + 0.04, cy, 0.15, 0, head_width=0.02, head_length=0.02, fc='blue', ec='blue')
            ax.text(cx + 0.12, cy + 0.03, f'Encrypted\nGradients', ha='center', va='bottom', fontsize=8)
        
        # Aggregation to edge
        ax.arrow(0.45, 0.5, 0.1, 0, head_width=0.03, head_length=0.02, fc='green', ec='green')
        ax.text(0.5, 0.53, 'Aggregated\nUpdate', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Edge to server
        ax.arrow(0.72, 0.5, 0.08, 0, head_width=0.03, head_length=0.02, fc='red', ec='red')
        ax.text(0.76, 0.53, 'Final\nModel', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add privacy protection details
        ax.text(0.02, 0.15, 'PRIVACY PROTECTIONS:\n\n• Differential Privacy Noise\n• Gradient Clipping\n• Secret Sharing (t-of-n)\n• Homomorphic Operations\n• Zero-Knowledge Proofs', 
                va='top', ha='left', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['security'], alpha=0.7))
        
        # Add aggregation formula
        ax.text(0.6, 0.15, 'AGGREGATION FORMULA:\n\nG_global = Σ(w_i × G_i) + DP_noise\n\nWhere:\n• w_i = client weight\n• G_i = encrypted gradient\n• DP_noise = calibrated noise', 
                va='top', ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['data'], alpha=0.7))
        
        self._finalize_plot(ax, 'qflare_secure_aggregation.png')
        return fig

    def create_edge_node_architecture(self):
        """Create detailed edge node architecture"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        ax.text(0.5, 0.95, 'QFLARE Edge Node Architecture', 
                ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Draw edge node container
        edge_box = Rectangle((0.1, 0.15), 0.8, 0.75, linewidth=3, 
                           edgecolor=self.colors['border'], facecolor='none')
        ax.add_patch(edge_box)
        ax.text(0.5, 0.87, 'QFLARE Edge Node', ha='center', va='center', 
               fontsize=14, fontweight='bold')
        
        # Core components
        components = [
            ('PQC Handler\n[Kyber/Dilithium]', 0.2, 0.75, self.colors['crypto']),
            ('Client Manager\n[100+ Connections]', 0.5, 0.75, self.colors['client']),
            ('Load Balancer\n[Traffic Routing]', 0.8, 0.75, self.colors['network']),
            
            ('Secure Aggregator\n[MPC Protocol]', 0.2, 0.6, self.colors['edge']),
            ('Model Cache\n[Encrypted Storage]', 0.5, 0.6, self.colors['data']),
            ('Performance Monitor\n[Metrics Collection]', 0.8, 0.6, self.colors['security']),
            
            ('Privacy Engine\n[DP Calibration]', 0.2, 0.45, self.colors['security']),
            ('Update Validator\n[Anomaly Detection]', 0.5, 0.45, self.colors['edge']),
            ('Network Interface\n[Encrypted Comms]', 0.8, 0.45, self.colors['network']),
            
            ('Local Storage\n[Encrypted DB]', 0.2, 0.3, self.colors['data']),
            ('Audit Logger\n[Security Events]', 0.5, 0.3, self.colors['security']),
            ('Health Monitor\n[System Status]', 0.8, 0.3, self.colors['server'])
        ]
        
        for name, x, y, color in components:
            self._draw_component(ax, name, x, y, 0.12, 0.08, color)
        
        # Draw internal connections
        connections = [
            ((0.2, 0.75), (0.2, 0.6)),    # PQC to Aggregator
            ((0.5, 0.75), (0.5, 0.6)),    # Client Mgr to Cache
            ((0.8, 0.75), (0.8, 0.6)),    # Load Bal to Monitor
            ((0.2, 0.6), (0.5, 0.6)),     # Aggregator to Cache
            ((0.5, 0.6), (0.8, 0.6)),     # Cache to Monitor
            ((0.2, 0.45), (0.5, 0.45)),   # Privacy to Validator
            ((0.5, 0.45), (0.8, 0.45)),   # Validator to Network
        ]
        
        for (x1, y1), (x2, y2) in connections:
            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=1)
        
        # External connections
        ax.arrow(0.05, 0.5, 0.04, 0, head_width=0.02, head_length=0.01, fc='blue', ec='blue')
        ax.text(0.02, 0.52, 'Client\nDevices', ha='center', va='bottom', fontsize=9)
        
        ax.arrow(0.91, 0.5, 0.04, 0, head_width=0.02, head_length=0.01, fc='green', ec='green')
        ax.text(0.98, 0.52, 'Central\nServer', ha='center', va='bottom', fontsize=9)
        
        # Resource specifications
        ax.text(0.02, 0.12, 'HARDWARE SPECS:\n• 8-core ARM/x86 CPU\n• 16GB RAM minimum\n• Hardware Security Module\n• Gigabit Ethernet\n• Local SSD storage', 
                va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['data'], alpha=0.7))
        
        ax.text(0.98, 0.12, 'PERFORMANCE:\n• 100+ concurrent clients\n• <500ms PQC handshake\n• 10MB/s aggregation\n• 99.9% uptime SLA\n• Auto-scaling support', 
                va='top', ha='right', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['server'], alpha=0.7))
        
        self._finalize_plot(ax, 'qflare_edge_node_architecture.png')
        return fig

    def create_threat_model_diagram(self):
        """Create threat model and attack surface diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        ax.text(0.5, 0.95, 'QFLARE Threat Model & Attack Surface Analysis', 
                ha='center', va='center', fontsize=16, fontweight='bold')
        
        # System components with threat indicators
        components = [
            ('Client Devices', 0.15, 0.8, self.colors['client'], ['Device Compromise', 'Model Extraction']),
            ('Network Channel', 0.5, 0.8, self.colors['network'], ['MITM Attack', 'Traffic Analysis']),
            ('Edge Nodes', 0.85, 0.8, self.colors['edge'], ['Node Compromise', 'Side-Channel']),
            ('Aggregation Protocol', 0.15, 0.5, self.colors['crypto'], ['Crypto Attack', 'Key Recovery']),
            ('Central Server', 0.5, 0.5, self.colors['server'], ['Server Breach', 'Admin Access']),
            ('Model Storage', 0.85, 0.5, self.colors['data'], ['Data Leak', 'Unauthorized Access']),
            ('Privacy Mechanisms', 0.15, 0.2, self.colors['security'], ['DP Bypass', 'Inference Attack']),
            ('Monitoring Systems', 0.5, 0.2, self.colors['security'], ['Log Tampering', 'Alert Evasion']),
            ('Key Management', 0.85, 0.2, self.colors['crypto'], ['Key Theft', 'Weak Generation'])
        ]
        
        # Draw components with threat annotations
        for name, x, y, color, threats in components:
            # Main component box
            self._draw_component(ax, name, x, y, 0.12, 0.08, color)
            
            # Threat indicators (red triangles)
            for i, threat in enumerate(threats):
                triangle_x = x + 0.08 + i * 0.02
                triangle_y = y + 0.05
                triangle = plt.Polygon([(triangle_x, triangle_y), 
                                      (triangle_x + 0.01, triangle_y + 0.02), 
                                      (triangle_x - 0.01, triangle_y + 0.02)], 
                                     color='red', alpha=0.7)
                ax.add_patch(triangle)
        
        # Attack vectors with risk levels
        attack_vectors = [
            ('Quantum Computer Attack\n[Future Risk]', 0.3, 0.65, 'HIGH', 'orange'),
            ('Membership Inference\n[Current Risk]', 0.7, 0.65, 'MEDIUM', 'yellow'),
            ('Gradient Inversion\n[Current Risk]', 0.3, 0.35, 'HIGH', 'orange'),
            ('Model Poisoning\n[Current Risk]', 0.7, 0.35, 'CRITICAL', 'red')
        ]
        
        for attack, x, y, risk_level, risk_color in attack_vectors:
            # Attack box
            box = FancyBboxPatch((x-0.08, y-0.04), 0.16, 0.08, 
                               boxstyle="round,pad=0.01", 
                               facecolor=risk_color, alpha=0.3,
                               edgecolor='black', linewidth=1)
            ax.add_patch(box)
            ax.text(x, y, attack, ha='center', va='center', fontsize=9, fontweight='bold')
            
            # Risk level indicator
            ax.text(x, y-0.06, f'Risk: {risk_level}', ha='center', va='center', 
                   fontsize=8, color=risk_color, fontweight='bold')
        
        # Mitigation strategies
        mitigations = [
            'PQC Implementation', 'Differential Privacy', 'Secure Aggregation',
            'Device Attestation', 'Encrypted Storage', 'Access Control',
            'Anomaly Detection', 'Key Rotation', 'Zero-Trust Architecture'
        ]
        
        mitigation_text = '\n'.join([f'✓ {m}' for m in mitigations])
        ax.text(0.02, 0.4, f'MITIGATION STRATEGIES:\n\n{mitigation_text}', 
                va='top', ha='left', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['security'], alpha=0.7))
        
        # Risk assessment matrix
        ax.text(0.98, 0.4, 'RISK ASSESSMENT:\n\n🔴 CRITICAL: Immediate action\n🟠 HIGH: Priority mitigation\n🟡 MEDIUM: Planned response\n🟢 LOW: Monitor only\n\nOverall Risk: MEDIUM\n(with mitigations)', 
                va='top', ha='right', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['data'], alpha=0.7))
        
        self._finalize_plot(ax, 'qflare_threat_model.png')
        return fig

    def _draw_component(self, ax, text, x, y, width, height, color):
        """Draw a system component box"""
        box = FancyBboxPatch((x-width/2, y-height/2), width, height, 
                           boxstyle="round,pad=0.01", 
                           facecolor=color, edgecolor=self.colors['border'],
                           linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    def _draw_diamond(self, ax, x, y, width, height, color):
        """Draw a diamond shape for decisions"""
        diamond = plt.Polygon([(x, y+height/2), (x+width/2, y), 
                              (x, y-height/2), (x-width/2, y)], 
                             facecolor=color, edgecolor=self.colors['border'], linewidth=1.5)
        ax.add_patch(diamond)
    
    def _draw_arrow(self, ax, x1, y1, x2, y2, label):
        """Draw an arrow between components"""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'))
        # Add label at midpoint
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=8,
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    def _draw_flow_arrow(self, ax, x1, y1, x2, y2):
        """Draw a flow arrow with automatic curve for better visibility"""
        if abs(x1 - x2) > 0.3:  # Long horizontal arrows - add curve
            mid_x = (x1 + x2) / 2
            mid_y = max(y1, y2) + 0.05
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkblue',
                                     connectionstyle=f"arc3,rad=0.3"))
        else:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    def _draw_message_arrow(self, ax, x1, y1, x2, y2, message, timing):
        """Draw message arrow with timing information"""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='green'))
        
        # Message label
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.02, message, ha='center', va='bottom', fontsize=8)
        ax.text(mid_x, mid_y - 0.02, timing, ha='center', va='top', fontsize=8, 
               style='italic', color='red')
    
    def _finalize_plot(self, ax, filename):
        """Finalize plot settings and save"""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"✅ Diagram saved: {self.output_dir / filename}")

def main():
    """Generate all QFLARE system diagrams"""
    print("QFLARE System Architecture Diagram Generator")
    print("=" * 50)
    
    generator = QFLAREDiagramGenerator()
    
    diagrams = [
        ("Overall Architecture", generator.create_overall_architecture),
        ("Federated Learning Flow", generator.create_federated_learning_flow),
        ("PQC Handshake Sequence", generator.create_pqc_handshake_sequence),
        ("Secure Aggregation Protocol", generator.create_secure_aggregation_diagram),
        ("Edge Node Architecture", generator.create_edge_node_architecture),
        ("Threat Model Analysis", generator.create_threat_model_diagram)
    ]
    
    for name, func in diagrams:
        print(f"\nGenerating {name}...")
        try:
            fig = func()
            plt.close(fig)  # Free memory
            print(f"✅ {name} completed successfully")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    print(f"\n🎯 All diagrams generated in: {generator.output_dir}")
    print("\nGenerated Files:")
    for diagram_file in generator.output_dir.glob("*.png"):
        print(f"  📊 {diagram_file.name}")

if __name__ == "__main__":
    main()