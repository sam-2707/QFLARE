"""
QFLARE System Block Diagram Generator - Clean Style
Creates structured block diagrams matching the provided style
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from pathlib import Path

class QFLAREBlockDiagramGenerator:
    """Generate clean, structured QFLARE block diagrams"""
    
    def __init__(self, output_dir="docs/diagrams"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean, professional colors matching your style
        self.colors = {
            'box': '#FFFFFF',         # White background
            'border': '#000000',      # Black border
            'text': '#000000',        # Black text
            'arrow': '#000000'        # Black arrows
        }
        
        # Box styling
        self.box_style = {
            'boxstyle': 'round,pad=0.3',
            'facecolor': self.colors['box'],
            'edgecolor': self.colors['border'],
            'linewidth': 2
        }
    
    def _draw_box(self, ax, text, x, y, width=2.5, height=0.8):
        """Draw a clean rectangular box with text"""
        box = FancyBboxPatch(
            (x - width/2, y - height/2), 
            width, height,
            **self.box_style
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=10, fontweight='normal', 
                color=self.colors['text'], wrap=True)
    
    def _draw_arrow(self, ax, start_x, start_y, end_x, end_y, offset_y=0.4):
        """Draw a clean downward arrow"""
        ax.annotate('', xy=(end_x, end_y + offset_y), xytext=(start_x, start_y - offset_y),
                    arrowprops=dict(arrowstyle='->', lw=2, color=self.colors['arrow']))
    
    def _draw_horizontal_arrow(self, ax, start_x, start_y, end_x, end_y, offset_x=0.4):
        """Draw a clean horizontal arrow"""
        ax.annotate('', xy=(end_x - offset_x, end_y), xytext=(start_x + offset_x, start_y),
                    arrowprops=dict(arrowstyle='->', lw=2, color=self.colors['arrow']))

    def create_overall_system_diagram(self):
        """Create Overall System Block Diagram matching your style"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Title
        ax.text(6, 9.5, 'Overall System Block Diagram', ha='center', va='center', 
                fontsize=16, fontweight='bold')
        
        # Main flow blocks
        blocks = [
            ('Dataset Acquisition and\nPreparation', 6, 8.5),
            ('Dataset Labeling and\nClassification', 6, 7),
            ('Model Training and Evaluation', 6, 5.5),
            ('Real-time Prediction Pipeline', 6, 4),
            ('NIDS Dashboard', 6, 2.5)
        ]
        
        # Draw blocks and arrows
        for i, (text, x, y) in enumerate(blocks):
            self._draw_box(ax, text, x, y)
            if i < len(blocks) - 1:
                next_y = blocks[i + 1][2]
                self._draw_arrow(ax, x, y, x, next_y)
        
        # Set limits and remove axes
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'qflare_overall_system.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_data_acquisition_diagram(self):
        """Create Data Acquisition and Preparation diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Title
        ax.text(6, 9.5, 'Data Acquisition and Preparation', ha='center', va='center', 
                fontsize=16, fontweight='bold')
        
        # Flow blocks
        blocks = [
            ('Bot-IoT Dataset\n(5% dataset, 4x files)', 6, 8),
            ('Analysis\n(class count, attack types, features)', 6, 6.5),
            ('Feature Selection', 6, 5),
            ('Data Extraction\n(merging, shuffling)', 6, 3.5),
            ('Export as DB1.csv', 6, 2)
        ]
        
        # Draw blocks and arrows
        for i, (text, x, y) in enumerate(blocks):
            self._draw_box(ax, text, x, y)
            if i < len(blocks) - 1:
                next_y = blocks[i + 1][2]
                self._draw_arrow(ax, x, y, x, next_y)
        
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'qflare_data_acquisition.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_combined_dataset_diagram(self):
        """Create Combined Dataset Creation diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Title
        ax.text(7, 9.5, 'Combined Dataset Creation', ha='center', va='center', 
                fontsize=16, fontweight='bold')
        
        # Top level dataset
        self._draw_box(ax, 'Bot-IoT 5% Dataset', 7, 8.5, width=3)
        
        # Four files branching out
        files = [
            ('File 1', 2.5, 7),
            ('File 2', 4.5, 7),
            ('File 3', 6.5, 7),
            ('File 4', 8.5, 7)
        ]
        
        for name, x, y in files:
            self._draw_box(ax, name, x, y, width=1.5)
            # Arrow from dataset to file
            ax.annotate('', xy=(x, y + 0.4), xytext=(7, 8.1),
                        arrowprops=dict(arrowstyle='->', lw=2, color=self.colors['arrow']))
        
        # Merging process
        self._draw_box(ax, 'Merging and Shuffling', 7, 5.5, width=3)
        
        # Arrows from files to merging
        for _, x, y in files:
            self._draw_arrow(ax, x, y, 7, 5.5)
        
        # Final output
        self._draw_box(ax, 'DB1.csv\nCombined Dataset', 7, 4, width=3)
        self._draw_arrow(ax, 7, 5.5, 7, 4)
        
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'qflare_combined_dataset.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_federated_learning_flow(self):
        """Create Federated Learning Flow diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Title
        ax.text(7, 9.5, 'QFLARE Federated Learning Flow', ha='center', va='center', 
                fontsize=16, fontweight='bold')
        
        # Central server
        self._draw_box(ax, 'Central Server\n(Global Model)', 7, 7.5, width=3)
        
        # Edge nodes in a circle around server
        edge_nodes = [
            ('Edge Node 1\n(Local Training)', 3, 7.5),
            ('Edge Node 2\n(Local Training)', 11, 7.5),
            ('Edge Node 3\n(Local Training)', 7, 4.5),
        ]
        
        for name, x, y in edge_nodes:
            self._draw_box(ax, name, x, y, width=2.5)
            # Bidirectional arrows
            if x < 7:  # Left node
                self._draw_horizontal_arrow(ax, x, y, 7, 7.5)
                self._draw_horizontal_arrow(ax, 7, 7.3, x, y)
            elif x > 7:  # Right node
                self._draw_horizontal_arrow(ax, 7, 7.5, x, y)
                self._draw_horizontal_arrow(ax, x, y, 7, 7.3)
            else:  # Bottom node
                self._draw_arrow(ax, 7, 7.5, x, y)
                self._draw_arrow(ax, x, y, 7, 7.2)
        
        # Secure aggregation box
        self._draw_box(ax, 'Secure Aggregation\n(PQC Protected)', 7, 2.5, width=3.5)
        self._draw_arrow(ax, 7, 4.5, 7, 2.5)
        
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'qflare_federated_learning_flow.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_pqc_handshake_diagram(self):
        """Create PQC Handshake Protocol diagram (without response times)"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        
        # Title
        ax.text(7, 11.5, 'QFLARE Post-Quantum Cryptographic Handshake', ha='center', va='center', 
                fontsize=16, fontweight='bold')
        
        # Client and Server columns
        ax.text(3, 10.5, 'Client', ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(11, 10.5, 'Server', ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Handshake steps
        steps = [
            ('1. Generate Kyber Keypair\n(Private/Public Keys)', 3, 9.5),
            ('2. Send Public Key +\nConnection Request', 7, 8.8),
            ('3. Verify Client Key\nGenerate Server Keypair', 11, 8.1),
            ('4. Send Server Public Key +\nEncrypted Session Key', 7, 7.4),
            ('5. Decrypt Session Key\nVerify Server Identity', 3, 6.7),
            ('6. Generate Dilithium\nSignature for Auth', 3, 6),
            ('7. Send Signed\nAuthentication', 7, 5.3),
            ('8. Verify Signature\nEstablish Secure Channel', 11, 4.6),
            ('9. Secure Communication\nChannel Active', 7, 3.5)
        ]
        
        for i, (text, x, y) in enumerate(steps):
            if x == 7:  # Message arrows
                if i % 2 == 1:  # Client to Server
                    self._draw_horizontal_arrow(ax, 4, y, 10, y)
                    self._draw_box(ax, text, x, y, width=4, height=0.6)
                else:  # Server to Client
                    self._draw_horizontal_arrow(ax, 10, y, 4, y)
                    self._draw_box(ax, text, x, y, width=4, height=0.6)
            else:  # Process boxes
                self._draw_box(ax, text, x, y, width=3)
        
        # Timeline arrows
        ax.annotate('', xy=(3, 3), xytext=(3, 10),
                    arrowprops=dict(arrowstyle='->', lw=1, color='gray'))
        ax.annotate('', xy=(11, 3), xytext=(11, 10),
                    arrowprops=dict(arrowstyle='->', lw=1, color='gray'))
        
        ax.set_xlim(0, 14)
        ax.set_ylim(2, 12)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'qflare_pqc_handshake.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_secure_aggregation_diagram(self):
        """Create Secure Aggregation Process diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Title
        ax.text(6, 9.5, 'QFLARE Secure Aggregation Process', ha='center', va='center', 
                fontsize=16, fontweight='bold')
        
        # Process flow
        blocks = [
            ('Local Model Updates\n(Edge Nodes)', 6, 8.5),
            ('Apply Differential Privacy\n(Noise Addition)', 6, 7.2),
            ('Encrypt with PQC\n(CRYSTALS-Kyber)', 6, 5.9),
            ('Secure Multi-Party\nComputation', 6, 4.6),
            ('Aggregate Encrypted\nUpdates', 6, 3.3),
            ('Global Model Update\n(Central Server)', 6, 2)
        ]
        
        # Draw blocks and arrows
        for i, (text, x, y) in enumerate(blocks):
            self._draw_box(ax, text, x, y)
            if i < len(blocks) - 1:
                next_y = blocks[i + 1][2]
                self._draw_arrow(ax, x, y, x, next_y)
        
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'qflare_secure_aggregation.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_edge_node_architecture(self):
        """Create Edge Node Architecture diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Title
        ax.text(6, 9.5, 'QFLARE Edge Node Architecture', ha='center', va='center', 
                fontsize=16, fontweight='bold')
        
        # Main components
        blocks = [
            ('Local Data Storage\n(Encrypted)', 6, 8.5),
            ('Data Preprocessing\n(Feature Engineering)', 6, 7.2),
            ('Local Model Training\n(Privacy-Preserving)', 6, 5.9),
            ('Gradient Computation\n(Differential Privacy)', 6, 4.6),
            ('PQC Encryption\n(Model Updates)', 6, 3.3),
            ('Secure Communication\n(To Central Server)', 6, 2)
        ]
        
        # Draw blocks and arrows
        for i, (text, x, y) in enumerate(blocks):
            self._draw_box(ax, text, x, y)
            if i < len(blocks) - 1:
                next_y = blocks[i + 1][2]
                self._draw_arrow(ax, x, y, x, next_y)
        
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'qflare_edge_node_architecture.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_all_diagrams(self):
        """Generate all QFLARE block diagrams"""
        print("QFLARE Block Diagram Generator")
        print("=" * 50)
        print("Creating clean, structured block diagrams...")
        print()
        
        diagrams = [
            ("Overall System", self.create_overall_system_diagram),
            ("Data Acquisition", self.create_data_acquisition_diagram),
            ("Combined Dataset", self.create_combined_dataset_diagram),
            ("Federated Learning Flow", self.create_federated_learning_flow),
            ("PQC Handshake Protocol", self.create_pqc_handshake_diagram),
            ("Secure Aggregation", self.create_secure_aggregation_diagram),
            ("Edge Node Architecture", self.create_edge_node_architecture)
        ]
        
        generated = []
        for name, func in diagrams:
            try:
                print(f"📊 Generating {name}...")
                func()
                generated.append(f"✅ {name}")
                print(f"   ✅ Completed: {name}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                generated.append(f"❌ {name} - Error: {str(e)}")
        
        print()
        print("=" * 50)
        print("DIAGRAM GENERATION COMPLETE")
        print("=" * 50)
        print(f"📁 Output Directory: {self.output_dir}")
        print(f"📊 Total Diagrams: {len([g for g in generated if g.startswith('✅')])}")
        print()
        print("Generated Diagrams:")
        for status in generated:
            print(f"   {status}")
        
        return generated

if __name__ == "__main__":
    generator = QFLAREBlockDiagramGenerator()
    generator.generate_all_diagrams()