"""
QFLARE Methodology Slide Image Generator
Generates visual diagrams for all 10 methodology slides
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np
import os

# Create output directory
output_dir = "methodology_slides"
os.makedirs(output_dir, exist_ok=True)

# Set style
plt.style.use('default')
colors = {
    'primary': '#000000',
    'secondary': '#ffffff',
    'accent1': '#333333',
    'accent2': '#666666',
    'accent3': '#999999',
}

def create_slide_1_architecture():
    """SLIDE 1: System Architecture & Core Infrastructure"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'QFLARE System Architecture', 
            fontsize=24, fontweight='bold', ha='center', family='monospace')
    
    # Central Server
    server = FancyBboxPatch((3.5, 7), 3, 1.2, 
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor='#cccccc', linewidth=3)
    ax.add_patch(server)
    ax.text(5, 7.6, 'CENTRAL SERVER', fontsize=12, fontweight='bold', ha='center', family='monospace')
    ax.text(5, 7.3, 'Kyber-1024 + Dilithium-2', fontsize=9, ha='center', family='monospace')
    
    # Regional Aggregators
    for i, x in enumerate([1.5, 5, 8.5]):
        agg = FancyBboxPatch((x-0.75, 4.5), 1.5, 1, 
                             boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor='#e6e6e6', linewidth=2)
        ax.add_patch(agg)
        ax.text(x, 5.2, f'AGGREGATOR {i+1}', fontsize=9, fontweight='bold', ha='center', family='monospace')
        ax.text(x, 4.8, 'SMPC + Byzantine', fontsize=7, ha='center', family='monospace')
        
        # Arrow from server to aggregator
        arrow = FancyArrowPatch((5, 7), (x, 5.5),
                               arrowstyle='->', mutation_scale=20, linewidth=2,
                               color='black')
        ax.add_patch(arrow)
    
    # Edge Devices
    device_positions = [
        (0.5, 2), (1.5, 2), (2.5, 2),
        (4, 2), (5, 2), (6, 2),
        (7.5, 2), (8.5, 2), (9.5, 2)
    ]
    
    for i, (x, y) in enumerate(device_positions):
        device = Rectangle((x-0.3, y), 0.6, 0.8, 
                          edgecolor='black', facecolor='white', linewidth=1.5)
        ax.add_patch(device)
        ax.text(x, y+0.4, f'D{i+1}', fontsize=8, ha='center', family='monospace')
        
        # Connect to nearest aggregator
        agg_x = 1.5 if x < 3.5 else (5 if x < 7 else 8.5)
        arrow = FancyArrowPatch((x, y+0.8), (agg_x, 4.5),
                               arrowstyle='->', mutation_scale=15, linewidth=1,
                               color='gray', linestyle='--')
        ax.add_patch(arrow)
    
    # Key Storage
    storage = FancyBboxPatch((0.5, 0.2), 2.5, 0.8,
                            boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor='#ffffcc', linewidth=2)
    ax.add_patch(storage)
    ax.text(1.75, 0.7, 'PostgreSQL + KMS', fontsize=9, fontweight='bold', ha='center', family='monospace')
    ax.text(1.75, 0.4, 'Envelope Encryption', fontsize=7, ha='center', family='monospace')
    
    # PKI/KDC
    pki = FancyBboxPatch((7, 0.2), 2.5, 0.8,
                        boxstyle="round,pad=0.05",
                        edgecolor='black', facecolor='#ccffcc', linewidth=2)
    ax.add_patch(pki)
    ax.text(8.25, 0.7, 'Key Distribution', fontsize=9, fontweight='bold', ha='center', family='monospace')
    ax.text(8.25, 0.4, 'HSM-based CA', fontsize=7, ha='center', family='monospace')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#cccccc', edgecolor='black', label='Central Server'),
        mpatches.Patch(facecolor='#e6e6e6', edgecolor='black', label='Regional Aggregator'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='Edge Device'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, frameon=True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/slide_01_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated Slide 1: System Architecture")

def create_slide_2_cryptography():
    """SLIDE 2: Quantum-Resistant Cryptography"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Quantum-Resistant Cryptography', 
            fontsize=24, fontweight='bold', ha='center', family='monospace')
    
    # CRYSTALS-Kyber
    kyber_box = FancyBboxPatch((0.5, 6.5), 4, 2.5,
                              boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='#e6f2ff', linewidth=3)
    ax.add_patch(kyber_box)
    ax.text(2.5, 8.7, 'CRYSTALS-Kyber-1024', fontsize=14, fontweight='bold', ha='center', family='monospace')
    ax.text(2.5, 8.3, 'Key Encapsulation Mechanism', fontsize=10, ha='center', family='monospace')
    ax.text(2.5, 7.9, '• 1568-byte keys', fontsize=9, ha='center', family='monospace')
    ax.text(2.5, 7.5, '• 256-bit quantum security', fontsize=9, ha='center', family='monospace')
    ax.text(2.5, 7.1, '• Module-LWE hardness', fontsize=9, ha='center', family='monospace')
    
    # CRYSTALS-Dilithium
    dilithium_box = FancyBboxPatch((5.5, 6.5), 4, 2.5,
                                  boxstyle="round,pad=0.1",
                                  edgecolor='black', facecolor='#fff2e6', linewidth=3)
    ax.add_patch(dilithium_box)
    ax.text(7.5, 8.7, 'CRYSTALS-Dilithium-2', fontsize=14, fontweight='bold', ha='center', family='monospace')
    ax.text(7.5, 8.3, 'Digital Signature Scheme', fontsize=10, ha='center', family='monospace')
    ax.text(7.5, 7.9, '• 2420-byte signatures', fontsize=9, ha='center', family='monospace')
    ax.text(7.5, 7.5, '• 128-bit quantum security', fontsize=9, ha='center', family='monospace')
    ax.text(7.5, 7.1, '• Module-LWE + Module-SIS', fontsize=9, ha='center', family='monospace')
    
    # Hybrid Encryption Flow
    ax.text(5, 5.8, 'Hybrid Encryption Strategy', fontsize=16, fontweight='bold', ha='center', family='monospace')
    
    # Flow boxes
    flow_steps = [
        (1.5, 'Kyber KEM\nShared Secret', 4.5),
        (4, 'HKDF-SHA3\nKey Derivation', 4.5),
        (6.5, 'AES-256-GCM\nBulk Encryption', 4.5),
        (9, 'Dilithium\nSignature', 4.5)
    ]
    
    for i, (x, text, y) in enumerate(flow_steps):
        box = FancyBboxPatch((x-0.6, y-0.5), 1.2, 1,
                            boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor='#f0f0f0', linewidth=2)
        ax.add_patch(box)
        for j, line in enumerate(text.split('\n')):
            ax.text(x, y+0.15-j*0.3, line, fontsize=8, ha='center', fontweight='bold', family='monospace')
        
        # Arrow to next step
        if i < len(flow_steps) - 1:
            arrow = FancyArrowPatch((x+0.6, y), (flow_steps[i+1][0]-0.6, y),
                                   arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
            ax.add_patch(arrow)
    
    # SHA3 and AES boxes
    sha3_box = FancyBboxPatch((0.5, 2), 4, 1.5,
                             boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor='#e6ffe6', linewidth=2)
    ax.add_patch(sha3_box)
    ax.text(2.5, 3.1, 'SHA3-512 Hash', fontsize=12, fontweight='bold', ha='center', family='monospace')
    ax.text(2.5, 2.7, '256-bit quantum security', fontsize=9, ha='center', family='monospace')
    ax.text(2.5, 2.3, 'Grover resistance: 2^256', fontsize=9, ha='center', family='monospace')
    
    aes_box = FancyBboxPatch((5.5, 2), 4, 1.5,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#ffe6f2', linewidth=2)
    ax.add_patch(aes_box)
    ax.text(7.5, 3.1, 'AES-256-GCM', fontsize=12, fontweight='bold', ha='center', family='monospace')
    ax.text(7.5, 2.7, 'Bulk encryption efficiency', fontsize=9, ha='center', family='monospace')
    ax.text(7.5, 2.3, 'Authenticated encryption', fontsize=9, ha='center', family='monospace')
    
    # Security level indicator
    security_box = FancyBboxPatch((2, 0.3), 6, 0.8,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='black', facecolor='#ffcccc', linewidth=3)
    ax.add_patch(security_box)
    ax.text(5, 0.7, '🔒 QUANTUM SECURITY: 2^256 BKZ complexity', 
            fontsize=11, fontweight='bold', ha='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/slide_02_cryptography.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated Slide 2: Quantum-Resistant Cryptography")

def create_slide_3_training_protocol():
    """SLIDE 3: Federated Training Protocol"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Federated Training Protocol', 
            fontsize=24, fontweight='bold', ha='center', family='monospace')
    
    # Round timeline
    timeline_y = 8.5
    ax.plot([1, 9], [timeline_y, timeline_y], 'k-', linewidth=3)
    
    phases = [
        (1.5, 'Participant\nSelection'),
        (3.5, 'Model\nDistribution'),
        (5.5, 'Local\nTraining'),
        (7.5, 'Update\nSubmission')
    ]
    
    for i, (x, label) in enumerate(phases):
        # Phase marker
        circle = plt.Circle((x, timeline_y), 0.2, color='black', zorder=5)
        ax.add_patch(circle)
        ax.text(x, timeline_y-0.6, label, fontsize=9, ha='center', 
                fontweight='bold', family='monospace')
        ax.text(x, timeline_y-1.2, f'Phase {i+1}', fontsize=8, ha='center', 
                style='italic', family='monospace')
    
    # Phase 1: Participant Selection
    phase1_box = FancyBboxPatch((0.5, 5.5), 2.5, 1.5,
                               boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='#e6f2ff', linewidth=2)
    ax.add_patch(phase1_box)
    ax.text(1.75, 6.8, 'Select 10% devices', fontsize=9, fontweight='bold', ha='center', family='monospace')
    ax.text(1.75, 6.4, 'Generate SHA3 nonce', fontsize=8, ha='center', family='monospace')
    ax.text(1.75, 6.0, 'Sign with Dilithium', fontsize=8, ha='center', family='monospace')
    
    # Phase 2: Model Distribution
    phase2_box = FancyBboxPatch((3.5, 5.5), 2.5, 1.5,
                               boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='#fff2e6', linewidth=2)
    ax.add_patch(phase2_box)
    ax.text(4.75, 6.8, 'Kyber encapsulation', fontsize=9, fontweight='bold', ha='center', family='monospace')
    ax.text(4.75, 6.4, 'AES-256 encryption', fontsize=8, ha='center', family='monospace')
    ax.text(4.75, 6.0, 'Dilithium signature', fontsize=8, ha='center', family='monospace')
    
    # Phase 3: Local Training
    phase3_box = FancyBboxPatch((0.5, 3.5), 5.5, 1.5,
                               boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='#e6ffe6', linewidth=2)
    ax.add_patch(phase3_box)
    ax.text(3.25, 4.8, 'LOCAL TRAINING ON DEVICE', fontsize=10, fontweight='bold', ha='center', family='monospace')
    ax.text(3.25, 4.4, '1. Verify signature → Decapsulate → Decrypt model', fontsize=8, ha='center', family='monospace')
    ax.text(3.25, 4.0, '2. Perform E epochs of SGD on private data Dᵢ', fontsize=8, ha='center', family='monospace')
    ax.text(3.25, 3.6, '3. Compute gradient update: Δᵢ = wᵢ - wₜ', fontsize=8, ha='center', family='monospace')
    
    # Device illustration
    for i, x in enumerate([6.5, 7.5, 8.5]):
        device = Rectangle((x-0.25, 3.7), 0.5, 0.8,
                          edgecolor='black', facecolor='white', linewidth=1.5)
        ax.add_patch(device)
        ax.text(x, 4.1, f'D{i+1}', fontsize=8, ha='center', fontweight='bold', family='monospace')
        ax.text(x, 3.9, 'SGD', fontsize=6, ha='center', family='monospace')
    
    # Mathematical formulas
    formula_box = FancyBboxPatch((0.5, 1.5), 9, 1.5,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='#f0f0f0', linewidth=2)
    ax.add_patch(formula_box)
    ax.text(5, 2.7, 'Training Equations', fontsize=11, fontweight='bold', ha='center', family='monospace')
    ax.text(5, 2.3, 'Local Update: wᵢᵗ⁺¹ = wᵢᵗ - η∇L(wᵢᵗ; Dᵢ)', fontsize=9, ha='center', family='monospace')
    ax.text(5, 1.9, 'Gradient Δᵢ = wᵢᵗ⁺¹ - wₜ  |  Loss L(w;Dᵢ) = (1/|Dᵢ|)Σ ℓ(w;xⱼ,yⱼ)', fontsize=9, ha='center', family='monospace')
    
    # Security indicators
    security_indicators = [
        (1, 0.5, '🔐 Kyber-1024'),
        (3, 0.5, '✍️ Dilithium-2'),
        (5, 0.5, '🔒 AES-256-GCM'),
        (7, 0.5, '🔑 SHA3-512'),
        (9, 0.5, '🛡️ Quantum-Safe')
    ]
    for x, y, text in security_indicators:
        ax.text(x, y, text, fontsize=8, ha='center', fontweight='bold', family='monospace')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/slide_03_training_protocol.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated Slide 3: Federated Training Protocol")

def create_slide_4_differential_privacy():
    """SLIDE 4: Differential Privacy Protection"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Differential Privacy Protection', 
            fontsize=24, fontweight='bold', ha='center', family='monospace')
    
    # Privacy Guarantee Box
    privacy_box = FancyBboxPatch((1, 8), 8, 1,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='#ffe6e6', linewidth=3)
    ax.add_patch(privacy_box)
    ax.text(5, 8.6, 'THEOREM 6: (ε=0.1, δ=10⁻⁶)-Differential Privacy', 
            fontsize=12, fontweight='bold', ha='center', family='monospace')
    ax.text(5, 8.2, 'Individual training examples indistinguishable with probability ≤ e^ε ≈ 1.105', 
            fontsize=9, ha='center', family='monospace')
    
    # DP Pipeline
    ax.text(5, 7.3, 'Differential Privacy Pipeline', fontsize=14, fontweight='bold', ha='center', family='monospace')
    
    # Step 1: Gradient Clipping
    step1_box = FancyBboxPatch((0.5, 5), 2.5, 1.8,
                              boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='#e6f2ff', linewidth=2)
    ax.add_patch(step1_box)
    ax.text(1.75, 6.5, 'STEP 1: Clipping', fontsize=10, fontweight='bold', ha='center', family='monospace')
    ax.text(1.75, 6.1, 'Compute ||Δᵢ||₂', fontsize=8, ha='center', family='monospace')
    ax.text(1.75, 5.7, 'Δ̃ᵢ = Δᵢ / max(1, ||Δᵢ||₂/C)', fontsize=8, ha='center', family='monospace')
    ax.text(1.75, 5.3, 'Sensitivity: Δf = 1', fontsize=8, ha='center', family='monospace')
    
    arrow1 = FancyArrowPatch((3, 5.9), (3.7, 5.9),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # Step 2: Noise Addition
    step2_box = FancyBboxPatch((3.7, 5), 2.6, 1.8,
                              boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='#fff2e6', linewidth=2)
    ax.add_patch(step2_box)
    ax.text(5, 6.5, 'STEP 2: Noise', fontsize=10, fontweight='bold', ha='center', family='monospace')
    ax.text(5, 6.1, 'ξᵢ ~ N(0, σ²I)', fontsize=8, ha='center', family='monospace')
    ax.text(5, 5.7, 'σ = √(2ln(1.25/δ)·Δf)/ε', fontsize=8, ha='center', family='monospace')
    ax.text(5, 5.3, 'σ ≈ 47.7', fontsize=8, ha='center', family='monospace')
    
    arrow2 = FancyArrowPatch((6.3, 5.9), (7, 5.9),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # Step 3: Private Gradient
    step3_box = FancyBboxPatch((7, 5), 2.5, 1.8,
                              boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='#e6ffe6', linewidth=2)
    ax.add_patch(step3_box)
    ax.text(8.25, 6.5, 'STEP 3: Output', fontsize=10, fontweight='bold', ha='center', family='monospace')
    ax.text(8.25, 6.1, 'Δ̂ᵢ = Δ̃ᵢ + ξᵢ', fontsize=8, ha='center', family='monospace')
    ax.text(8.25, 5.7, 'Private gradient', fontsize=8, ha='center', family='monospace')
    ax.text(8.25, 5.3, 'Ready for aggregation', fontsize=8, ha='center', family='monospace')
    
    # Visual representation of clipping and noise
    # Original gradient
    ax.arrow(1, 3.8, 0.8, 0, head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
    ax.text(1.4, 3.5, 'Original Δᵢ', fontsize=8, ha='center', family='monospace')
    
    # Clipped gradient
    ax.arrow(3, 3.8, 0.5, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue', linewidth=2)
    ax.text(3.25, 3.5, 'Clipped Δ̃ᵢ', fontsize=8, ha='center', family='monospace')
    
    # Noisy gradient
    ax.arrow(5, 3.8, 0.5, 0.2, head_width=0.1, head_length=0.1, fc='green', ec='green', linewidth=2)
    ax.text(5.25, 3.5, 'Noisy Δ̂ᵢ', fontsize=8, ha='center', family='monospace')
    
    # Privacy Budget Accounting
    budget_box = FancyBboxPatch((0.5, 1.5), 9, 1.3,
                               boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='#f0f0f0', linewidth=2)
    ax.add_patch(budget_box)
    ax.text(5, 2.5, 'Privacy Budget Accounting (Moment Accountant)', 
            fontsize=11, fontweight='bold', ha='center', family='monospace')
    ax.text(5, 2.1, 'Composition over T rounds: (ε√(2T ln(1/δ)), Tδ)-DP', 
            fontsize=9, ha='center', family='monospace')
    ax.text(5, 1.7, 'Cumulative privacy loss tracking ensures long-term privacy guarantee', 
            fontsize=9, ha='center', family='monospace')
    
    # Privacy parameters
    param_box = FancyBboxPatch((1.5, 0.3), 7, 0.8,
                              boxstyle="round,pad=0.05",
                              edgecolor='black', facecolor='#ccffcc', linewidth=2)
    ax.add_patch(param_box)
    ax.text(5, 0.7, '🔒 ε = 0.1  |  δ = 10⁻⁶  |  C = 1.0  |  σ = 47.7  |  E[||ξ||] = σ√d', 
            fontsize=9, fontweight='bold', ha='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/slide_04_differential_privacy.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated Slide 4: Differential Privacy Protection")

def create_slide_5_update_submission():
    """SLIDE 5: Secure Update Submission"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Secure Update Submission', 
            fontsize=24, fontweight='bold', ha='center', family='monospace')
    
    # Device side
    ax.text(2.5, 8.7, 'EDGE DEVICE', fontsize=14, fontweight='bold', ha='center', family='monospace')
    
    # Step 1: Encryption
    enc_box = FancyBboxPatch((0.5, 7), 4, 1.3,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#e6f2ff', linewidth=2)
    ax.add_patch(enc_box)
    ax.text(2.5, 7.9, '1. UPDATE ENCRYPTION', fontsize=10, fontweight='bold', ha='center', family='monospace')
    ax.text(2.5, 7.5, 'Generate random AES-256 key k', fontsize=8, ha='center', family='monospace')
    ax.text(2.5, 7.2, 'Encrypt: c = AES-GCM(k, Δ̂ᵢ)', fontsize=8, ha='center', family='monospace')
    
    arrow1 = FancyArrowPatch((2.5, 7), (2.5, 6.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # Step 2: Key Encapsulation
    kem_box = FancyBboxPatch((0.5, 5.2), 4, 1.3,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#fff2e6', linewidth=2)
    ax.add_patch(kem_box)
    ax.text(2.5, 6.1, '2. KEY ENCAPSULATION', fontsize=10, fontweight='bold', ha='center', family='monospace')
    ax.text(2.5, 5.7, 'Encapsulate: (ct, ss) = Kyber.Encaps(pk_server)', fontsize=8, ha='center', family='monospace')
    ax.text(2.5, 5.4, 'Encrypted key: ct', fontsize=8, ha='center', family='monospace')
    
    arrow2 = FancyArrowPatch((2.5, 5.2), (2.5, 4.7),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # Step 3: Signature
    sig_box = FancyBboxPatch((0.5, 3.4), 4, 1.3,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#e6ffe6', linewidth=2)
    ax.add_patch(sig_box)
    ax.text(2.5, 4.3, '3. DIGITAL SIGNATURE', fontsize=10, fontweight='bold', ha='center', family='monospace')
    ax.text(2.5, 3.9, 'Commitment: hᵢ = SHA3-512(Δ̂ᵢ || nonce || round)', fontsize=8, ha='center', family='monospace')
    ax.text(2.5, 3.6, 'Signature: σᵢ = Dilithium.Sign(sk_device, hᵢ)', fontsize=8, ha='center', family='monospace')
    
    # Transmission arrow
    trans_arrow = FancyArrowPatch((4.5, 5.5), (5.5, 5.5),
                                 arrowstyle='->', mutation_scale=30, linewidth=3, color='black')
    ax.add_patch(trans_arrow)
    ax.text(5, 5.8, 'TRANSMIT', fontsize=9, fontweight='bold', ha='center', family='monospace')
    
    # Server side
    ax.text(7.5, 8.7, 'CENTRAL SERVER', fontsize=14, fontweight='bold', ha='center', family='monospace')
    
    # Received packet
    packet_box = FancyBboxPatch((5.5, 7), 4, 1.3,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='#f0f0f0', linewidth=2)
    ax.add_patch(packet_box)
    ax.text(7.5, 7.9, 'RECEIVED PACKET', fontsize=10, fontweight='bold', ha='center', family='monospace')
    ax.text(7.5, 7.5, '• Ciphertext c', fontsize=8, ha='center', family='monospace')
    ax.text(7.5, 7.2, '• Kyber ciphertext ct  •  Signature σᵢ', fontsize=8, ha='center', family='monospace')
    
    arrow3 = FancyArrowPatch((7.5, 7), (7.5, 6.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow3)
    
    # Verification
    verify_box = FancyBboxPatch((5.5, 5.2), 4, 1.3,
                               boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='#ccffcc', linewidth=2)
    ax.add_patch(verify_box)
    ax.text(7.5, 6.1, 'VERIFICATION', fontsize=10, fontweight='bold', ha='center', family='monospace')
    ax.text(7.5, 5.7, 'Verify: Dilithium.Verify(pk_device, hᵢ, σᵢ)', fontsize=8, ha='center', family='monospace')
    ax.text(7.5, 5.4, 'Check certificate validity', fontsize=8, ha='center', family='monospace')
    
    arrow4 = FancyArrowPatch((7.5, 5.2), (7.5, 4.7),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow4)
    
    # Decryption
    dec_box = FancyBboxPatch((5.5, 3.4), 4, 1.3,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#ffe6f2', linewidth=2)
    ax.add_patch(dec_box)
    ax.text(7.5, 4.3, 'DECRYPTION', fontsize=10, fontweight='bold', ha='center', family='monospace')
    ax.text(7.5, 3.9, 'Decapsulate: k = Kyber.Decaps(sk_server, ct)', fontsize=8, ha='center', family='monospace')
    ax.text(7.5, 3.6, 'Decrypt: Δ̂ᵢ = AES-GCM.Decrypt(k, c)', fontsize=8, ha='center', family='monospace')
    
    # Optional ZK Proof
    zk_box = FancyBboxPatch((1, 1.5), 8, 1.2,
                           boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor='#f9f9f9', linewidth=2)
    ax.add_patch(zk_box)
    ax.text(5, 2.4, 'OPTIONAL: Zero-Knowledge Proof (every 10th round)', 
            fontsize=10, fontweight='bold', ha='center', family='monospace')
    ax.text(5, 2.0, 'Generate πᵢ proving correct local training without revealing private data', 
            fontsize=9, ha='center', family='monospace')
    ax.text(5, 1.7, 'Server verifies: ZK.Verify(public_params, Δ̂ᵢ, πᵢ) = Accept/Reject', 
            fontsize=9, ha='center', family='monospace')
    
    # Security properties
    security_box = FancyBboxPatch((1.5, 0.3), 7, 0.7,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='black', facecolor='#ffcccc', linewidth=2)
    ax.add_patch(security_box)
    ax.text(5, 0.7, '🔐 Authentication  |  🔒 Confidentiality  |  ✍️ Non-repudiation  |  🛡️ Integrity', 
            fontsize=9, fontweight='bold', ha='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/slide_05_update_submission.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated Slide 5: Secure Update Submission")

# Generate all diagrams
if __name__ == "__main__":
    print("\n🎨 Generating QFLARE Methodology Slide Images...")
    print("=" * 60)
    
    create_slide_1_architecture()
    create_slide_2_cryptography()
    create_slide_3_training_protocol()
    create_slide_4_differential_privacy()
    create_slide_5_update_submission()
    
    # Note: Slides 6-10 will be created in part 2
    print("=" * 60)
    print(f"\n✅ Generated 5 slides in '{output_dir}/' directory")
    print("\n📋 Next: Run part 2 script to generate slides 6-10")
