"""
QFLARE Methodology Slide Image Generator - Part 2
Generates visual diagrams for slides 6-10
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Wedge
import numpy as np
import os

# Create output directory
output_dir = "methodology_slides"
os.makedirs(output_dir, exist_ok=True)

def create_slide_6_byzantine_aggregation():
    """SLIDE 6: Byzantine-Resilient Aggregation"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Byzantine-Resilient Aggregation', 
            fontsize=24, fontweight='bold', ha='center', family='monospace')
    
    # Multi-layer verification pipeline
    ax.text(5, 8.7, 'Multi-Layer Defense Pipeline', fontsize=14, fontweight='bold', ha='center', family='monospace')
    
    # Layer 1: Cryptographic Validation
    layer1 = FancyBboxPatch((0.5, 7), 9, 1.2,
                           boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor='#e6f2ff', linewidth=2)
    ax.add_patch(layer1)
    ax.text(5, 7.8, 'LAYER 1: Cryptographic Validation', fontsize=10, fontweight='bold', ha='center', family='monospace')
    ax.text(5, 7.4, 'Verify Dilithium signature → Check certificate → Verify SHA3 commitment → Optional ZK proof', 
            fontsize=8, ha='center', family='monospace')
    
    arrow1 = FancyArrowPatch((5, 7), (5, 6.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # Layer 2: Statistical Byzantine Detection
    layer2 = FancyBboxPatch((0.5, 5.3), 9, 1.2,
                           boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor='#fff2e6', linewidth=2)
    ax.add_patch(layer2)
    ax.text(5, 6.1, 'LAYER 2: Statistical Byzantine Detection (Krum Algorithm)', 
            fontsize=10, fontweight='bold', ha='center', family='monospace')
    ax.text(5, 5.7, 'Compute pairwise distances → Select m peer-similar updates → Identify outliers → Apply median filtering', 
            fontsize=8, ha='center', family='monospace')
    
    arrow2 = FancyArrowPatch((5, 5.3), (5, 4.8),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # Layer 3: Reputation Management
    layer3 = FancyBboxPatch((0.5, 3.6), 9, 1.2,
                           boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor='#e6ffe6', linewidth=2)
    ax.add_patch(layer3)
    ax.text(5, 4.4, 'LAYER 3: Reputation Management', fontsize=10, fontweight='bold', ha='center', family='monospace')
    ax.text(5, 4.0, 'Track Rep[Dᵢ] scores → Decrease for suspicious: Rep ← 0.9·Rep → Exclude persistent attackers', 
            fontsize=8, ha='center', family='monospace')
    
    arrow3 = FancyArrowPatch((5, 3.6), (5, 3.1),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow3)
    
    # Global Model Update
    update_box = FancyBboxPatch((1, 2.3), 8, 0.8,
                               boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='#ccffcc', linewidth=3)
    ax.add_patch(update_box)
    ax.text(5, 2.7, 'GLOBAL MODEL UPDATE', fontsize=11, fontweight='bold', ha='center', family='monospace')
    
    # Visual representation of filtering
    # Before filtering
    ax.text(2, 1.5, 'BEFORE FILTERING', fontsize=9, fontweight='bold', ha='center', family='monospace')
    good_updates = [(1.3, 0.8), (1.7, 0.9), (2.1, 0.85), (2.5, 0.9)]
    bad_updates = [(1.5, 0.3), (2.3, 0.2)]
    
    for x, y in good_updates:
        circle = plt.Circle((x, y), 0.08, color='green', alpha=0.6)
        ax.add_patch(circle)
    for x, y in bad_updates:
        circle = plt.Circle((x, y), 0.08, color='red', alpha=0.6)
        ax.add_patch(circle)
    
    # After filtering
    ax.text(5, 1.5, 'AFTER FILTERING', fontsize=9, fontweight='bold', ha='center', family='monospace')
    for x, y in [(4.3, 0.8), (4.7, 0.9), (5.1, 0.85), (5.5, 0.9)]:
        circle = plt.Circle((x, y), 0.08, color='green', alpha=0.6)
        ax.add_patch(circle)
    
    # Aggregated result
    ax.text(8, 1.5, 'AGGREGATED', fontsize=9, fontweight='bold', ha='center', family='monospace')
    big_circle = plt.Circle((8, 0.85), 0.15, color='blue', alpha=0.8)
    ax.add_patch(big_circle)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='green', alpha=0.6, label='Valid Updates'),
        mpatches.Patch(facecolor='red', alpha=0.6, label='Byzantine/Malicious'),
        mpatches.Patch(facecolor='blue', alpha=0.8, label='Filtered Aggregate'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, frameon=True)
    
    # Byzantine tolerance guarantee
    tolerance_box = FancyBboxPatch((1.5, 0.1), 7, 0.5,
                                  boxstyle="round,pad=0.05",
                                  edgecolor='black', facecolor='#ffcccc', linewidth=2)
    ax.add_patch(tolerance_box)
    ax.text(5, 0.35, '🛡️ TOLERANCE: f < n/3 Byzantine participants (33% threshold)', 
            fontsize=10, fontweight='bold', ha='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/slide_06_byzantine_aggregation.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated Slide 6: Byzantine-Resilient Aggregation")

def create_slide_7_security_analysis():
    """SLIDE 7: Formal Security Analysis"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Formal Security Analysis', 
            fontsize=24, fontweight='bold', ha='center', family='monospace')
    
    # Theorem 4: Kyber Security
    theorem4 = FancyBboxPatch((0.5, 7.2), 9, 1.8,
                             boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor='#e6f2ff', linewidth=3)
    ax.add_patch(theorem4)
    ax.text(5, 8.7, 'THEOREM 4: Quantum-Safe Key Exchange', 
            fontsize=12, fontweight='bold', ha='center', family='monospace')
    ax.text(5, 8.3, 'Security Property: IND-CCA2 (Indistinguishability under Adaptive Chosen Ciphertext Attack)', 
            fontsize=9, ha='center', family='monospace')
    ax.text(5, 7.9, 'Hardness Assumption: Module-LWE with parameters (n=256, q=3329, η=2)', 
            fontsize=9, ha='center', family='monospace')
    ax.text(5, 7.5, 'Quantum Cryptanalysis: BKZ lattice attack complexity ≥ 2^256 operations', 
            fontsize=9, ha='center', family='monospace')
    
    # Theorem 5: Dilithium Security
    theorem5 = FancyBboxPatch((0.5, 5.1), 9, 2.0,
                             boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor='#fff2e6', linewidth=3)
    ax.add_patch(theorem5)
    ax.text(5, 6.8, 'THEOREM 5: Unforgeable Signatures', 
            fontsize=12, fontweight='bold', ha='center', family='monospace')
    ax.text(5, 6.4, 'Security Property: EU-CMA (Existential Unforgeability under Chosen Message Attack)', 
            fontsize=9, ha='center', family='monospace')
    ax.text(5, 6.0, 'Hardness Assumptions: Module-LWE + Module-SIS (Short Integer Solution)', 
            fontsize=9, ha='center', family='monospace')
    ax.text(5, 5.6, 'Forgery Requirements: Hash collision (negligible) OR solve MSIS (hard) OR distinguish MLWE (hard)', 
            fontsize=9, ha='center', family='monospace')
    ax.text(5, 5.3, 'Quantum Security: 128-bit classical, 64-bit quantum (Grover speedup accounted)', 
            fontsize=9, ha='center', family='monospace')
    
    # Byzantine Resilience
    byzantine = FancyBboxPatch((0.5, 3.2), 9, 1.8,
                              boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='#e6ffe6', linewidth=3)
    ax.add_patch(byzantine)
    ax.text(5, 4.7, 'BYZANTINE FAULT TOLERANCE', 
            fontsize=12, fontweight='bold', ha='center', family='monospace')
    ax.text(5, 4.3, 'Defense Layers: Cryptographic verification + Statistical filtering (Krum) + Reputation management', 
            fontsize=9, ha='center', family='monospace')
    ax.text(5, 3.9, 'Tolerance Threshold: f < n/3 malicious participants', 
            fontsize=9, ha='center', family='monospace')
    ax.text(5, 3.5, 'Convergence Guarantee: Correct global model convergence despite Byzantine behavior', 
            fontsize=9, ha='center', family='monospace')
    
    # Security Reduction Diagram
    ax.text(5, 2.7, 'Security Reduction Chain', fontsize=12, fontweight='bold', ha='center', family='monospace')
    
    # Reduction boxes
    reductions = [
        (1.5, 'QFLARE\nSecurity', 2.0),
        (3.5, 'Kyber/Dilithium\nSecurity', 2.0),
        (5.5, 'MLWE/MSIS\nHardness', 2.0),
        (7.5, 'Lattice\nProblems', 2.0)
    ]
    
    for i, (x, text, y) in enumerate(reductions):
        box = FancyBboxPatch((x-0.5, y-0.4), 1, 0.8,
                            boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor='#f0f0f0', linewidth=2)
        ax.add_patch(box)
        for j, line in enumerate(text.split('\n')):
            ax.text(x, y+0.1-j*0.25, line, fontsize=8, ha='center', fontweight='bold', family='monospace')
        
        if i < len(reductions) - 1:
            arrow = FancyArrowPatch((x+0.5, y), (reductions[i+1][0]-0.5, y),
                                   arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
            ax.add_patch(arrow)
            ax.text((x + reductions[i+1][0])/2, y+0.5, 'reduces to', 
                   fontsize=7, ha='center', style='italic', family='monospace')
    
    # Formal Verification Tools
    verification = FancyBboxPatch((0.5, 0.5), 9, 1.0,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='black', facecolor='#f9f9f9', linewidth=2)
    ax.add_patch(verification)
    ax.text(5, 1.2, 'FORMAL VERIFICATION TOOLS', fontsize=11, fontweight='bold', ha='center', family='monospace')
    ax.text(5, 0.8, 'Isabelle/HOL (Crypto correctness) | Tamarin (Protocol properties) | SPIN (Deadlock freedom)', 
            fontsize=9, ha='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/slide_07_security_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated Slide 7: Formal Security Analysis")

def create_slide_8_implementation():
    """SLIDE 8: Implementation & Performance"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Implementation & Performance', 
            fontsize=24, fontweight='bold', ha='center', family='monospace')
    
    # Architecture Stack
    ax.text(1.5, 8.7, 'PRODUCTION ARCHITECTURE', fontsize=12, fontweight='bold', ha='left', family='monospace')
    
    # Stack layers
    layers = [
        ('Web Interface & REST API', '#e6f2ff', 8.0),
        ('FastAPI + WebSocket', '#fff2e6', 7.4),
        ('SQLAlchemy ORM', '#e6ffe6', 6.8),
        ('PostgreSQL + Redis Cache', '#ffe6f2', 6.2),
        ('Cloud KMS Integration', '#f0f0f0', 5.6)
    ]
    
    for text, color, y in layers:
        box = FancyBboxPatch((0.5, y), 3.5, 0.5,
                            boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(2.25, y+0.25, text, fontsize=9, ha='center', fontweight='bold', family='monospace')
    
    # Performance Optimizations
    ax.text(6.5, 8.7, 'OPTIMIZATIONS', fontsize=12, fontweight='bold', ha='left', family='monospace')
    
    optimizations = [
        ('Batch Signature Verification', '70% reduction', 8.0),
        ('Connection Pooling', 'Concurrent handling', 7.4),
        ('Intelligent Key Caching', 'Reduced KMS calls', 6.8),
        ('AsyncIO Architecture', '1000+ devices', 6.2),
        ('Merkle Tree Batching', 'Efficient verification', 5.6)
    ]
    
    for text, metric, y in optimizations:
        box = FancyBboxPatch((5, y), 4.5, 0.5,
                            boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor='#f9f9f9', linewidth=1.5)
        ax.add_patch(box)
        ax.text(5.3, y+0.25, f'• {text}', fontsize=8, ha='left', family='monospace')
        ax.text(9.2, y+0.25, metric, fontsize=8, ha='right', style='italic', family='monospace')
    
    # Performance Metrics Bar Chart
    ax.text(2.5, 4.9, 'PERFORMANCE OVERHEAD', fontsize=11, fontweight='bold', ha='center', family='monospace')
    
    metrics = [
        ('Communication', 15, '#e6f2ff'),
        ('Time/Round', 77, '#fff2e6'),
        ('Energy', 15.1, '#e6ffe6'),
        ('Accuracy Loss', 3.6, '#ffe6f2')
    ]
    
    bar_width = 0.6
    x_positions = [0.8, 1.8, 2.8, 3.8]
    
    for i, (label, value, color) in enumerate(metrics):
        # Bar
        bar_height = value / 100 * 2.5  # Scale to fit
        rect = Rectangle((x_positions[i], 1.5), bar_width, bar_height,
                        facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        # Value label
        ax.text(x_positions[i] + bar_width/2, 1.5 + bar_height + 0.1, 
               f'{value}%', fontsize=9, ha='center', fontweight='bold', family='monospace')
        # X label
        ax.text(x_positions[i] + bar_width/2, 1.3, label, 
               fontsize=7, ha='center', rotation=0, family='monospace')
    
    # Y axis
    ax.plot([0.5, 0.5], [1.5, 4.2], 'k-', linewidth=2)
    for y_val in [0, 25, 50, 75, 100]:
        y_pos = 1.5 + (y_val / 100 * 2.5)
        ax.plot([0.45, 0.5], [y_pos, y_pos], 'k-', linewidth=1)
        ax.text(0.35, y_pos, f'{y_val}%', fontsize=7, ha='right', family='monospace')
    
    # Security Validation Results
    ax.text(7, 4.9, 'SECURITY VALIDATION', fontsize=11, fontweight='bold', ha='center', family='monospace')
    
    security_results = [
        ('Membership Inference', '52%', 'green'),
        ('Model Inversion', '4.7%', 'green'),
        ('Byzantine Detection', '97.3%', 'green'),
        ('Quantum Resistance', '2^256', 'green')
    ]
    
    for i, (attack, result, status_color) in enumerate(security_results):
        y = 4.3 - i * 0.35
        # Status indicator
        circle = plt.Circle((5.5, y), 0.08, color=status_color, alpha=0.7)
        ax.add_patch(circle)
        # Attack name
        ax.text(5.7, y, attack, fontsize=8, ha='left', family='monospace')
        # Result
        ax.text(8.3, y, result, fontsize=8, ha='right', fontweight='bold', family='monospace')
    
    # Enterprise Features
    features_box = FancyBboxPatch((0.5, 0.3), 9, 0.8,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='black', facecolor='#ccffcc', linewidth=2)
    ax.add_patch(features_box)
    ax.text(5, 0.8, 'ENTERPRISE FEATURES', fontsize=10, fontweight='bold', ha='center', family='monospace')
    ax.text(5, 0.5, 'Automated KMS rotation | Audit logging | Graceful fallback | GDPR/HIPAA compliance | Open-source', 
            fontsize=8, ha='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/slide_08_implementation.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated Slide 8: Implementation & Performance")

def create_slide_9_experimental_results():
    """SLIDE 9: Experimental Validation Results"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Experimental Validation Results', 
            fontsize=24, fontweight='bold', ha='center', family='monospace')
    
    # Multi-dataset testing
    ax.text(1.5, 8.7, 'DATASET COVERAGE', fontsize=11, fontweight='bold', ha='left', family='monospace')
    
    datasets = [
        'MNIST', 'Fashion-MNIST', 'CIFAR-10', 'CIFAR-100',
        'SVHN', 'EMNIST', 'KMNIST', 'ImageNet'
    ]
    
    for i, dataset in enumerate(datasets):
        x = 0.5 + (i % 4) * 1
        y = 8.1 - (i // 4) * 0.5
        box = FancyBboxPatch((x, y), 0.9, 0.35,
                            boxstyle="round,pad=0.03",
                            edgecolor='black', facecolor='#e6f2ff', linewidth=1)
        ax.add_patch(box)
        ax.text(x + 0.45, y + 0.175, dataset, fontsize=7, ha='center', family='monospace')
    
    # Accuracy Performance Chart
    ax.text(2.5, 6.8, 'ACCURACY PERFORMANCE', fontsize=11, fontweight='bold', ha='center', family='monospace')
    
    # Stacked comparison
    systems = ['Baseline', 'QFLARE']
    accuracies = [92.5, 89.3]
    colors_acc = ['#90EE90', '#FFB6C1']
    
    bar_width = 0.8
    x_pos = [1, 2.5]
    
    for i, (system, acc, color) in enumerate(zip(systems, accuracies, colors_acc)):
        # Bar
        bar_height = acc / 100 * 3.5
        rect = Rectangle((x_pos[i], 2.5), bar_width, bar_height,
                        facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        # Value
        ax.text(x_pos[i] + bar_width/2, 2.5 + bar_height + 0.1,
               f'{acc}%', fontsize=10, ha='center', fontweight='bold', family='monospace')
        # Label
        ax.text(x_pos[i] + bar_width/2, 2.2, system,
               fontsize=9, ha='center', family='monospace')
    
    # Accuracy loss indicator
    arrow = FancyArrowPatch((1.4, 5.7), (2.5, 5.5),
                           arrowstyle='<->', mutation_scale=15, linewidth=2, color='red')
    ax.add_patch(arrow)
    ax.text(1.95, 5.9, '3.2% loss', fontsize=8, ha='center', color='red', 
           fontweight='bold', family='monospace')
    
    # Scalability Results
    ax.text(6.5, 6.8, 'SCALABILITY', fontsize=11, fontweight='bold', ha='center', family='monospace')
    
    # Line graph
    devices = [100, 500, 1000]
    overhead = [1.68, 1.75, 1.75]
    
    # Plot line
    x_coords = [5 + i * 0.8 for i in range(len(devices))]
    y_coords = [2.5 + oh * 0.8 for oh in overhead]
    
    ax.plot(x_coords, y_coords, 'o-', color='blue', linewidth=2, markersize=8)
    
    # Data points
    for x, y, dev, oh in zip(x_coords, y_coords, devices, overhead):
        ax.text(x, y + 0.3, f'{dev}\ndevices', fontsize=7, ha='center', family='monospace')
        ax.text(x, y - 0.3, f'{oh}×', fontsize=7, ha='center', fontweight='bold', family='monospace')
    
    # Axis
    ax.plot([4.8, 7], [2.5, 2.5], 'k-', linewidth=1)
    ax.plot([4.8, 4.8], [2.5, 4.5], 'k-', linewidth=1)
    ax.text(4.5, 3.5, 'Overhead', fontsize=8, ha='center', rotation=90, family='monospace')
    ax.text(5.9, 2.2, '# Devices', fontsize=8, ha='center', family='monospace')
    
    # Comparative Analysis
    ax.text(5, 1.8, 'COMPARATIVE ANALYSIS (Score /10)', fontsize=11, fontweight='bold', ha='center', family='monospace')
    
    systems_comp = [
        ('FedAvg', 2.1),
        ('DP-FedAvg', 4.2),
        ('Krum', 4.8),
        ('BRIDGE', 5.7),
        ('QFLARE', 9.8)
    ]
    
    for i, (name, score) in enumerate(systems_comp):
        y = 1.3 - i * 0.25
        # Bar
        bar_length = score * 0.6
        rect = Rectangle((2, y - 0.08), bar_length, 0.16,
                        facecolor='#4CAF50' if name == 'QFLARE' else '#cccccc',
                        edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        # Name
        ax.text(1.8, y, name, fontsize=8, ha='right', family='monospace')
        # Score
        ax.text(2 + bar_length + 0.1, y, f'{score}', fontsize=8, ha='left',
               fontweight='bold', family='monospace')
    
    # Key achievements
    achievements_box = FancyBboxPatch((0.5, 0.1), 9, 0.5,
                                     boxstyle="round,pad=0.05",
                                     edgecolor='black', facecolor='#ffffe0', linewidth=2)
    ax.add_patch(achievements_box)
    ax.text(5, 0.35, '🏆 89.3% avg accuracy | 📈 Linear scalability | 🔒 97.3% Byzantine detection | ⚡ Sub-50ms latency', 
            fontsize=9, fontweight='bold', ha='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/slide_09_experimental_results.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated Slide 9: Experimental Validation Results")

def create_slide_10_key_innovations():
    """SLIDE 10: Key Methodological Innovations"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Key Methodological Innovations', 
            fontsize=24, fontweight='bold', ha='center', family='monospace')
    
    # Innovation 1: Synergistic Security Design
    innov1 = FancyBboxPatch((0.3, 7.2), 4.7, 2.0,
                           boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor='#e6f2ff', linewidth=3)
    ax.add_patch(innov1)
    ax.text(2.65, 8.9, '1. SYNERGISTIC SECURITY', fontsize=11, fontweight='bold', ha='center', family='monospace')
    ax.text(2.65, 8.5, 'Defense-in-Depth Architecture', fontsize=9, ha='center', style='italic', family='monospace')
    
    # Three circles representing synergy
    circles = [
        (1.5, 7.8, 'Quantum\nResistance'),
        (2.65, 7.8, 'Differential\nPrivacy'),
        (3.8, 7.8, 'Byzantine\nTolerance')
    ]
    for x, y, label in circles:
        circle = plt.Circle((x, y), 0.35, facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        for i, line in enumerate(label.split('\n')):
            ax.text(x, y+0.1-i*0.2, line, fontsize=7, ha='center', fontweight='bold', family='monospace')
    
    # Synergy arrows
    for i in range(len(circles)-1):
        x1, y1, _ = circles[i]
        x2, y2, _ = circles[i+1]
        arrow = FancyArrowPatch((x1+0.35, y1), (x2-0.35, y2),
                               arrowstyle='<->', mutation_scale=12, linewidth=1.5, color='green')
        ax.add_patch(arrow)
    
    # Innovation 2: Formal Verification
    innov2 = FancyBboxPatch((5.0, 7.2), 4.7, 2.0,
                           boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor='#fff2e6', linewidth=3)
    ax.add_patch(innov2)
    ax.text(7.35, 8.9, '2. FORMAL VERIFICATION', fontsize=11, fontweight='bold', ha='center', family='monospace')
    ax.text(7.35, 8.5, 'Mathematical Rigor & Proofs', fontsize=9, ha='center', style='italic', family='monospace')
    ax.text(7.35, 8.1, '• Isabelle/HOL', fontsize=8, ha='center', family='monospace')
    ax.text(7.35, 7.7, '• Tamarin Prover', fontsize=8, ha='center', family='monospace')
    ax.text(7.35, 7.3, '• SPIN Model Checker', fontsize=8, ha='center', family='monospace')
    
    # Innovation 3: Production-Ready System
    innov3 = FancyBboxPatch((0.3, 4.7), 4.7, 2.4,
                           boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor='#e6ffe6', linewidth=3)
    ax.add_patch(innov3)
    ax.text(2.65, 6.8, '3. PRODUCTION-READY', fontsize=11, fontweight='bold', ha='center', family='monospace')
    ax.text(2.65, 6.4, 'Enterprise Deployment', fontsize=9, ha='center', style='italic', family='monospace')
    
    features = [
        '✓ Open-source implementation',
        '✓ GDPR/HIPAA compliance',
        '✓ Algorithm agility',
        '✓ Comprehensive docs',
        '✓ A+ security (98/100)'
    ]
    for i, feature in enumerate(features):
        ax.text(2.65, 6.0 - i * 0.3, feature, fontsize=8, ha='center', family='monospace')
    
    # Innovation 4: Research Impact
    innov4 = FancyBboxPatch((5.0, 4.7), 4.7, 2.4,
                           boxstyle="round,pad=0.1",
                           edgecolor='black', facecolor='#ffe6f2', linewidth=3)
    ax.add_patch(innov4)
    ax.text(7.35, 6.8, '4. RESEARCH IMPACT', fontsize=11, fontweight='bold', ha='center', family='monospace')
    ax.text(7.35, 6.4, 'First Complete Solution', fontsize=9, ha='center', style='italic', family='monospace')
    
    impacts = [
        '📜 NIST PQC standards',
        '🔐 (ε,δ)-DP guarantees',
        '🛡️ 33% Byzantine tolerance',
        '⚡ Practical deployment',
        '🏆 Complete integration'
    ]
    for i, impact in enumerate(impacts):
        ax.text(7.35, 6.0 - i * 0.3, impact, fontsize=8, ha='center', family='monospace')
    
    # Central achievement diagram
    ax.text(5, 4.0, 'QFLARE ACHIEVEMENT', fontsize=13, fontweight='bold', ha='center', family='monospace')
    
    # Central circle
    central = plt.Circle((5, 3.0), 0.7, facecolor='#FFD700', edgecolor='black', linewidth=3)
    ax.add_patch(central)
    ax.text(5, 3.0, 'QFLARE\nComplete\nSolution', fontsize=9, ha='center', 
           fontweight='bold', family='monospace')
    
    # Surrounding achievements
    achievements = [
        (2.5, 3.0, 'Post-Quantum\nCrypto'),
        (5, 4.5, 'Differential\nPrivacy'),
        (7.5, 3.0, 'Byzantine\nTolerance'),
        (5, 1.5, 'Production\nReady')
    ]
    
    for x, y, label in achievements:
        circle = plt.Circle((x, y), 0.5, facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        for i, line in enumerate(label.split('\n')):
            ax.text(x, y+0.1-i*0.2, line, fontsize=7, ha='center', family='monospace')
        
        # Connect to central
        arrow = FancyArrowPatch((x, y), (5, 3.0),
                               arrowstyle='-', mutation_scale=15, linewidth=2, 
                               color='gray', linestyle='--')
        ax.add_patch(arrow)
    
    # Bottom banner
    banner = FancyBboxPatch((0.5, 0.2), 9, 0.7,
                           boxstyle="round,pad=0.05",
                           edgecolor='black', facecolor='#ccffcc', linewidth=3)
    ax.add_patch(banner)
    ax.text(5, 0.7, '🌟 UNIQUE CONTRIBUTION 🌟', fontsize=11, fontweight='bold', ha='center', family='monospace')
    ax.text(5, 0.4, 'First system combining NIST PQC + Formal DP + Byzantine Resilience + Enterprise Deployment', 
            fontsize=8, ha='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/slide_10_key_innovations.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated Slide 10: Key Methodological Innovations")

# Generate slides 6-10
if __name__ == "__main__":
    print("\n🎨 Generating QFLARE Methodology Slides 6-10...")
    print("=" * 60)
    
    create_slide_6_byzantine_aggregation()
    create_slide_7_security_analysis()
    create_slide_8_implementation()
    create_slide_9_experimental_results()
    create_slide_10_key_innovations()
    
    print("=" * 60)
    print(f"\n✅ All 10 methodology slides generated in '{output_dir}/' directory")
    print("\n📁 Files created:")
    for i in range(1, 11):
        print(f"   - slide_{i:02d}_*.png")
