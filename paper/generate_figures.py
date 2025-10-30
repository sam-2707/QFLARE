#!/usr/bin/env python3
"""
QFLARE Conference Paper - Figure Generation Script
Generates charts and diagrams for the conference paper
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style for professional figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def generate_accuracy_comparison():
    """Generate accuracy comparison chart"""
    datasets = ['MNIST', 'CIFAR-10', 'IMDB', 'AGNews', 'Average']
    baseline = [99.2, 91.5, 89.3, 92.1, 93.0]
    dp_fedavg = [96.8, 87.3, 84.1, 87.8, 89.0]
    krum = [98.1, 86.7, 82.9, 86.3, 88.5]
    qflare = [98.7, 89.2, 86.7, 89.4, 91.0]
    
    x = np.arange(len(datasets))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - 1.5*width, baseline, width, label='Baseline (Insecure)', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, dp_fedavg, width, label='DP-FedAvg', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, krum, width, label='Krum', color='#f39c12', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, qflare, width, label='QFLARE', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Comparison Across Federated Learning Systems', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(75, 100)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('paper/figures/accuracy_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_performance_overhead():
    """Generate performance overhead breakdown"""
    components = ['Local\nTraining', 'Crypto\nOperations', 'Network\nTransfer', 'Byzantine\nDetection', 'Total\nOverhead']
    baseline_times = [85, 5, 45, 5, 140]
    qflare_times = [89, 59, 67, 23, 238]
    
    x = np.arange(len(components))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width/2, baseline_times, width, label='Baseline', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, qflare_times, width, label='QFLARE', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('System Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time per Round (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Overhead Breakdown by Component', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    
    # Add value labels and overhead percentages
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        
        ax.annotate(f'{height1}ms',
                   xy=(bar1.get_x() + bar1.get_width() / 2, height1),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
        
        ax.annotate(f'{height2}ms',
                   xy=(bar2.get_x() + bar2.get_width() / 2, height2),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
        
        # Add overhead percentage
        if height1 > 0:
            overhead = ((height2 - height1) / height1) * 100
            if overhead > 10:  # Only show significant overheads
                ax.annotate(f'+{overhead:.0f}%',
                           xy=(bar2.get_x() + bar2.get_width() / 2, height2),
                           xytext=(0, 15),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8,
                           color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('paper/figures/performance_overhead.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/performance_overhead.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_scalability_analysis():
    """Generate scalability analysis chart"""
    participants = [10, 50, 100, 250, 500, 1000]
    baseline_latency = [142, 148, 156, 168, 182, 195]
    qflare_latency = [156, 198, 243, 275, 318, 341]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Absolute latency comparison
    ax1.plot(participants, baseline_latency, 'o-', label='Baseline', linewidth=2, markersize=6, color='#3498db')
    ax1.plot(participants, qflare_latency, 's-', label='QFLARE', linewidth=2, markersize=6, color='#e74c3c')
    ax1.set_xlabel('Number of Participants', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Latency per Round (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Scalability: Absolute Latency', fontsize=13, fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Overhead ratio
    overhead_ratio = [q/b for q, b in zip(qflare_latency, baseline_latency)]
    ax2.plot(participants, overhead_ratio, 'D-', linewidth=2, markersize=6, color='#f39c12')
    ax2.set_xlabel('Number of Participants', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Overhead Ratio (QFLARE/Baseline)', fontsize=12, fontweight='bold')
    ax2.set_title('Scalability: Overhead Ratio', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='2x Overhead')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    
    # Add annotations for key points
    ax2.annotate(f'{overhead_ratio[-1]:.2f}x @ 1000 devices',
                xy=(participants[-1], overhead_ratio[-1]),
                xytext=(-50, 20),
                textcoords='offset points',
                ha='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig('paper/figures/scalability_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/scalability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_security_radar():
    """Generate security capabilities radar chart"""
    categories = ['Quantum\nResistance', 'Privacy\nProtection', 'Byzantine\nTolerance', 
                 'Storage\nSecurity', 'Authentication', 'Audit\nCapability']
    
    # Scores out of 10
    systems = {
        'FedAvg': [0, 1, 0, 1, 3, 2],
        'DP-FedAvg': [0, 6, 0, 1, 3, 2],
        'Krum': [0, 1, 8, 1, 3, 2],
        'QFLARE': [10, 9, 8, 10, 10, 9]
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    for idx, (system, scores) in enumerate(systems.items()):
        ax = axes[idx]
        
        # Add categories to the plot
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        scores += scores[:1]  # Complete the circle
        angles += angles[:1]
        
        ax.plot(angles, scores, 'o-', linewidth=2, label=system, color=colors[idx])
        ax.fill(angles, scores, alpha=0.25, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=8)
        ax.grid(True)
        ax.set_title(f'{system}', fontsize=12, fontweight='bold', pad=20)
        
        # Highlight QFLARE
        if system == 'QFLARE':
            ax.set_facecolor('#ffe6e6')
    
    plt.tight_layout()
    plt.savefig('paper/figures/security_radar.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/security_radar.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_architecture_diagram():
    """Generate detailed architecture diagram with Python"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define colors
    colors = {
        'server': '#ff9999',
        'device': '#66b3ff', 
        'storage': '#99ff99',
        'crypto': '#ffcc99',
        'network': '#ff99cc'
    }
    
    # Central Server
    server_rect = Rectangle((6, 7), 2, 1.5, facecolor=colors['server'], edgecolor='black', linewidth=2)
    ax.add_patch(server_rect)
    ax.text(7, 7.75, 'Central Server\n(Quantum-Safe)', ha='center', va='center', fontweight='bold')
    
    # Edge Devices
    device_positions = [(2, 4), (4, 2), (7, 1), (10, 2), (12, 4)]
    for i, (x, y) in enumerate(device_positions):
        device_rect = Rectangle((x-0.75, y-0.5), 1.5, 1, facecolor=colors['device'], edgecolor='black')
        ax.add_patch(device_rect)
        ax.text(x, y, f'Device {i+1}', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Secure Storage
    storage_rect = Rectangle((10, 7), 2.5, 1.5, facecolor=colors['storage'], edgecolor='black', linewidth=2)
    ax.add_patch(storage_rect)
    ax.text(11.25, 7.75, 'Secure Storage\nPostgreSQL + KMS', ha='center', va='center', fontweight='bold')
    
    # Privacy Engine
    privacy_rect = Rectangle((2, 7), 2.5, 1.5, facecolor=colors['crypto'], edgecolor='black', linewidth=2)
    ax.add_patch(privacy_rect)
    ax.text(3.25, 7.75, 'Privacy Engine\nDifferential Privacy', ha='center', va='center', fontweight='bold')
    
    # Key Management
    key_rect = Rectangle((6, 10), 2, 1, facecolor=colors['crypto'], edgecolor='black', linewidth=2)
    ax.add_patch(key_rect)
    ax.text(7, 10.5, 'Key Management\nCRYSTALS-Kyber/Dilithium', ha='center', va='center', fontweight='bold')
    
    # Draw connections
    connections = [
        # Devices to server
        ((2, 4.5), (6.2, 7.2)),
        ((4, 2.5), (6.4, 7.0)),
        ((7, 1.5), (7, 7)),
        ((10, 2.5), (7.6, 7.0)),
        ((12, 4.5), (7.8, 7.2)),
        # Server to storage
        ((8, 7.75), (10, 7.75)),
        # Privacy to server
        ((4.5, 7.75), (6, 7.75)),
        # Key management to server
        ((7, 10), (7, 8.5))
    ]
    
    for (x1, y1), (x2, y2) in connections:
        ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.1, head_length=0.1, 
                fc='black', ec='black', alpha=0.7)
    
    # Add security annotations
    ax.text(1, 9, 'Post-Quantum\nCryptography', ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    ax.text(13, 9, 'Envelope\nEncryption', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    ax.text(7, 5.5, 'Byzantine Fault\nTolerance', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('QFLARE System Architecture Overview', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('paper/figures/qflare_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/qflare_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all figures for the conference paper"""
    import os
    
    # Create figures directory
    os.makedirs('paper/figures', exist_ok=True)
    
    print("🎨 Generating conference paper figures...")
    
    try:
        print("📊 Generating accuracy comparison chart...")
        generate_accuracy_comparison()
        
        print("⏱️ Generating performance overhead analysis...")
        generate_performance_overhead()
        
        print("📈 Generating scalability analysis...")
        generate_scalability_analysis()
        
        print("🛡️ Generating security radar charts...")
        generate_security_radar()
        
        print("🏗️ Generating architecture diagram...")
        generate_architecture_diagram()
        
        print("✅ All figures generated successfully!")
        print("📁 Figures saved in: paper/figures/")
        
    except Exception as e:
        print(f"❌ Error generating figures: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()