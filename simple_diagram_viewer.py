#!/usr/bin/env python3
"""
Simple QFLARE Diagram Viewer - Create a simple HTML index
"""

from pathlib import Path
import webbrowser

def create_simple_diagram_index():
    """Create a simple HTML index for diagrams"""
    
    diagrams_dir = Path("docs/diagrams")
    if not diagrams_dir.exists():
        print("❌ Diagrams directory not found. Run generate_diagrams.py first.")
        return
    
    diagram_files = list(diagrams_dir.glob("*.png"))
    
    if not diagram_files:
        print("❌ No diagram files found. Run generate_diagrams.py first.")
        return
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>QFLARE System Architecture Diagrams</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .diagram {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .diagram h3 {{ color: #34495e; margin-top: 0; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ccc; cursor: pointer; }}
        .description {{ margin-top: 10px; color: #666; font-style: italic; }}
        .download {{ background: #3498db; color: white; padding: 5px 10px; text-decoration: none; border-radius: 3px; }}
        .download:hover {{ background: #2980b9; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 QFLARE System Architecture Diagrams</h1>
        <p style="text-align: center; color: #666;">
            <strong>Generated Diagrams: {len(diagram_files)}</strong> | 
            Comprehensive block diagrams for quantum-safe federated learning
        </p>
"""

    # Diagram descriptions
    descriptions = {
        'qflare_overall_architecture': 'Complete system overview showing all four layers: client devices, edge aggregation, central server, and security monitoring.',
        'qflare_federated_learning_flow': 'Step-by-step process flow of federated learning with differential privacy and post-quantum cryptography.',
        'qflare_pqc_handshake': 'Post-quantum cryptographic handshake sequence using CRYSTALS-Kyber and Dilithium.',
        'qflare_secure_aggregation': 'Secure multi-party computation protocol for privacy-preserving gradient aggregation.',
        'qflare_edge_node_architecture': 'Internal architecture of edge nodes with PQC handling and secure aggregation components.',
        'qflare_threat_model': 'Threat model analysis showing attack surfaces, risks, and mitigation strategies.'
    }
    
    # Add each diagram
    for diagram_file in sorted(diagram_files):
        filename_no_ext = diagram_file.stem
        title = filename_no_ext.replace('qflare_', '').replace('_', ' ').title()
        description = descriptions.get(filename_no_ext, 'System architecture diagram component.')
        
        html_content += f"""
        <div class="diagram">
            <h3>{title}</h3>
            <img src="{diagram_file.relative_to(Path('docs'))}" alt="{title}" onclick="window.open(this.src, '_blank')">
            <div class="description">{description}</div>
            <p><a href="{diagram_file.relative_to(Path('docs'))}" download class="download">📥 Download PNG</a></p>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>"""
    
    # Save HTML file
    html_file = Path("docs/diagram_index.html")
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Simple diagram index created: {html_file}")
    
    # Try to open in browser
    try:
        webbrowser.open(f'file://{html_file.absolute()}')
        print(f"🌐 Opening diagram viewer in browser...")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print(f"📖 Manually open: {html_file.absolute()}")
    
    return html_file

if __name__ == "__main__":
    create_simple_diagram_index()