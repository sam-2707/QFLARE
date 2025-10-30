#!/usr/bin/env python3
"""
QFLARE Diagram Viewer - Quick viewer for generated architecture diagrams
"""

import os
from pathlib import Path
import subprocess
import webbrowser

def create_diagram_index():
    """Create an HTML index page for all diagrams"""
    
    diagrams_dir = Path("docs/diagrams")
    if not diagrams_dir.exists():
        print("❌ Diagrams directory not found. Run generate_diagrams.py first.")
        return
    
    # Find all PNG files
    diagram_files = list(diagrams_dir.glob("*.png"))
    
    if not diagram_files:
        print("❌ No diagram files found. Run generate_diagrams.py first.")
        return
    
    # Create HTML index
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QFLARE System Architecture Diagrams</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        .diagram-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-top: 30px;
        }
        .diagram-card {
            border: 2px solid #e1e8ed;
            border-radius: 10px;
            padding: 20px;
            background-color: #fafafa;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .diagram-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
            border-color: #3498db;
        }
        .diagram-title {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
            text-transform: capitalize;
        }
        .diagram-image {
            width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid #ddd;
            cursor: pointer;
        }
        .diagram-description {
            margin-top: 10px;
            font-size: 14px;
            color: #666;
            line-height: 1.5;
        }
        .download-btn {
            background-color: #3498db;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            margin-top: 10px;
            text-decoration: none;
            display: inline-block;
            transition: background-color 0.3s ease;
        }
        .download-btn:hover {
            background-color: #2980b9;
        }
        .stats {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
        }
        .modal-content {
            margin: auto;
            display: block;
            width: 90%;
            max-width: 1000px;
            max-height: 90%;
            object-fit: contain;
        }
        .close {
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }
        .close:hover {
            color: #3498db;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 QFLARE System Architecture Diagrams</h1>
        
        <div class="stats">
            <strong>Generated Diagrams: {diagram_count}</strong> | 
            <strong>System Components: 25+</strong> | 
            <strong>Security Layers: 4</strong>
        </div>
        
        <p style="text-align: center; font-size: 16px; color: #666; margin-bottom: 30px;">
            Comprehensive block diagrams and flowcharts for the QFLARE quantum-safe federated learning system.
            Click on any diagram to view in full size.
        </p>
        
        <div class="diagram-grid">
            {diagram_cards}
        </div>
    </div>
    
    <!-- Modal for full-size viewing -->
    <div id="imageModal" class="modal">
        <span class="close">&times;</span>
        <img class="modal-content" id="modalImg">
    </div>
    
    <script>
        // Modal functionality
        const modal = document.getElementById('imageModal');
        const modalImg = document.getElementById('modalImg');
        const images = document.getElementsByClassName('diagram-image');
        const closeBtn = document.getElementsByClassName('close')[0];
        
        for (let i = 0; i < images.length; i++) {
            images[i].onclick = function() {
                modal.style.display = 'block';
                modalImg.src = this.src;
            }
        }
        
        closeBtn.onclick = function() {
            modal.style.display = 'none';
        }
        
        modal.onclick = function(event) {
            if (event.target === modal) {
                modal.style.display = 'none';
            }
        }
    </script>
</body>
</html>
    """
    
    # Diagram descriptions
    descriptions = {
        'qflare_overall_architecture': 'Complete system overview showing all four layers: client devices, edge aggregation, central server, and security monitoring.',
        'qflare_federated_learning_flow': 'Step-by-step process flow of federated learning with differential privacy and post-quantum cryptography integration.',
        'qflare_pqc_handshake': 'Detailed sequence diagram of the post-quantum cryptographic handshake using CRYSTALS-Kyber and Dilithium.',
        'qflare_secure_aggregation': 'Secure multi-party computation protocol for privacy-preserving gradient aggregation.',
        'qflare_edge_node_architecture': 'Internal architecture of edge nodes showing PQC handling, secure aggregation, and monitoring components.',
        'qflare_threat_model': 'Comprehensive threat model analysis showing attack surfaces, risks, and mitigation strategies.'
    }
    
    # Generate diagram cards
    diagram_cards = ""
    for diagram_file in sorted(diagram_files):
        filename_no_ext = diagram_file.stem
        title = filename_no_ext.replace('qflare_', '').replace('_', ' ').title()
        description = descriptions.get(filename_no_ext, 'System architecture diagram component.')
        
        diagram_cards += f'''
            <div class="diagram-card">
                <div class="diagram-title">{title}</div>
                <img src="{diagram_file.relative_to(Path('docs'))}" alt="{title}" class="diagram-image">
                <div class="diagram-description">{description}</div>
                <a href="{diagram_file.relative_to(Path('docs'))}" download class="download-btn">
                    📥 Download PNG
                </a>
            </div>
        '''
    
    # Fill in the template
    html_content = html_content.format(
        diagram_count=len(diagram_files),
        diagram_cards=diagram_cards
    )
    
    # Save HTML file
    html_file = Path("docs/diagram_index.html")
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Diagram index created: {html_file}")
    
    # Try to open in browser
    try:
        webbrowser.open(f'file://{html_file.absolute()}')
        print(f"🌐 Opening diagram viewer in browser...")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print(f"📖 Manually open: {html_file.absolute()}")
    
    return html_file

def show_diagram_summary():
    """Show summary of generated diagrams"""
    
    diagrams_dir = Path("docs/diagrams")
    if not diagrams_dir.exists():
        print("❌ Diagrams directory not found.")
        return
    
    diagram_files = list(diagrams_dir.glob("*.png"))
    
    print("QFLARE Architecture Diagrams Summary")
    print("=" * 50)
    print(f"📊 Total Diagrams: {len(diagram_files)}")
    print(f"📁 Location: {diagrams_dir.absolute()}")
    print()
    
    for i, diagram_file in enumerate(sorted(diagram_files), 1):
        file_size = diagram_file.stat().st_size / 1024  # KB
        title = diagram_file.stem.replace('qflare_', '').replace('_', ' ').title()
        print(f"{i}. {title}")
        print(f"   📄 File: {diagram_file.name}")
        print(f"   💾 Size: {file_size:.1f} KB")
        print()
    
    print("Available Actions:")
    print("• View all: python view_diagrams.py --index")
    print("• Open folder: explorer docs\\diagrams")
    print("• Regenerate: python generate_diagrams.py")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='QFLARE Diagram Viewer')
    parser.add_argument('--index', action='store_true', help='Create and open HTML index')
    parser.add_argument('--summary', action='store_true', help='Show diagram summary')
    
    args = parser.parse_args()
    
    if args.index:
        create_diagram_index()
    elif args.summary:
        show_diagram_summary()
    else:
        # Default: show summary and create index
        show_diagram_summary()
        print("\n" + "=" * 50)
        print("Creating diagram index...")
        create_diagram_index()

if __name__ == "__main__":
    main()