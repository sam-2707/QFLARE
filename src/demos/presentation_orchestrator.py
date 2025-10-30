#!/usr/bin/env python3
"""
QFLARE Presentation Orchestrator
Master script to coordinate all presentation materials and demonstrations
"""

import os
import sys
import time
import subprocess
from pathlib import Path

class QFLAREPresentationOrchestrator:
    def __init__(self):
        self.presentation_files = []
        self.demo_ready = False
        
    def check_dependencies(self):
        """Check if all required dependencies are available"""
        print("🔍 Checking presentation dependencies...")
        
        required_packages = ['matplotlib', 'numpy', 'fastapi', 'uvicorn']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"   ❌ {package} - Missing")
        
        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        
        print("✅ All dependencies satisfied!")
        return True
    
    def prepare_presentation_materials(self):
        """Generate all presentation materials"""
        print("\n📋 Preparing QFLARE presentation materials...")
        
        # Step 1: Generate competitive analysis
        print("\n1️⃣  Running competitive analysis...")
        try:
            import competitive_analysis_tool
            competitive_analysis_tool.run_complete_analysis()
            print("✅ Competitive analysis complete")
        except Exception as e:
            print(f"❌ Competitive analysis failed: {e}")
        
        # Step 2: Generate visual diagrams
        print("\n2️⃣  Generating presentation visuals...")
        try:
            import generate_presentation_visuals
            generate_presentation_visuals.generate_all_visuals()
            print("✅ Visual diagrams generated")
        except Exception as e:
            print(f"❌ Visual generation failed: {e}")
    
    def start_demo_backend(self):
        """Start the QFLARE demo backend"""
        print("\n🚀 Starting QFLARE demo backend...")
        
        backend_path = Path("backend/professional_backend.py")
        if not backend_path.exists():
            print(f"❌ Backend file not found: {backend_path}")
            return False
        
        try:
            # Start backend in background
            backend_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", 
                "backend.professional_backend:app", 
                "--host", "0.0.0.0", 
                "--port", "8000",
                "--reload"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            time.sleep(3)  # Give it time to start
            
            # Check if process is running
            if backend_process.poll() is None:
                print("✅ Backend started on http://localhost:8000")
                return True
            else:
                print("❌ Backend failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Backend startup failed: {e}")
            return False
    
    def run_security_demo(self):
        """Run the interactive security demonstration"""
        print("\n🔐 Starting QFLARE security demonstration...")
        
        try:
            import demo_security_showcase
            demo = demo_security_showcase.QFLARESecurityDemo()
            demo.run_full_demo()
            print("✅ Security demo completed successfully")
            return True
        except Exception as e:
            print(f"❌ Security demo failed: {e}")
            return False
    
    def create_presentation_checklist(self):
        """Create a checklist for the presentation"""
        checklist = """
🎯 QFLARE PRESENTATION CHECKLIST
===============================

PRE-PRESENTATION SETUP (30 minutes before):
------------------------------------------
□ Run presentation orchestrator: python presentation_orchestrator.py
□ Verify backend is running on localhost:8000
□ Test frontend on localhost:3000  
□ Check all visual files are generated
□ Review competitive analysis reports
□ Practice security demo flow

PRESENTATION FLOW (45-60 minutes):
---------------------------------

OPENING (5 minutes):
□ Introduce QFLARE vision & team
□ Outline agenda: Problem → Solution → Demo → Proof

PROBLEM STATEMENT (8 minutes):
□ Current federated learning security gaps
□ Quantum computing threat timeline
□ Enterprise security requirements
□ Show quantum_threat_timeline.png

QFLARE SOLUTION (12 minutes):
□ Post-quantum cryptography overview
□ Multi-layer security architecture
□ Show qflare_key_architecture.png
□ Hardware security integration
□ Real-time threat monitoring

LIVE DEMONSTRATION (15 minutes):
□ Start security demo: python demo_security_showcase.py
□ Show key generation process
□ Demonstrate attack simulations
□ Display real-time monitoring
□ Navigate live application dashboard

COMPETITIVE PROOF (10 minutes):
□ Present security comparison charts
□ Show qflare_security_comparison.png
□ Review competitive analysis scores
□ Highlight quantum readiness advantage
□ Reference executive summary

Q&A SESSION (5-10 minutes):
□ Address technical questions
□ Discuss implementation timeline
□ Share contact information

PRESENTATION MATERIALS CHECKLIST:
---------------------------------
□ QFLARE_SECURITY_PRESENTATION.md (Presentation guide)
□ qflare_security_analysis.txt (Detailed security analysis)
□ qflare_performance_analysis.txt (Performance benchmarks)
□ qflare_quantum_readiness.txt (Quantum timeline)
□ qflare_executive_summary.txt (Key talking points)
□ qflare_key_architecture.png (Architecture diagram)
□ qflare_security_comparison.png (Competitive charts)
□ quantum_threat_timeline.png (Timeline visualization)

DEMO ENVIRONMENT CHECKLIST:
---------------------------
□ Backend running: curl http://localhost:8000/health
□ Frontend accessible: http://localhost:3000
□ Login credentials working: admin/admin123, user/user123
□ WebSocket connections stable
□ Real-time metrics updating
□ No console errors

BACKUP PLANS:
------------
□ Screenshots of working demo available
□ Pre-recorded video demo ready
□ Static presentation slides prepared
□ Competitive analysis printouts
□ Contact information cards

KEY TALKING POINTS:
------------------
□ "Only quantum-ready federated learning platform"
□ "Zero migration effort - secure from day one"
□ "Enterprise-grade performance with military security"
□ "20+ year security guarantee against quantum threats"
□ "Hardware-backed cryptographic protection"

SUCCESS METRICS:
---------------
□ Audience understands quantum threat urgency
□ QFLARE's unique value proposition is clear
□ Technical credibility established
□ Interest in pilot/partnership generated
□ Follow-up meetings scheduled

POST-PRESENTATION:
-----------------
□ Collect contact information
□ Schedule follow-up meetings
□ Send presentation materials
□ Provide trial access information
□ Document feedback and questions
"""
        
        with open('presentation_checklist.txt', 'w') as f:
            f.write(checklist)
        
        print("✅ Presentation checklist saved to presentation_checklist.txt")
        return checklist
    
    def validate_demo_environment(self):
        """Validate that the demo environment is ready"""
        print("\n🔍 Validating demo environment...")
        
        checks = [
            ("Backend health endpoint", "http://localhost:8000/health"),
            ("Authentication endpoint", "http://localhost:8000/api/auth/login"),
            ("Client data endpoint", "http://localhost:8000/api/clients"),
            ("Training status endpoint", "http://localhost:8000/api/training/status")
        ]
        
        import requests
        
        for check_name, url in checks:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code < 400:
                    print(f"   ✅ {check_name}")
                else:
                    print(f"   ⚠️  {check_name} - Status: {response.status_code}")
            except requests.exceptions.RequestException:
                print(f"   ❌ {check_name} - Connection failed")
        
        # Check for generated files
        required_files = [
            'qflare_key_architecture.png',
            'qflare_security_comparison.png', 
            'quantum_threat_timeline.png',
            'qflare_executive_summary.txt',
            'presentation_checklist.txt'
        ]
        
        print("\n📁 Checking presentation files...")
        for file in required_files:
            if Path(file).exists():
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} - Missing")
    
    def run_full_presentation_prep(self):
        """Run complete presentation preparation"""
        print("🎯 QFLARE PRESENTATION PREPARATION")
        print("="*50)
        
        # Step 1: Check dependencies
        if not self.check_dependencies():
            print("\n❌ Cannot proceed - missing dependencies")
            return False
        
        # Step 2: Prepare materials
        self.prepare_presentation_materials()
        
        # Step 3: Start demo environment
        print("\n🚀 Setting up demo environment...")
        backend_ready = self.start_demo_backend()
        
        # Step 4: Run security demo test
        print("\n🔐 Testing security demonstration...")
        demo_ready = self.run_security_demo()
        
        # Step 5: Create checklist
        print("\n📋 Creating presentation checklist...")
        self.create_presentation_checklist()
        
        # Step 6: Final validation
        time.sleep(2)  # Give backend time to fully start
        self.validate_demo_environment()
        
        # Summary
        print("\n" + "="*50)
        if backend_ready and demo_ready:
            print("🎉 PRESENTATION PREPARATION COMPLETE!")
            print("\nYour presentation environment is ready:")
            print("   🌐 Backend: http://localhost:8000")
            print("   🖥️  Frontend: http://localhost:3000") 
            print("   🔐 Demo: python demo_security_showcase.py")
            print("   📊 Analysis: Open qflare_executive_summary.txt")
            print("   ✅ Checklist: presentation_checklist.txt")
            print("\n🎯 Ready to showcase QFLARE's quantum-ready security!")
        else:
            print("⚠️  Presentation preparation completed with some issues.")
            print("   Check the logs above and resolve any failed components.")
            print("   You can still use the generated materials for your presentation.")
        
        return True

def main():
    """Main orchestrator function"""
    orchestrator = QFLAREPresentationOrchestrator()
    orchestrator.run_full_presentation_prep()

if __name__ == "__main__":
    main()