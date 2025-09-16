#!/usr/bin/env python3
"""
QFLARE Quantum System Launcher
One-click startup for dashboard and testing
"""

import subprocess
import sys
import time
import os
import webbrowser
from pathlib import Path

def print_banner():
    """Print startup banner"""
    print("""
    ===============================================================
                                                               
        QFLARE QUANTUM KEY EXCHANGE SYSTEM                   
                                                               
        Advanced Quantum-Safe Cryptography Dashboard              
        & Testing Suite                                            
                                                               
    ===============================================================
    """)

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        "fastapi", "uvicorn", "jinja2", "python-multipart"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(packages):
    """Install missing dependencies"""
    print(f"Installing missing packages: {', '.join(packages)}")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", *packages
        ])
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False

def start_dashboard():
    """Start the quantum dashboard"""
    print("Starting Quantum Dashboard...")
    
    try:
        # Start dashboard in background
        dashboard_process = subprocess.Popen([
            sys.executable, "quantum_dashboard.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for startup
        time.sleep(3)
        
        # Check if process is still running
        if dashboard_process.poll() is None:
            print("Dashboard started successfully!")
            print("Dashboard URL: http://localhost:8002")
            return dashboard_process
        else:
            stdout, stderr = dashboard_process.communicate()
            print(f"Dashboard failed to start:")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return None
    
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        return None

def run_tests():
    """Run the quantum system tests"""
    print("\nRunning Quantum System Tests...")
    
    try:
        result = subprocess.run([
            sys.executable, "test_quantum_system.py"
        ], capture_output=False, text=True)
        
        return result.returncode == 0
    
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def main():
    """Main launcher function"""
    print_banner()
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    # Check if we're in the right directory
    required_files = ["quantum_dashboard.py", "test_quantum_system.py"]
    missing_files = [f for f in required_files if not (current_dir / f).exists()]
    
    if missing_files:
        print(f"Missing required files: {', '.join(missing_files)}")
        print("Please run this script from the QFLARE project directory.")
        return
    
    # Check dependencies
    print("Checking dependencies...")
    missing_deps = check_dependencies()
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        install_choice = input("Install missing dependencies? (y/n): ").lower().strip()
        
        if install_choice in ['y', 'yes']:
            if not install_dependencies(missing_deps):
                print("Failed to install dependencies. Exiting.")
                return
        else:
            print("Cannot proceed without dependencies. Exiting.")
            return
    else:
        print("All dependencies satisfied!")
    
    # Show menu
    print("\nWhat would you like to do?")
    print("1. Start Dashboard Only")
    print("2. Start Dashboard + Run Tests")
    print("3. Run Tests Only")
    print("4. Open Browser to Dashboard")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        # Start dashboard only
        dashboard_process = start_dashboard()
        if dashboard_process:
            print("\n✅ Dashboard is running!")
            print("🌐 Visit: http://localhost:8002")
            print("📱 The dashboard includes:")
            print("   • Real-time quantum key exchange visualization")
            print("   • Device registration and management")
            print("   • Security threat simulation")
            print("   • Performance monitoring")
            print("   • Interactive testing controls")
            print("\nPress Ctrl+C to stop the dashboard...")
            
            try:
                dashboard_process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping dashboard...")
                dashboard_process.terminate()
                dashboard_process.wait()
                print("✅ Dashboard stopped.")
    
    elif choice == "2":
        # Start dashboard and run tests
        dashboard_process = start_dashboard()
        if dashboard_process:
            print("\n⏳ Waiting for dashboard to fully initialize...")
            time.sleep(5)
            
            # Run tests
            test_success = run_tests()
            
            if test_success:
                print("\n🎉 All tests completed successfully!")
            else:
                print("\n⚠️  Some tests failed. Check the output above.")
            
            print("\n🌐 Dashboard is still running at: http://localhost:8002")
            print("Press Ctrl+C to stop the dashboard...")
            
            try:
                dashboard_process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping dashboard...")
                dashboard_process.terminate()
                dashboard_process.wait()
                print("✅ Dashboard stopped.")
    
    elif choice == "3":
        # Run tests only
        print("\n⚠️  Note: Tests require the dashboard to be running on port 8002")
        proceed = input("Continue anyway? (y/n): ").lower().strip()
        
        if proceed in ['y', 'yes']:
            run_tests()
        else:
            print("Tests cancelled.")
    
    elif choice == "4":
        # Open browser
        print("🌐 Opening browser to dashboard...")
        webbrowser.open("http://localhost:8002")
        print("✅ Browser opened! If dashboard isn't running, start it first with option 1.")
    
    elif choice == "5":
        print("👋 Goodbye!")
    
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Script interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your Python environment and try again.")