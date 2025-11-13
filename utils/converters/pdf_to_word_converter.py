"""
QFLARE PDF to Word Converter
Uses pdf2docx library for accurate conversion
"""

import os
import sys
from pathlib import Path

def convert_pdf_to_word():
    print("=" * 60)
    print("QFLARE PDF to Word Converter")
    print("=" * 60)
    print()
    
    # Define paths
    pdf_path = Path("D:/QFLARE_Project_Structure/QFLARE_Final_Report.pdf")
    word_path = Path("D:/QFLARE_Project_Structure/QFLARE_Final_Report.docx")
    
    # Check if PDF exists
    if not pdf_path.exists():
        print(f"❌ ERROR: PDF file not found at: {pdf_path}")
        return False
    
    print(f"✓ PDF file found: {pdf_path}")
    print(f"✓ Output path: {word_path}")
    print()
    
    # Try to import pdf2docx
    try:
        from pdf2docx import Converter
        print("✓ pdf2docx library is installed")
    except ImportError:
        print("⚠️  pdf2docx library not found. Installing now...")
        print()
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "install", "pdf2docx"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to install pdf2docx: {result.stderr}")
            return False
        print("✓ pdf2docx installed successfully")
        print()
        from pdf2docx import Converter
    
    # Convert PDF to Word
    print("🔄 Converting PDF to Word...")
    print("   This may take a few minutes for large documents...")
    print()
    
    try:
        # Create converter
        cv = Converter(str(pdf_path))
        
        # Convert with progress
        cv.convert(str(word_path), start=0, end=None)
        
        # Close converter
        cv.close()
        
        # Check if conversion was successful
        if word_path.exists():
            file_size_mb = word_path.stat().st_size / (1024 * 1024)
            print()
            print("=" * 60)
            print("✅ SUCCESS! Conversion completed!")
            print("=" * 60)
            print()
            print(f"📄 Word document created: {word_path}")
            print(f"📊 File size: {file_size_mb:.2f} MB")
            print()
            print("📌 Note: Please review the document as complex formatting,")
            print("   mathematical equations, and tables may need adjustment.")
            print()
            return True
        else:
            print("❌ Conversion failed - output file not created")
            return False
            
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        print()
        print("Alternative solutions:")
        print("1. Use online converter: https://www.ilovepdf.com/pdf_to_word")
        print("2. Use Adobe Acrobat if available")
        print("3. Try LibreOffice: File > Open > Select PDF")
        return False

def main():
    try:
        success = convert_pdf_to_word()
        
        if success:
            print("Would you like to open the Word document? (Y/N): ", end="")
            try:
                choice = input().strip().upper()
                if choice == 'Y':
                    word_path = Path("D:/QFLARE_Project_Structure/QFLARE_Final_Report.docx")
                    if sys.platform == "win32":
                        os.startfile(str(word_path))
                        print("✓ Opening Word document...")
                    else:
                        print(f"Please manually open: {word_path}")
            except:
                pass
        
        print()
        input("Press Enter to exit...")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Conversion cancelled by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
