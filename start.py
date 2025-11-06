"""Simple entry point to run MedExplain AI."""
import os
import subprocess
import sys

def check_dependencies():
    """Check if required packages are installed."""
    required = ['streamlit', 'torch', 'torchvision', 'transformers', 'PIL', 'datasets']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n📦 Install them with:")
        print("   pip install streamlit torch torchvision transformers pillow datasets")
        return False
    
    print("✅ All required packages are installed!")
    return True

def main():
    """Run the Streamlit app."""
    print("=" * 60)
    print("🏥 MedExplain AI - Multimodal Clinical Insight Assistant")
    print("=" * 60)
    print()
    
    if not check_dependencies():
        sys.exit(1)
    
    print("\n🚀 Starting Streamlit app...")
    print("📌 The app will open in your browser automatically")
    print("📌 Press Ctrl+C to stop the server\n")
    
    # Run streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error running app: {e}")
        print("\n💡 Try running manually:")
        print("   streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()