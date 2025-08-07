#!/usr/bin/env python3
"""
Setup script for React-Streamlit integration
This script helps you set up the React frontend with your existing Streamlit backend
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step, description):
    """Print a formatted step"""
    print(f"\n{step}. {description}")
    print("-" * 40)

def check_requirements():
    """Check if required tools are installed"""
    print_header("Checking Requirements")
    
    requirements = {
        "python": "Python 3.8+",
        "node": "Node.js 16+",
        "npm": "npm 8+",
        "ollama": "Ollama (for LLM models)"
    }
    
    missing = []
    
    # Check Python
    if sys.version_info >= (3, 8):
        print("✅ Python 3.8+ is installed")
    else:
        print("❌ Python 3.8+ is required")
        missing.append("python")
    
    # Check Node.js
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Node.js is installed")
        else:
            print("❌ Node.js is not installed")
            missing.append("node")
    except FileNotFoundError:
        print("❌ Node.js is not installed")
        missing.append("node")
    
    # Check npm
    try:
        result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ npm is installed")
        else:
            print("❌ npm is not installed")
            missing.append("npm")
    except FileNotFoundError:
        print("❌ npm is not installed")
        missing.append("npm")
    
    # Check Ollama
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed")
        else:
            print("❌ Ollama is not installed")
            missing.append("ollama")
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        missing.append("ollama")
    
    if missing:
        print(f"\n❌ Missing requirements: {', '.join(missing)}")
        print("Please install the missing requirements before continuing.")
        return False
    
    return True

def install_python_dependencies():
    """Install Python dependencies for FastAPI backend"""
    print_step("1", "Installing Python Dependencies")
    
    requirements = [
        "fastapi",
        "uvicorn[standard]",
        "pydantic",
        "requests",
        "python-multipart"
    ]
    
    for req in requirements:
        print(f"Installing {req}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", req], check=True)
            print(f"✅ {req} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {req}")
            return False
    
    return True

def setup_react_frontend():
    """Set up the React frontend"""
    print_step("2", "Setting up React Frontend")
    
    react_dir = Path("react_frontend")
    
    if not react_dir.exists():
        print("Creating React frontend directory...")
        react_dir.mkdir()
    
    # Create package.json if it doesn't exist
    package_json = react_dir / "package.json"
    if not package_json.exists():
        print("Creating package.json...")
        package_data = {
            "name": "material-research-rag-frontend",
            "version": "1.0.0",
            "description": "React frontend for Material Research RAG System",
            "private": True,
            "type": "module",
            "scripts": {
                "dev": "vite",
                "build": "tsc && vite build",
                "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
                "preview": "vite preview"
            },
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "framer-motion": "^10.16.4",
                "class-variance-authority": "^0.7.0",
                "lucide-react": "^0.292.0",
                "axios": "^1.6.0",
                "react-query": "^3.39.3",
                "recharts": "^2.8.0",
                "react-force-graph": "^1.43.4",
                "tailwindcss": "^3.3.5",
                "autoprefixer": "^10.4.16",
                "postcss": "^8.4.31"
            },
            "devDependencies": {
                "@types/react": "^18.2.37",
                "@types/react-dom": "^18.2.15",
                "@typescript-eslint/eslint-plugin": "^6.10.0",
                "@typescript-eslint/parser": "^6.10.0",
                "@vitejs/plugin-react": "^4.1.1",
                "eslint": "^8.53.0",
                "eslint-plugin-react-hooks": "^4.6.0",
                "eslint-plugin-react-refresh": "^0.4.4",
                "typescript": "^5.2.2",
                "vite": "^4.5.0"
            }
        }
        
        with open(package_json, 'w') as f:
            json.dump(package_data, f, indent=2)
        
        print("✅ package.json created")
    
    # Install npm dependencies
    print("Installing npm dependencies...")
    try:
        subprocess.run(["npm", "install"], cwd=react_dir, check=True)
        print("✅ npm dependencies installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install npm dependencies")
        return False
    
    return True

def create_config_files():
    """Create configuration files"""
    print_step("3", "Creating Configuration Files")
    
    # Create vite.config.ts
    vite_config = """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\\/api/, '')
      }
    }
  }
})
"""
    
    with open("react_frontend/vite.config.ts", 'w') as f:
        f.write(vite_config)
    
    print("✅ vite.config.ts created")
    
    # Create tailwind.config.js
    tailwind_config = """/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
"""
    
    with open("react_frontend/tailwind.config.js", 'w') as f:
        f.write(tailwind_config)
    
    print("✅ tailwind.config.js created")
    
    # Create postcss.config.js
    postcss_config = """export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
"""
    
    with open("react_frontend/postcss.config.js", 'w') as f:
        f.write(postcss_config)
    
    print("✅ postcss.config.js created")

def create_startup_scripts():
    """Create startup scripts"""
    print_step("4", "Creating Startup Scripts")
    
    # Create start_backend.py
    backend_script = """#!/usr/bin/env python3
\"\"\"
Start FastAPI backend server
\"\"\"

import subprocess
import sys
import os

def main():
    print("Starting FastAPI backend server...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([sys.executable, "fastapi_backend.py"])
    except KeyboardInterrupt:
        print("\\nServer stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    main()
"""
    
    with open("start_backend.py", 'w') as f:
        f.write(backend_script)
    
    # Make it executable
    os.chmod("start_backend.py", 0o755)
    print("✅ start_backend.py created")
    
    # Create start_frontend.py
    frontend_script = """#!/usr/bin/env python3
\"\"\"
Start React frontend development server
\"\"\"

import subprocess
import sys
import os

def main():
    print("Starting React frontend development server...")
    print("Frontend will be available at: http://localhost:3000")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(["npm", "run", "dev"], cwd="react_frontend")
    except KeyboardInterrupt:
        print("\\nServer stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    main()
"""
    
    with open("start_frontend.py", 'w') as f:
        f.write(frontend_script)
    
    # Make it executable
    os.chmod("start_frontend.py", 0o755)
    print("✅ start_frontend.py created")

def create_readme():
    """Create README with setup instructions"""
    print_step("5", "Creating Documentation")
    
    readme_content = """# Material Research RAG - React Integration

This project integrates a React frontend with your existing Streamlit-based RAG system.

## Architecture

- **FastAPI Backend**: Exposes your existing Streamlit functionality as REST APIs
- **React Frontend**: Modern UI built with React, TypeScript, and Tailwind CSS
- **Streamlit Bridge**: Optional integration for embedding React components in Streamlit

## Quick Start

### 1. Start the Backend
```bash
python start_backend.py
```
The FastAPI server will start on http://localhost:8000

### 2. Start the Frontend
```bash
python start_frontend.py
```
The React development server will start on http://localhost:3000

### 3. Access the Application
- **React Frontend**: http://localhost:3000
- **FastAPI Docs**: http://localhost:8000/docs
- **Original Streamlit**: http://localhost:8501

## Development

### Backend Development
- Edit `fastapi_backend.py` to modify API endpoints
- The backend reuses your existing Streamlit logic and vector store

### Frontend Development
- Edit files in `react_frontend/src/` to modify the UI
- The frontend communicates with the backend via REST APIs

### Adding New Features
1. Add new endpoints to `fastapi_backend.py`
2. Add corresponding API calls in `react_frontend/src/api/client.ts`
3. Create new React components in `react_frontend/src/components/`

## File Structure

```
RHEA_RAG/
├── fastapi_backend.py          # FastAPI server
├── streamlit_react_bridge.py   # Streamlit-React bridge
├── start_backend.py           # Backend startup script
├── start_frontend.py          # Frontend startup script
├── react_frontend/            # React application
│   ├── src/
│   │   ├── api/client.ts      # API client
│   │   ├── components/        # React components
│   │   └── App.tsx           # Main app
│   └── package.json
└── app.py                     # Original Streamlit app
```

## Troubleshooting

### Connection Issues
- Ensure Ollama is running: `ollama serve`
- Check if vector store exists and is properly configured
- Verify FastAPI server is running on port 8000

### Frontend Issues
- Clear npm cache: `npm cache clean --force`
- Reinstall dependencies: `rm -rf node_modules && npm install`

### Backend Issues
- Check Python dependencies: `pip install -r requirements.txt`
- Verify vector store path in `fastapi_backend.py`

## API Endpoints

- `GET /health` - Health check
- `GET /papers` - Get all papers
- `POST /search` - Search papers
- `GET /models` - Get available LLM models
- `GET /correlations` - Get paper correlations
- `GET /network` - Get network data
- `GET /stats` - Get system statistics

## Migration from Streamlit

Your existing Streamlit app (`app.py`) remains unchanged and can still be used. The React frontend provides an alternative interface while maintaining all the same functionality.
"""
    
    with open("REACT_INTEGRATION_README.md", 'w') as f:
        f.write(readme_content)
    
    print("✅ REACT_INTEGRATION_README.md created")

def main():
    """Main setup function"""
    print_header("Material Research RAG - React Integration Setup")
    
    print("This script will help you set up React integration with your existing Streamlit RAG system.")
    print("You'll have both options available:")
    print("1. Original Streamlit interface (unchanged)")
    print("2. New React frontend with FastAPI backend")
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Setup cannot continue. Please install missing requirements.")
        return
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("\n❌ Failed to install Python dependencies.")
        return
    
    # Setup React frontend
    if not setup_react_frontend():
        print("\n❌ Failed to setup React frontend.")
        return
    
    # Create configuration files
    create_config_files()
    
    # Create startup scripts
    create_startup_scripts()
    
    # Create documentation
    create_readme()
    
    print_header("Setup Complete!")
    print("✅ React integration has been set up successfully!")
    print("\nNext steps:")
    print("1. Start the backend: python start_backend.py")
    print("2. Start the frontend: python start_frontend.py")
    print("3. Open http://localhost:3000 in your browser")
    print("\nYour original Streamlit app remains unchanged at http://localhost:8501")
    print("\nFor detailed instructions, see REACT_INTEGRATION_README.md")

if __name__ == "__main__":
    main()
