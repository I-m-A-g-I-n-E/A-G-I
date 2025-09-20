#!/usr/bin/env python3
"""
Deploy notebooks and code to Deepnote using the API.
Requires DEEPNOTE_API_KEY in environment or .env file.
"""

import os
import sys
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional
import requests
from datetime import datetime

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed, reading from environment only")

class DeepnoteDeployer:
    """Deploy notebooks and files to Deepnote workspace."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('DEEPNOTE_API_KEY')
        if not self.api_key:
            raise ValueError("DEEPNOTE_API_KEY not found in environment or .env file")
        
        self.base_url = "https://api.deepnote.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.project_id = None
        self.workspace_id = None
    
    def create_or_get_project(self, name: str = "48-Manifold") -> str:
        """Create a new project or get existing one."""
        # List existing projects
        response = requests.get(
            f"{self.base_url}/projects",
            headers=self.headers
        )
        
        if response.status_code == 200:
            projects = response.json().get('projects', [])
            for project in projects:
                if project.get('name') == name:
                    self.project_id = project['id']
                    print(f"✅ Found existing project: {name}")
                    return self.project_id
        
        # Create new project
        data = {
            "name": name,
            "description": "{wtf²B•2^4*3} Manifold - Revolutionary computational framework"
        }
        
        response = requests.post(
            f"{self.base_url}/projects",
            headers=self.headers,
            json=data
        )
        
        if response.status_code == 201:
            self.project_id = response.json()['id']
            print(f"✅ Created new project: {name}")
            return self.project_id
        else:
            print(f"❌ Failed to create project: {response.text}")
            return None
    
    def configure_environment(self):
        """Configure project environment for CUDA."""
        if not self.project_id:
            print("❌ No project ID available")
            return False
        
        # Environment configuration
        env_config = {
            "hardware": {
                "gpu": True,  # Request GPU
                "gpu_type": "T4",  # or "V100", "A100"
                "cpu": 4,
                "ram": 16
            },
            "docker_image": "deepnote/python:3.10-cuda",
            "init_script": """
#!/bin/bash
# Install requirements
pip install -r /work/deepnote/requirements.txt

# Set Python path
export PYTHONPATH=/work:$PYTHONPATH

# Run setup
chmod +x /work/deepnote/setup_deepnote.sh
/work/deepnote/setup_deepnote.sh
"""
        }
        
        response = requests.patch(
            f"{self.base_url}/projects/{self.project_id}/environment",
            headers=self.headers,
            json=env_config
        )
        
        if response.status_code == 200:
            print("✅ Environment configured for CUDA")
            return True
        else:
            print(f"⚠️ Environment configuration failed: {response.text}")
            return False
    
    def upload_file(self, local_path: Path, remote_path: str) -> bool:
        """Upload a single file to Deepnote."""
        if not self.project_id:
            print("❌ No project ID available")
            return False
        
        # Read file content
        if local_path.suffix == '.ipynb':
            with open(local_path, 'r') as f:
                content = f.read()
        else:
            with open(local_path, 'rb') as f:
                content = base64.b64encode(f.read()).decode('utf-8')
        
        data = {
            "path": remote_path,
            "content": content,
            "type": "notebook" if local_path.suffix == '.ipynb' else "file"
        }
        
        response = requests.post(
            f"{self.base_url}/projects/{self.project_id}/files",
            headers=self.headers,
            json=data
        )
        
        if response.status_code in [200, 201]:
            print(f"  ✅ Uploaded: {remote_path}")
            return True
        else:
            print(f"  ❌ Failed to upload {remote_path}: {response.text}")
            return False
    
    def upload_directory(self, local_dir: Path, remote_dir: str = "/work/deepnote"):
        """Upload entire directory structure to Deepnote."""
        if not local_dir.exists():
            print(f"❌ Directory not found: {local_dir}")
            return
        
        uploaded = 0
        failed = 0
        
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                # Skip certain files
                if file_path.name.startswith('.'):
                    continue
                if file_path.suffix in ['.pyc', '.pyo', '.pyd']:
                    continue
                if '__pycache__' in str(file_path):
                    continue
                
                # Calculate remote path
                relative_path = file_path.relative_to(local_dir)
                remote_path = f"{remote_dir}/{relative_path}".replace('\\', '/')
                
                # Upload file
                if self.upload_file(file_path, remote_path):
                    uploaded += 1
                else:
                    failed += 1
        
        print(f"\n📊 Upload Summary: {uploaded} succeeded, {failed} failed")
    
    def create_starter_notebook(self):
        """Create a starter notebook with initialization code."""
        starter_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# 🚀 {wtf²B•2^4*3} Manifold - Getting Started\n",
                        "\n",
                        "Welcome to the 48-Manifold Deepnote workspace!\n",
                        "\n",
                        "This notebook will help you get started with the environment."
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Initialize environment\n",
                        "import sys\n",
                        "sys.path.append('/work')\n",
                        "\n",
                        "from deepnote.cuda_devices import cuda_manager, get_device\n",
                        "\n",
                        "device = get_device()\n",
                        "print(f'Device: {device}')\n",
                        "\n",
                        "if device.type == 'cuda':\n",
                        "    info = cuda_manager.get_device_info()\n",
                        "    print(f\"GPU: {info.get('name')}\")\n",
                        "    print(f\"Memory: {info.get('total_memory_gb'):.1f} GB\")\n",
                        "    print(f\"CUDA: {info.get('cuda_version')}\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Run smoke tests\n",
                        "!python /work/deepnote/smoke_tests.py"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 📚 Available Notebooks\n",
                        "\n",
                        "1. [Memory Performance Benchmarks](./notebooks/1_Memory_Performance.ipynb)\n",
                        "2. [Molar - Protein Folding](./notebooks/2_Molar_Protein.ipynb)\n",
                        "3. [Motor - Hand Visualization](./notebooks/3_Motor_Hand.ipynb)\n",
                        "4. [Manifold - Interactive Playground](./notebooks/4_Manifold_Playground.ipynb)\n",
                        "\n",
                        "## 📖 Documentation\n",
                        "\n",
                        "- [AI Developer Guide](./AI_DEVELOPER_GUIDE.md)\n",
                        "- [Requirements](./requirements.txt)\n",
                        "- [Setup Script](./setup_deepnote.sh)"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Write to temp file
        starter_path = Path("/tmp/0_Getting_Started.ipynb")
        with open(starter_path, 'w') as f:
            json.dump(starter_content, f, indent=2)
        
        # Upload
        self.upload_file(starter_path, "/work/deepnote/0_Getting_Started.ipynb")
    
    def deploy(self):
        """Main deployment function."""
        print("\n" + "="*60)
        print("🚀 DEPLOYING TO DEEPNOTE")
        print("="*60 + "\n")
        
        # Create or get project
        if not self.create_or_get_project():
            print("❌ Failed to create/get project")
            return False
        
        # Configure environment
        self.configure_environment()
        
        # Upload deepnote directory
        deepnote_dir = Path(__file__).parent
        print(f"\n📁 Uploading from: {deepnote_dir}")
        self.upload_directory(deepnote_dir)
        
        # Create starter notebook
        print("\n📓 Creating starter notebook...")
        self.create_starter_notebook()
        
        # Generate access link
        print("\n" + "="*60)
        print("✅ DEPLOYMENT COMPLETE!")
        print("="*60)
        print(f"\n🔗 Access your project at:")
        print(f"   https://deepnote.com/project/{self.project_id}")
        print(f"\n📝 Next steps:")
        print(f"   1. Open the project in Deepnote")
        print(f"   2. Run the Getting Started notebook")
        print(f"   3. Explore the four main notebooks")
        print(f"\n🎉 Happy coding!")
        
        return True

def main():
    """Main entry point."""
    try:
        # Check for API key
        api_key = os.getenv('DEEPNOTE_API_KEY')
        if not api_key:
            print("❌ DEEPNOTE_API_KEY not found!")
            print("\nPlease set it in one of these ways:")
            print("1. Add to .env file: DEEPNOTE_API_KEY=your-key-here")
            print("2. Export in shell: export DEEPNOTE_API_KEY='your-key-here'")
            print("3. Pass as argument: python deploy_to_deepnote.py your-key-here")
            
            if len(sys.argv) > 1:
                api_key = sys.argv[1]
                print(f"\n✅ Using API key from command line argument")
            else:
                sys.exit(1)
        
        # Deploy
        deployer = DeepnoteDeployer(api_key)
        success = deployer.deploy()
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()