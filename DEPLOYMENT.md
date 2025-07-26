# Deployment Guide for RHEA Paper Search & QA System

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Streamlit Cloud** account (or other deployment platform)
3. **Vector store data** uploaded to your repository

## Quick Deployment Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Optional Dependencies

The app will work without these, but they enhance functionality:

- **Graphviz**: For RAG flowchart visualization
  ```bash
  pip install graphviz
  ```

- **Ollama**: For local LLM support (optional)
  - Install from: https://ollama.ai/
  - Or use online LLM services

### 3. Deploy to Streamlit Cloud

1. **Push your code to GitHub** (including the vector store data)
2. **Connect to Streamlit Cloud**:
   - Go to https://share.streamlit.io/
   - Connect your GitHub repository
   - Set the main file path to: `RHEA_RAG/app_modular.py`

### 4. Environment Variables (Optional)

If using online LLM services, set these in Streamlit Cloud:

```bash
OPENAI_API_KEY=your_openai_key_here
```

## Deployment Platforms

### Streamlit Cloud (Recommended)
- Free tier available
- Automatic deployment from GitHub
- Easy to set up

### Heroku
- Add `setup.sh` and `Procfile` for Heroku deployment
- Set buildpacks for Python

### Docker
- Create a `Dockerfile` for containerized deployment
- Deploy to any container platform

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'graphviz'**
   - Solution: The app will work without graphviz, but install it for full functionality
   - Run: `pip install graphviz`

2. **Vector store not found**
   - Ensure your vector store data is included in the repository
   - Check the path in `vector_store.py`

3. **LLM not loading**
   - The app works in "Paper Management & Note-Taking" mode without LLM
   - Install Ollama or configure online LLM services

### Performance Tips

1. **Reduce memory usage**:
   - Limit the number of papers loaded
   - Use smaller LLM models

2. **Speed up loading**:
   - Pre-process and cache vector embeddings
   - Use CDN for static assets

## File Structure for Deployment

```
RHEA_RAG/
├── app_modular.py          # Main Streamlit app
├── requirements.txt        # Python dependencies
├── DEPLOYMENT.md          # This file
├── utils/                 # Utility modules
├── VectorSpace/           # Vector store data (include in repo)
└── Papers/               # Paper files (optional for deployment)
```

## Features Available Without LLM

When LLM is not available, the app provides:
- ✅ Paper search and retrieval
- ✅ Paper preview and figure viewing
- ✅ Note-taking and management
- ✅ Google Scholar integration
- ✅ Vector store operations

## Features Requiring LLM

These features need an LLM to work:
- 🤖 Question optimization
- 🤖 AI-powered answer generation
- 🤖 Research gap analysis
- 🤖 Paper grouping and classification
- 🤖 RAG flowchart generation

## Support

For deployment issues:
1. Check the Streamlit Cloud logs
2. Verify all dependencies are installed
3. Ensure vector store data is accessible
4. Test locally before deploying 