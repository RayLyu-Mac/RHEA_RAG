# Paper Search & QA System GUI

A Streamlit-based GUI for searching and asking questions about your research papers with parent-child document structure and figure integration.

## Features

- 📚 **Paper Selection**: Choose specific papers or let the LLM select automatically
- 💬 **Question Answering**: Ask questions about materials science concepts
- 🖼️ **Figure Integration**: View extracted figures from papers
- 🔍 **Advanced Search**: Search abstracts only, full text, or both
- 📊 **Statistics**: View paper and figure counts
- 🤖 **Multiple LLM Models**: Choose from different Ollama models

## Installation

1. **Install Streamlit** (if not already installed):
```bash
pip install streamlit
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Make sure Ollama is running** with the required models:
```bash
ollama pull nomic-embed-text:latest
ollama pull qwen2.5:14b
```

## Usage

### Option 1: Run directly with Streamlit
```bash
cd GUI
streamlit run app.py
```

### Option 2: Use the run script
```bash
cd GUI
python run_app.py
```

The app will open in your browser at `http://localhost:8501`

## GUI Layout

### Left Sidebar - Paper Selection
- **Paper Statistics**: Shows total papers, vectorized papers, and figures
- **Selection Mode**: 
  - "Let LLM select automatically" - LLM chooses relevant papers
  - "Select specific papers" - Manually choose papers by folder
- **Upload New Paper**: Upload and process new PDF files (coming soon)

### Main Area - Question Interface
- **Question Input**: Large text area for your questions
- **LLM Model Selection**: Choose from available Ollama models
- **Search Type**: Abstract only, full text + figures, or both
- **Number of Results**: Control how many documents to retrieve

### Right Panel - Figure Demo
- Shows extracted figures from selected papers or search results
- Displays paper name and figure captions
- Updates based on your selections and search results

## Example Questions

Try asking questions like:
- "What is precipitation strengthening?"
- "How does heat treatment affect microstructure?"
- "What are the mechanical properties of refractory alloys?"
- "How do grain boundaries affect material properties?"
- "What experimental methods were used to characterize the materials?"

## File Structure

```
GUI/
├── app.py              # Main Streamlit application
├── run_app.py          # Script to run the app
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Dependencies

The app expects the following files in the parent directory:
- `vectorization_tracker.csv` - Tracking file for processed papers
- `VectorSpace/paper_vector_db_nomic-embed-text_latest_parent_child/` - Vector database
- `extracted_images/` - Directory with extracted figures

## Troubleshooting

### Common Issues

1. **"Vector store not found"**
   - Make sure you've run the vectorization process first
   - Check that the vector store path is correct

2. **"Failed to load LLM model"**
   - Ensure Ollama is running: `ollama serve`
   - Check that the model is installed: `ollama list`

3. **"No figures found"**
   - Ensure the `extracted_images` directory exists
   - Check that figures were extracted during vectorization

4. **"Failed to load paper list"**
   - Make sure `vectorization_tracker.csv` exists
   - Run the paper vectorization process first

### Performance Tips

- Use fewer results (3-5) for faster responses
- Select specific papers instead of searching all papers
- Choose lighter LLM models for faster inference

## Customization

You can customize the app by modifying:
- **Colors**: Edit the CSS in the `st.markdown()` section
- **Models**: Add more models to the selectbox options
- **Layout**: Adjust column ratios and component placement
- **Features**: Add new functionality in the main() function

## Future Enhancements

- [ ] Upload and process new papers directly in the GUI
- [ ] Export search results and answers
- [ ] Advanced filtering options
- [ ] Figure annotation and analysis
- [ ] Batch question processing
- [ ] Search history and favorites 