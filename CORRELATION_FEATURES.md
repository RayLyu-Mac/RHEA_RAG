# Paper Correlation Management System

## Overview

The Paper Correlation Management System is a new feature that allows you to store, manage, and visualize correlations between research papers. This system enhances the paper retrieval process by providing contextual relationships between papers, making the search results more correlated and meaningful.

## Features

### 1. 📊 Paper Correlation Management Tab

**Location**: Second tab in the main interface

**Features**:
- **Overview**: View all research topics and their correlations
- **Add Correlation**: Create new correlations between papers
- **View Correlations**: Browse all correlations in table format
- **Network Visualization**: Interactive network diagrams
- **Export Data**: Export correlations to CSV or JSON

### 2. 📈 Network Analysis Tab

**Location**: Third tab in the main interface

**Features**:
- Interactive network visualization using Plotly
- Network statistics (nodes, edges, average degree)
- Topic-based filtering
- Correlation details table

### 3. 🔍 Enhanced Search Integration

**Integration**: Automatically included in search results

**How it works**:
- When you search for papers, the system automatically finds correlations for the papers in your search results
- Correlation information is added to the LLM context, providing richer, more connected answers
- The LLM can now understand relationships between papers and provide more comprehensive responses

## SSS (Solid Solution Strengthening) Correlations

The system comes pre-loaded with correlations for the SSS research area:

### Shear Modulus Mismatch
- **Papers**: zhou2025strategies, zheng2022development, dou2024modulus
- **Effect**: Induces lattice distortion, improving yield and ductility

### Atomic Size Mismatch - Room Temperature
- **Papers**: wang2025dual, wang2025hf, fang2024composition, chen2025low
- **Effect**: Induces lattice distortion, improving yield but reducing ductility

### Atomic Size Mismatch - High Temperature
- **Papers**: ko2025boron
- **Effect**: Induces lattice distortion, improving yield but reducing ductility

### Atomic Size Mismatch - RT with Comparable Ductility
- **Papers**: ji2022effect
- **Effect**: Induces lattice distortion, improving yield while maintaining comparable ductility

### Solid Solution Treatment
- **Positive Effect**: liu2025effect - Heat treatment dissolves secondary phase, increasing solute concentration
- **Negative Effect**: fang2024composition - High temperature/long time leads to secondary phase formation

## How to Use

### Adding New Correlations

1. Go to the "📊 Paper Correlations" tab
2. Select "Add Correlation" from the sidebar
3. Choose a topic (e.g., "Solid Solution Strengthening (SSS)")
4. Select source and target papers
5. Define relationship type and description
6. Set correlation strength (0.0 to 1.0)
7. Add optional evidence
8. Click "Add Correlation"

### Viewing Network Visualizations

1. Go to the "📈 Network Analysis" tab
2. Select a topic for analysis
3. View the interactive network diagram
4. Check network statistics
5. Browse correlation details in the table below

### Enhanced Search

1. Ask questions in the "🔍 Search & QA" tab as usual
2. The system automatically includes correlation context
3. Get more comprehensive answers that understand paper relationships

## Data Structure

### PaperCorrelation
```python
@dataclass
class PaperCorrelation:
    source: str              # Source paper name
    target: str              # Target paper name
    relationship_type: str   # Type of relationship
    description: str         # Description of the relationship
    strength: float          # Correlation strength (0.0-1.0)
    evidence: Optional[str]  # Supporting evidence
    date_added: str          # When correlation was added
```

### ResearchTopic
```python
@dataclass
class ResearchTopic:
    name: str                    # Topic name
    description: str             # Topic description
    papers: List[str]            # Papers in this topic
    correlations: List[PaperCorrelation]  # Correlations
    parent_topic: Optional[str]  # Parent topic (for hierarchy)
    sub_topics: List[str]        # Sub-topics
```

## Scalability

The system is designed to be easily scalable:

1. **Add New Topics**: Use `manager.add_topic()` to create new research areas
2. **Add Papers**: Use `manager.add_paper_to_topic()` to add papers to topics
3. **Add Correlations**: Use `manager.add_correlation()` to create relationships
4. **Hierarchical Organization**: Support for parent-child topic relationships

## File Storage

- **Data File**: `paper_correlations.json` (automatically created)
- **Format**: JSON with topics, papers, and correlations
- **Backup**: Data is automatically saved after each modification

## Integration Benefits

### For Researchers
- **Better Context**: Understand relationships between papers
- **Enhanced Search**: Get more relevant and connected answers
- **Visual Analysis**: See research networks and patterns
- **Knowledge Discovery**: Identify gaps and connections in research

### For the System
- **Richer Context**: LLM gets more information about paper relationships
- **Better Answers**: More comprehensive and connected responses
- **Scalable**: Easy to add new research areas and correlations
- **Persistent**: Data is saved and persists between sessions

## Example Usage

### Adding a New Research Area
```python
manager = PaperCorrelationManager()
manager.add_topic("Precipitation Hardening", "Research on precipitation strengthening mechanisms")
manager.add_paper_to_topic("Precipitation Hardening", "paper1")
manager.add_paper_to_topic("Precipitation Hardening", "paper2")
```

### Creating a Correlation
```python
correlation = PaperCorrelation(
    source="paper1",
    target="paper2",
    relationship_type="precipitation_mechanism",
    description="Both papers study similar precipitation mechanisms",
    strength=0.8
)
manager.add_correlation("Precipitation Hardening", correlation)
```

### Enhanced Search Context
When you search for papers, the system automatically includes correlation context like:
```
📊 **Paper Correlations Found:**
• **paper1** → **paper2**: precipitation_mechanism - Both papers study similar precipitation mechanisms (Strength: 0.8)
```

This context helps the LLM provide more comprehensive and connected answers about the relationships between the papers in your search results. 