"""
Notes utilities for the Paper Search & QA System.
Handles meeting notes management, including adding, viewing, and Q&A functionality.
"""

import streamlit as st
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from langchain.schema import Document


def load_meeting_notes() -> List[Dict]:
    """Load meeting notes from JSON file"""
    notes_file = "meeting_notes.json"
    if os.path.exists(notes_file):
        try:
            with open(notes_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert dictionary to list if the data is stored as a dictionary
                if isinstance(data, dict):
                    return list(data.values())
                elif isinstance(data, list):
                    return data
                else:
                    st.error("Invalid meeting notes format")
                    return []
        except Exception as e:
            st.error(f"Failed to load meeting notes: {e}")
            return []
    return []


def save_meeting_notes(notes: List[Dict]):
    """Save meeting notes to JSON file"""
    notes_file = "meeting_notes.json"
    try:
        # Convert list back to dictionary format for storage
        notes_dict = {}
        for note in notes:
            if 'id' in note:
                notes_dict[note['id']] = note
        
        with open(notes_file, 'w', encoding='utf-8') as f:
            json.dump(notes_dict, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Failed to save meeting notes: {e}")


def add_note_to_vectorstore(note_data: Dict, vectorstore):
    """Add a meeting note to the vector store"""
    try:
        # Handle both old and new note formats
        section = note_data.get('section', 'General Discussion')
        papers_discussed = note_data.get('papers_discussed', [])
        tags = note_data.get('tags', [])
        
        # For old notes, check if papers and tags are in metadata
        if not papers_discussed and 'metadata' in note_data:
            papers_discussed = note_data['metadata'].get('papers_discussed', [])
        if not tags and 'metadata' in note_data:
            tags = note_data['metadata'].get('tags', [])
        
        # Prepare metadata
        metadata = {
            'content_type': 'meeting_notes',
            'title': note_data['title'],
            'meeting_date': note_data['date'],
            'section': section,
            'papers_discussed': ', '.join(papers_discussed),
            'tags': ', '.join(tags),
            'document_type': 'meeting_notes'
        }
        
        # Create document
        doc = Document(
            page_content=note_data['content'],
            metadata=metadata
        )
        
        # Add to vector store
        vectorstore.add_documents([doc])
        return True
    except Exception as e:
        st.error(f"Failed to add note to vector store: {e}")
        return False


def display_notes_section():
    """Display the meeting notes section in sidebar"""
    st.markdown("### 📝 Meeting Notes")
    
    # Load existing notes
    notes = load_meeting_notes()
    
    # Add new note form
    with st.expander("➕ Add New Note", expanded=False):
        st.markdown("**Add a new meeting note:**")
        
        title = st.text_input("Meeting Title:", key="new_note_title")
        date = st.date_input("Meeting Date:", key="new_note_date")
        section = st.selectbox(
            "Section:",
            ["General Discussion", "Literature Review", "Methodology", "Results", "Future Work", "Other"],
            key="new_note_section"
        )
        
        content = st.text_area(
            "Meeting Content:",
            height=150,
            placeholder="Enter the meeting notes content...",
            key="new_note_content"
        )
        
        papers_discussed = st.text_input(
            "Papers Discussed (comma-separated):",
            placeholder="paper1.pdf, paper2.pdf",
            key="new_note_papers"
        )
        
        tags = st.text_input(
            "Tags (comma-separated):",
            placeholder="precipitation, microstructure, mechanical",
            key="new_note_tags"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Note", key="save_note"):
                if title and content:
                    # Parse papers and tags
                    papers_list = [p.strip() for p in papers_discussed.split(',') if p.strip()] if papers_discussed else []
                    tags_list = [t.strip() for t in tags.split(',') if t.strip()] if tags else []
                    
                    # Generate a unique ID based on date and count
                    date_str = date.strftime('%Y-%m-%d')
                    existing_notes_for_date = [n for n in notes if n['date'] == date_str]
                    note_count = len(existing_notes_for_date)
                    note_id = f"{date_str}_{note_count}"
                    
                    new_note = {
                        'id': note_id,
                        'title': title,
                        'date': date_str,
                        'section': section,
                        'content': content,
                        'papers_discussed': papers_list,
                        'tags': tags_list,
                        'created_at': datetime.now().isoformat()
                    }
                    
                    notes.append(new_note)
                    save_meeting_notes(notes)
                    st.success("Note saved successfully!")
                    st.rerun()
                else:
                    st.warning("Please fill in title and content")
        
        with col2:
            if st.button("🔄 Clear Form", key="clear_note_form"):
                st.rerun()
    
    # Display existing notes
    if notes:
        st.markdown("**📋 Existing Notes:**")
        
        # Group notes by date
        notes_by_date = {}
        for note in notes:
            date_key = note['date']
            if date_key not in notes_by_date:
                notes_by_date[date_key] = []
            notes_by_date[date_key].append(note)
        
        # Display notes grouped by date
        for date_key in sorted(notes_by_date.keys(), reverse=True):
            st.markdown(f"**📅 {date_key}**")
            
            for note in notes_by_date[date_key]:
                # Handle both old and new note formats
                section = note.get('section', 'General Discussion')
                papers_discussed = note.get('papers_discussed', [])
                tags = note.get('tags', [])
                
                # For old notes, check if papers and tags are in metadata
                if not papers_discussed and 'metadata' in note:
                    papers_discussed = note['metadata'].get('papers_discussed', [])
                if not tags and 'metadata' in note:
                    tags = note['metadata'].get('tags', [])
                
                with st.expander(f"📝 {note['title']} ({section})", expanded=False):
                    st.markdown(f"**Section:** {section}")
                    st.markdown(f"**Content:**")
                    st.markdown(note['content'])
                    
                    if papers_discussed:
                        st.markdown(f"**Papers Discussed:** {', '.join(papers_discussed)}")
                    
                    if tags:
                        st.markdown(f"**Tags:** {', '.join(tags)}")
                    
                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("🗑️ Delete", key=f"delete_{note['id']}"):
                            notes.remove(note)
                            save_meeting_notes(notes)
                            st.success("Note deleted!")
                            st.rerun()
                    
                    with col2:
                        if st.button("📤 Sync to Vector Store", key=f"sync_{note['id']}"):
                            if st.session_state.vectorstore:
                                if add_note_to_vectorstore(note, st.session_state.vectorstore):
                                    st.success("Note synced to vector store!")
                                else:
                                    st.error("Failed to sync note")
                            else:
                                st.error("Vector store not loaded")
                    
                    with col3:
                        if st.button("❓ Ask Question", key=f"qa_{note['id']}"):
                            st.session_state.selected_notes_for_qa = [note['id']]
                            st.session_state.qa_mode = True
                            st.rerun()
            
            st.divider()
        
        # Q&A interface
        if st.session_state.get('selected_notes_for_qa'):
            st.markdown("### 💬 Ask Questions About Notes")
            
            # Note selection
            selected_note_ids = st.multiselect(
                "Select notes to query:",
                options=[note['id'] for note in notes],
                default=st.session_state.get('selected_notes_for_qa', []),
                format_func=lambda x: next((note['title'] for note in notes if note['id'] == x), f"Note {x}")
            )
            
            if selected_note_ids:
                # Question input
                question = st.text_area(
                    "Your question about the selected notes:",
                    placeholder="What was discussed about precipitation strengthening?",
                    height=100
                )
                
                if st.button("🔍 Ask Question", key="ask_notes_qa"):
                    if question.strip():
                        # Check if LLM is ready
                        if st.session_state.get('llm'):
                            answer = ask_question_about_notes(question, selected_note_ids, notes, st.session_state.llm)
                            if answer:
                                st.markdown("**Answer:**")
                                st.markdown(answer)
                        else:
                            st.error("❌ LLM not loaded. Please load the LLM first to use this feature.")
                    else:
                        st.warning("Please enter a question")
            
            # Clear selection button
            if st.button("👁️ View All Notes", key="view_all_notes"):
                st.session_state.selected_notes_for_qa = []
                st.rerun()
    
    else:
        st.info("No meeting notes yet. Add your first note above!")


def ask_question_about_notes(question: str, note_ids: List[str], notes: List[Dict], llm) -> Optional[str]:
    """Ask a question about selected meeting notes using LLM"""
    try:
        # Get selected notes
        selected_notes = [note for note in notes if note['id'] in note_ids]
        if not selected_notes:
            return "No notes selected."
        
        # Prepare context
        context_parts = []
        for note in selected_notes:
            # Handle both old and new note formats
            section = note.get('section', 'General Discussion')
            papers_discussed = note.get('papers_discussed', [])
            tags = note.get('tags', [])
            
            # For old notes, check if papers and tags are in metadata
            if not papers_discussed and 'metadata' in note:
                papers_discussed = note['metadata'].get('papers_discussed', [])
            if not tags and 'metadata' in note:
                tags = note['metadata'].get('tags', [])
            
            context_parts.append(f"""
Meeting: {note['title']}
Date: {note['date']}
Section: {section}
Content: {note['content']}
Papers Discussed: {', '.join(papers_discussed)}
Tags: {', '.join(tags)}
---""")
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following meeting notes, please answer the question. Provide specific references to the meeting notes when possible.

Meeting Notes:
{context}

Question: {question}

Answer:"""
        
        # Get LLM response
        response = llm.invoke(prompt)
        return response
        
    except Exception as e:
        st.error(f"Error asking question about notes: {e}")
        return None


def sync_selected_notes_to_vectorstore(selected_note_ids: List[str], notes: List[Dict], vectorstore) -> bool:
    """Sync selected notes to vector store"""
    try:
        selected_notes = [note for note in notes if note['id'] in selected_note_ids]
        success_count = 0
        
        for note in selected_notes:
            if add_note_to_vectorstore(note, vectorstore):
                success_count += 1
        
        if success_count == len(selected_notes):
            st.success(f"Successfully synced {success_count} notes to vector store!")
            return True
        else:
            st.warning(f"Synced {success_count}/{len(selected_notes)} notes. Some failed.")
            return False
            
    except Exception as e:
        st.error(f"Error syncing notes to vector store: {e}")
        return False


def display_selective_sync_interface(notes: List[Dict]):
    """Display interface for selective note syncing"""
    if not notes:
        st.info("No notes available for syncing.")
        return
    
    st.markdown("### 📤 Selective Sync to Vector Store")
    st.markdown("Select which notes to sync to the vector store:")
    
    # Note selection
    selected_note_ids = st.multiselect(
        "Choose notes to sync:",
        options=[note['id'] for note in notes],
        format_func=lambda x: next((note['title'] for note in notes if note['id'] == x), f"Note {x}")
    )
    
    if selected_note_ids:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Sync Selected", key="sync_selected"):
                if st.session_state.vectorstore:
                    sync_selected_notes_to_vectorstore(selected_note_ids, notes, st.session_state.vectorstore)
                else:
                    st.error("Vector store not loaded")
        
        with col2:
            if st.button("📤 Sync All", key="sync_all"):
                if st.session_state.vectorstore:
                    all_note_ids = [note['id'] for note in notes]
                    sync_selected_notes_to_vectorstore(all_note_ids, notes, st.session_state.vectorstore)
                else:
                    st.error("Vector store not loaded")
    else:
        st.info("Select notes above to sync them to the vector store.") 