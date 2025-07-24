"""
Google Scholar Scraper utilities for the Paper Search & QA System.
Provides functionality to fetch abstracts from Google Scholar using the scholarly package.
"""

import streamlit as st
import datetime

# Try to import scholarly, else show error
try:
    from scholarly import scholarly
    SCHOLARLY_AVAILABLE = True
except ImportError:
    SCHOLARLY_AVAILABLE = False


def display_scholar_scraper_tab():
    """Display the Google Scholar scraper tab"""
    st.markdown("# 🌐 Scholar Abstract Scraper")
    st.info("Enter a paper title or query to fetch the abstract using the scholarly package (Google Scholar API wrapper).\n\n⚠️ This is for research/prototyping. For production, consider SerpAPI.")
    
    # Year selection
    current_year = datetime.datetime.now().year
    years = ["All"] + [str(y) for y in range(current_year, 1999, -1)]
    selected_year = st.selectbox("Select publication year (optional):", years, index=0)
    
    query = st.text_input("Enter paper title or search query:")
    
    if st.button("🔍 Scrape Abstract"):
        if not query.strip():
            st.warning("Please enter a query.")
            return
        
        if not SCHOLARLY_AVAILABLE:
            st.error("The 'scholarly' package is not installed. Please install it with 'pip install scholarly'.")
            return
        
        try:
            search_iter = scholarly.search_pubs(query)
            filtered_result = None
            
            for result in search_iter:
                bib = result.get('bib', {})
                year = str(bib.get('pub_year', bib.get('year', '')))
                if selected_year == "All" or year == selected_year:
                    filtered_result = result
                    break
            
            if not filtered_result:
                st.warning(f"No results found for year {selected_year}." if selected_year != "All" else "No results found.")
                return
            
            bib = filtered_result.get('bib', {})
            title = bib.get('title', '(No title found)')
            authors = bib.get('author', '(No authors found)')
            year = bib.get('pub_year', bib.get('year', ''))
            venue = bib.get('venue', bib.get('journal', ''))
            abstract = bib.get('abstract', '(No abstract found)')
            url = bib.get('url', '')
            num_citations = filtered_result.get('num_citations', None)
            
            # Display formatted result
            st.markdown(f"""
**<span style='font-size:1.3em'>{title}</span>**

**Authors:** {authors}

**Year:** {year if year else 'N/A'}

**Venue:** {venue if venue else 'N/A'}

**Abstract:**
> {abstract}

{f'**URL:** [{url}]({url})' if url else ''}

{f'**Citations:** {num_citations}' if num_citations is not None else ''}
""", unsafe_allow_html=True)
            
            # Raw result as expandable debug
            with st.expander("Show raw result (debug)"):
                st.write(filtered_result)
                
        except Exception as e:
            st.error(f"Error during scholarly search: {e}")


def search_scholar_followup(query: str, year_filter: str = "All"):
    """
    Search Google Scholar for follow-up reading suggestions.
    
    Args:
        query (str): Search query
        year_filter (str): Year filter ("All" or specific year)
    
    Returns:
        dict: Formatted result or None if error
    """
    if not SCHOLARLY_AVAILABLE:
        return None
    
    try:
        search_iter = scholarly.search_pubs(query)
        filtered_result = None
        
        for result in search_iter:
            bib = result.get('bib', {})
            year = str(bib.get('pub_year', bib.get('year', '')))
            if year_filter == "All" or year == year_filter:
                filtered_result = result
                break
        
        if not filtered_result:
            return None
        
        bib = filtered_result.get('bib', {})
        return {
            'title': bib.get('title', '(No title found)'),
            'authors': bib.get('author', '(No authors found)'),
            'year': bib.get('pub_year', bib.get('year', '')),
            'venue': bib.get('venue', bib.get('journal', '')),
            'abstract': bib.get('abstract', '(No abstract found)'),
            'url': bib.get('url', ''),
            'num_citations': filtered_result.get('num_citations', None),
            'raw_result': filtered_result
        }
        
    except Exception as e:
        st.error(f"Error during scholarly search: {e}")
        return None 