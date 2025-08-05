"""
Centralized prompt management for the RHEA Paper Search & QA System.
Contains all prompts used throughout the application for consistency and easy maintenance.
"""

from typing import List


class PromptTemplates:
    """Centralized prompt templates for the RHEA RAG system"""
    
    # Question optimization prompt with sub-question breakdown and improved keyword guidance
    QUESTION_OPTIMIZATION = """You are a materials science research expert. Optimize the following question for improved search in a scientific paper database about Refractory High-Entropy Alloys (RHEA).

    Original question: "{original_question}"

    Tasks:
    1. Rewrite the question to be more specific and technical for materials science literature search.
    2. Break down the main question into 2-3 hierarchical sub-questions that clarify the key aspects or steps needed to answer the main question.
    3. Suggest 3-5 relevant keywords that would help retrieve the most pertinent papers. **Avoid generic or overly broad terms such as "HEA", "RHEA", "BCC", or similar alloy system acronyms.** Instead, focus on specific materials science concepts, mechanisms, or properties.

    Format your response as:
    OPTIMIZED QUESTION: [your optimized question]
    SUB-QUESTIONS:
    - [Sub-question 1]
    - [Sub-question 2]
    - [Sub-question 3] (if applicable)
    KEYWORDS: keyword1, keyword2, keyword3, keyword4, keyword5, ...

    Focus on materials science terminology such as: microstructure, precipitation, dislocation, grain boundary, mechanical properties, strengthening mechanisms, phase formation, diffusion, lattice distortion, solid solution, etc.

Response:"""

    # Main answer generation prompt
    ANSWER_GENERATION = """You are a materials science research expert. Based on the following context from scientific papers, answer the user's question comprehensively and accurately.

Context from papers:
{context}

Question: {question}

Instructions:
1. Provide a comprehensive answer based on the context provided
2. Focus on materials science concepts, mechanisms, and relationships
3. If figures are mentioned in the context, reference them appropriately
4. Use technical terminology appropriately
5. Structure your answer clearly with main points and supporting details
6. If specific papers are mentioned, cite them in your response

Answer:"""

    # Summary requirement suffix
    SUMMARY_REQUIREMENT = "\n\n**IMPORTANT**: After providing the detailed answer above, conclude with a concise summary in exactly 3-5 sentences that captures the key points of your response."

    # Research gap analysis prompt
    RESEARCH_GAP_ANALYSIS = """You are an expert research assistant specialized in materials science, tasked with analyzing a set of retrieved research papers to identify research gaps. The papers focus on [insert specific topic, e.g., "refractory high-entropy alloys (RHEAs)"] and have been retrieved from a vector database based on their relevance to the topic. Your goal is to synthesize key findings, methodologies, and limitations from these papers and identify underexplored areas, contradictions, or open questions that could guide future research. Follow these steps:

1. **Summarize Key Findings**: Provide a concise summary of the main results, trends, or conclusions from the retrieved papers, focusing on [specific aspect, e.g., "mechanical properties, microstructure, or dislocation mechanisms in RHEAs"].
2. **Identify Methodologies**: Highlight the primary experimental, computational, or theoretical approaches used in these papers, noting any recurring techniques or tools.
3. **Analyze Limitations**: Point out explicitly stated limitations or challenges in the papers, such as incomplete datasets, specific alloy compositions not studied, or unexplored conditions (e.g., temperature, pressure).
4. **Detect Contradictions**: Identify any conflicting findings or interpretations across the papers, such as differing conclusions about [specific aspect, e.g., "the role of lattice distortion in RHEA strength"].
5. **Suggest Research Gaps**: Based on the summaries, limitations, and contradictions, propose specific research gaps or unanswered questions. Focus on areas that are underexplored, novel, or have potential for significant impact in [field, e.g., "RHEA design for aerospace applications"]. Provide at least 3 concrete suggestions, each with a brief justification.
6. **Prioritize Feasibility**: For each suggested gap, briefly assess its feasibility based on current methodologies or technologies mentioned in the papers, and suggest a potential approach to address it (e.g., experimental, simulation-based, or theoretical).

**Input Context**: You have access to [number, e.g., "10"] retrieved research papers or document chunks stored in a vector database, with summaries and metadata including titles, abstracts, and key sections (e.g., results, conclusions). If figures or tables are available, consider their data (e.g., mechanical properties, phase diagrams) in your analysis.

**Output Format**:
- **Summary of Key Findings**: [Brief summary, 3-4 sentences]
- **Methodologies Used**: [List key methods, 2-3 sentences]
- **Limitations Identified**: [List limitations, 2-3 sentences]
- **Contradictions Noted**: [Describe contradictions or lack thereof, 2-3 sentences]
- **Research Gaps and Suggestions**:
- Gap 1: [Description and justification]
    - Feasibility: [Brief assessment and suggested approach]
- Gap 2: [Description and justification]
    - Feasibility: [Brief assessment and suggested approach]
- Gap 3: [Description and justification]
    - Feasibility: [Brief assessment and suggested approach]

**Constraints**:
- Be concise, precise, and avoid speculation beyond the provided data.
- Focus on gaps relevant to [specific topic, e.g., "RHEAs"] and avoid overly broad suggestions.
- If insufficient data is available to identify gaps, state this clearly and suggest ways to refine the retrieval (e.g., adjust query terms, include more recent papers).
- Use technical language appropriate for materials science but ensure clarity for a researcher audience.

**Example Context (if needed)**: The papers discuss topics like [e.g., "dislocation dynamics, phase stability, or high-temperature performance of RHEAs"], with some including experimental data (e.g., tensile strength tests) and others using simulations (e.g., molecular dynamics).

Please analyze the provided papers and generate a detailed research gap analysis following the structure above."""

    # Follow-up question prompt
    FOLLOW_UP_QUESTION = """You are a materials science research expert. Based on the previous answer, selected context from previous follow-ups, and the new follow-up question, provide a comprehensive response.

Previous Answer:
{previous_answer}{additional_context}

New Follow-up Question: {follow_up_question}

Instructions:
1. Use the previous answer and any selected context as background information
2. Address the new follow-up question specifically
3. Build upon the information from the previous answer and context
4. Provide additional insights or clarifications as needed
5. Maintain consistency with the previous responses
6. If the follow-up question requires new information not covered in the previous answers, acknowledge this and suggest how to obtain that information

Response:"""

    # Meeting notes Q&A prompt
    MEETING_NOTES_QA = """Based on the following meeting notes, please answer the question. Provide specific references to the meeting notes when possible.

Meeting Notes:
{context}

Question: {question}

Answer:"""

    # Paper grouping prompt
    PAPER_GROUPING = """Given the following abstract and the user's question:

Question: {user_question}

Abstract:
{abstract}

What is the main mechanism, type, or conclusion discussed in this paper relevant to the question? Summarize in one sentence."""

    # RAG flowchart generation prompt
    RAG_FLOWCHART = """Generate a Graphviz DOT flowchart representing a Retrieval-Augmented Generation (RAG) pipeline. Include nodes for Query, Retrieve Documents, Generate Response, and Display, with directed edges connecting them in sequence. Use clear, concise DOT syntax suitable for rendering with the graphviz Python library. The following papers are selected as context: {paper_titles}. Only output the DOT code, no explanation."""

    # LLM grouping refinement prompt
    LLM_GROUPING_REFINEMENT = """Given the following list of papers and their initial groupings, refine the groups to be more scientifically meaningful. Consider grouping by crystal lattice type, composition, or other relevant scientific criteria. Output a new table with columns: Paper Title, Refined Group.

Paper Title | Group
{table_str}

Refined Table:"""

    # Scholar summary prompt
    SCHOLAR_SUMMARY = """You are a scientific research assistant. Given the following abstracts from Google Scholar search results, summarize the main findings and trends in 5 sentences. Present the summary as a numbered list.

ABSTRACTS:
{abstracts}

SUMMARY (5 sentences as a list):"""


def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with the provided arguments"""
    return template.format(**kwargs)


def add_summary_requirement(prompt: str) -> str:
    """Add summary requirement to a prompt"""
    return prompt + PromptTemplates.SUMMARY_REQUIREMENT


# Convenience functions for common prompt operations
def get_question_optimization_prompt(original_question: str) -> str:
    """Get the question optimization prompt"""
    return format_prompt(PromptTemplates.QUESTION_OPTIMIZATION, original_question=original_question)


def get_answer_generation_prompt(context: str, question: str, summarize: bool = False) -> str:
    """Get the answer generation prompt"""
    prompt = format_prompt(PromptTemplates.ANSWER_GENERATION, context=context, question=question)
    if summarize:
        prompt = add_summary_requirement(prompt)
    return prompt


def get_research_gap_prompt(abstracts: List[str], summarize: bool = False) -> str:
    """Get the research gap analysis prompt"""
    prompt = PromptTemplates.RESEARCH_GAP_ANALYSIS + "\n\n" + "\n\n".join(abstracts)
    if summarize:
        prompt = add_summary_requirement(prompt)
    return prompt


def get_follow_up_prompt(previous_answer: str, additional_context: str, follow_up_question: str, summarize: bool = False) -> str:
    """Get the follow-up question prompt"""
    prompt = format_prompt(
        PromptTemplates.FOLLOW_UP_QUESTION,
        previous_answer=previous_answer,
        additional_context=additional_context,
        follow_up_question=follow_up_question
    )
    if summarize:
        prompt = add_summary_requirement(prompt)
    return prompt


def get_meeting_notes_prompt(context: str, question: str) -> str:
    """Get the meeting notes Q&A prompt"""
    return format_prompt(PromptTemplates.MEETING_NOTES_QA, context=context, question=question)


def get_paper_grouping_prompt(user_question: str, abstract: str) -> str:
    """Get the paper grouping prompt"""
    return format_prompt(PromptTemplates.PAPER_GROUPING, user_question=user_question, abstract=abstract)


def get_rag_flowchart_prompt(paper_titles: str) -> str:
    """Get the RAG flowchart generation prompt"""
    return format_prompt(PromptTemplates.RAG_FLOWCHART, paper_titles=paper_titles)


def get_llm_grouping_refinement_prompt(table_str: str) -> str:
    """Get the LLM grouping refinement prompt"""
    return format_prompt(PromptTemplates.LLM_GROUPING_REFINEMENT, table_str=table_str)


def get_scholar_summary_prompt(abstracts: List[str]) -> str:
    """Get the scholar summary prompt"""
    return format_prompt(PromptTemplates.SCHOLAR_SUMMARY, abstracts="\n\n".join(abstracts)) 