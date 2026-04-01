import streamlit as st
from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import re
 
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
if LLAMA_API_KEY is None:
    st.error("API key not found. Make sure LLAMA_API_KEY is set.")
    st.stop()
 
st.set_page_config(
    page_title="ER Triage RAG Assistant",
    page_icon="🩺",
    layout="wide"
)
 
st.title("ER Triage RAG Assistant")
st.write("This app supports ER triage decisions using dense retrieval, grounded generation, and citations.")
 
client_or = OpenAI(
    api_key=LLAMA_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)
 
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
 
@st.cache_resource
def load_vectorstore():
    embedding_model = load_embedding_model()
    vectorstore = Chroma(
        persist_directory="chroma_db_phase2",
        embedding_function=embedding_model
    )
    return vectorstore
 
def run_dense_search_with_scores(query, vectorstore, k=5):
    return vectorstore.similarity_search_with_score(query, k=k)
 
def has_sufficient_context_scored(scored_results, max_best_score=1.2, min_docs=1):
    if not scored_results or len(scored_results) < min_docs:
        return False
    best_score = scored_results[0][1]
    if best_score > max_best_score:
        return False
    return True
 
def build_context_and_ieee_sources(question, vectorstore, k=5):
    scored_results = run_dense_search_with_scores(question, vectorstore, k=k)
    retrieved_docs = [doc for doc, score in scored_results]
 
    context_blocks = []
    source_lines = []
 
    for i, doc in enumerate(retrieved_docs, start=1):
        meta = doc.metadata
 
        topic = meta.get("topic", "Unknown Topic")
        source = meta.get("source", "Unknown Source")
        url = meta.get("url", "No URL")
        section_title = meta.get("section_title", "Unknown Section")
        document_date = meta.get("document_date", "n.d.")
 
        context_block = f"[{i}]\n{doc.page_content}"
        context_blocks.append(context_block)
 
        source_line = (
            f'[{i}] {source}, "{topic}," '
            f'section: "{section_title}," {document_date}. '
            f'[Online]. Available: {url}'
        )
        source_lines.append(source_line)
 
    context_text = "\n\n".join(context_blocks)
    sources_text = "\n".join(source_lines)
 
    return context_text, sources_text, retrieved_docs, scored_results
 
def build_grounded_prompt(question, context_text, sources_text):
    prompt = f"""
You are an Emergency Department triage assistant supporting an ER nurse.
 
## Purpose
Help the nurse determine the appropriate level of care based ONLY on the provided clinical context.
 
## Critical Rules
- Use ONLY the provided context.
- Do NOT use outside medical knowledge.
- Do NOT provide a diagnosis.
- If the context is insufficient, say exactly:
"I don't have enough information about this in my knowledge base."
 
## Urgency Categories (choose EXACTLY ONE)
- Urgent: Patient requires immediate or same-day evaluation in the ER (consider admission or urgent workup).
- Routine: Patient can be managed with non-urgent evaluation (clinic or scheduled appointment).
- Self-care: Patient can be safely managed at home with monitoring and basic care.
 
## Clinical Framing
- Assume the user is an ER nurse making triage decisions.
- Focus on identifying red flags and risk features.
- Be conservative when serious symptoms are present.
- Suggest whether the patient should:
  - remain in ER / be admitted
  - receive urgent evaluation
  - be discharged with follow-up
  - be managed at home
 
## Citation Rules
- Use citation numbers like [1], [2], [3] inline in the Reasoning and Next steps.
- Use only citation numbers that exist in the provided sources.
- Citations must appear in ascending order based on first use.
- The first citation used anywhere in the answer must be [1].
- Do not skip numbers.
- Do not invent sources.
- At the end, include the FULL IEEE-style references corresponding only to the citations actually used, in the same numeric order.
 
## CONTEXT:
{context_text}
 
## AVAILABLE SOURCES (IEEE FORMAT):
{sources_text}
 
## USER QUESTION:
{question}
 
## Output Format (STRICT)
 
Urgency: <Urgent / Routine / Self-care / Insufficient information>
 
Reasoning:
<2–4 concise sentences based ONLY on the provided context. Include inline citations like [1], [2].>
 
Recommendation:
<Clear clinical recommendation: admit / urgent evaluation / outpatient follow-up / home care.>
 
Next steps:
- <Bullet point 1 with citations if relevant>
- <Bullet point 2 with citations if relevant>
- <Bullet point 3 (optional)>
 
Sources:
<List the FULL IEEE-style references corresponding ONLY to the citation numbers used>
"""
    return prompt
 
def build_fallback_judgment_prompt(question):
    prompt = f"""
You are an Emergency Department triage assistant supporting an ER nurse.
 
The retrieved knowledge base did not provide enough reliable context for a grounded answer.
 
Your task is to provide a cautious BEST-EFFORT triage judgment based ONLY on the patient symptom description below.
 
## Critical Rules
- Choose EXACTLY ONE:
  - Urgent
  - Routine
  - Self-care
- Do NOT give a diagnosis.
- Do NOT prescribe medication.
- Be conservative when symptoms sound potentially serious.
- Include EXACTLY this sentence in your reasoning:
"I don't have enough information about this in my knowledge base."
- Clearly state that this is a provisional judgment based only on the reported symptoms.
- Do NOT include citations.
- Sources must be None.
 
## USER QUESTION:
{question}
 
## Output Format (STRICT)
 
Urgency: <Urgent / Routine / Self-care>
 
Reasoning:
<1–3 concise sentences INCLUDING EXACTLY this sentence: "I don't have enough information about this in my knowledge base.">
 
Recommendation:
<Short clinical recommendation for review>
 
Next steps:
- <Bullet 1>
- <Bullet 2>
- <Bullet 3 if needed>
 
Sources:
None
"""
    return prompt.strip()
 
def prepare_rag_input(question, vectorstore, k=3):
    context_text, sources_text, retrieved_docs, scored_results = build_context_and_ieee_sources(
        question=question,
        vectorstore=vectorstore,
        k=k
    )
 
    enough_context = has_sufficient_context_scored(
        scored_results=scored_results,
        max_best_score=1.2,
        min_docs=1
    )
 
    if not enough_context:
        return {
            "question": question,
            "retrieved_docs": retrieved_docs,
            "scored_results": scored_results,
            "context_text": context_text,
            "sources_text": sources_text,
            "prompt": None,
            "has_context": False
        }
 
    final_prompt = build_grounded_prompt(
        question=question,
        context_text=context_text,
        sources_text=sources_text
    )
 
    return {
        "question": question,
        "retrieved_docs": retrieved_docs,
        "scored_results": scored_results,
        "context_text": context_text,
        "sources_text": sources_text,
        "prompt": final_prompt,
        "has_context": True
    }
 
def generate_with_llama(prompt, model_name="meta-llama/llama-3-8b-instruct"):
    response = client_or.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=600
    )
    return response.choices[0].message.content
 
def extract_source_map(sources_text):
    source_map = {}
    for line in sources_text.splitlines():
        match = re.match(r'^\[(\d+)\]\s*(.+)$', line.strip())
        if match:
            source_map[int(match.group(1))] = match.group(2)
    return source_map
 
def normalize_citations_and_sources(answer, sources_text):
    source_map = extract_source_map(sources_text)
 
    if "\nSources:" in answer:
        main_text = answer.split("\nSources:")[0].rstrip()
    else:
        main_text = answer.rstrip()
 
    citation_pattern = re.compile(r'\[(\d+)\]')
    original_citations = [int(x) for x in citation_pattern.findall(main_text)]
 
    ordered_unique = []
    for c in original_citations:
        if c in source_map and c not in ordered_unique:
            ordered_unique.append(c)
 
    remap = {old: new for new, old in enumerate(ordered_unique, start=1)}
 
    def replace_citation(match):
        old_num = int(match.group(1))
        if old_num in remap:
            return f'[{remap[old_num]}]'
        return ''
 
    normalized_text = citation_pattern.sub(replace_citation, main_text)
    normalized_text = re.sub(r'\s+\.', '.', normalized_text)
    normalized_text = re.sub(r'\(\s*\)', '', normalized_text)
    normalized_text = re.sub(r'(?m)[ \t]+$', '', normalized_text)
    normalized_text = re.sub(r'\n{3,}', '\n\n', normalized_text).strip()
 
    used_sources = []
    for old_num in ordered_unique:
        new_num = remap[old_num]
        used_sources.append(f'[{new_num}] {source_map[old_num]}')
 
    if used_sources:
        return normalized_text + "\n\nSources:\n" + "\n".join(used_sources)
    return normalized_text + "\n\nSources:\nNone"
 
def answer_question_with_rag_llama(question, vectorstore, k=3):
    rag_input = prepare_rag_input(question, vectorstore, k=k)
 
    if rag_input["has_context"]:
        final_answer = generate_with_llama(rag_input["prompt"])
        final_answer = normalize_citations_and_sources(final_answer, rag_input["sources_text"])
        rag_input["mode"] = "grounded_rag"
        return final_answer, rag_input
 
    fallback_prompt = build_fallback_judgment_prompt(question)
    final_answer = generate_with_llama(fallback_prompt)
    rag_input["mode"] = "fallback_judgment"
    rag_input["fallback_prompt"] = fallback_prompt
    return final_answer, rag_input
 
vectorstore = load_vectorstore()
 
st.write("Please include key symptoms, duration, severity, and any relevant patient details.")
 
question = st.text_area(
    "Enter the triage scenario",
    placeholder="Example: A 2-month-old infant has a fever of 38.8°C and is unusually sleepy.",
    height=150
)
 
k_value = st.selectbox("Top-k retrieved chunks", [3, 5, 7], index=1)
 
show_debug = st.checkbox("Show retrieved sources and debug information")
 
if st.button("Run triage"):
    if not question.strip():
        st.warning("Please enter a triage scenario.")
    else:
        with st.spinner("Running retrieval and generation..."):
            answer, rag_input = answer_question_with_rag_llama(
                question=question,
                vectorstore=vectorstore,
                k=k_value
            )
 
        st.subheader("Final answer")
        st.text(answer)
 
        if show_debug:
           
            if rag_input.get("mode") == "grounded_rag":
                if rag_input.get("context_text"):
                    st.subheader("Retrieved context")
                    st.text(rag_input["context_text"])
 
                if rag_input.get("sources_text"):
                    st.subheader("Available sources")
                    st.text(rag_input["sources_text"])