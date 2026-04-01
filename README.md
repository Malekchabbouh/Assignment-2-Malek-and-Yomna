# Assignment 2 - Emergency Room Triage LLM System

## Overview
This project implements and evaluates a healthcare triage system using Large Language Models (LLMs) to help nurses in the emergency room. The system classifies patient inputs into three urgency levels:

- Self-care  
- Routine  
- Urgent  

Two versions of the system are developed and evaluated:

- **A1 (Baseline):** Prompt-based classification  
- **A2 (Enhanced):** Retrieval-Augmented Generation (RAG) with improved grounding  

The goal is to analyze performance, reliability, and safety in medical triage scenarios.

---

## Project Structure
 A2_Malek_Yomna/
│
├── Part A-C/
│ ├── LLM_Assignment2_Malek_Yomna.ipynb
│ ├── a1_results.txt
│ ├── a2_rag_results.txt
│ ├── testcases_assignment_2.json
│ ├── chroma_db_phase2/
│ ├── cleaned_docs_phase2/
│ └── chunked_docs_phase2.json
│
├── Streamlit_app/
│ ├── app.py
│ ├── chroma_db_phase2/
│ └── requirements.txt
│
├── DEMO.mp4
└── README.md


---

## Features

- Triage classification into 3 urgency levels  
- Retrieval-Augmented Generation (RAG) integration  
- Evaluation on structured test cases  
- Confusion matrix analysis  
- Comparison between A1 and A2 systems  
- Interactive Streamlit application  

---

## Results Summary

- A1 performs well on clear cases but shows **hallucination and input distortion**  
- A2 improves grounding but introduces **retrieval mismatches and occasional fabricated details**  
- Most errors occur in **borderline cases (Routine vs Urgent)**  
- Both models tend to **over-triage**, which is safer than under-triage in healthcare settings  

### Example Observations

- **TC11 (A1):** Introduced "persistent cough" not present in input  
- **TC18 (A1):** Assumed worsening symptoms and added treatment ("IV fluids")  
- **TC09 / TC15 (A1):** Over-diagnosis beyond input  
- **TC04 (A2):** Irrelevant retrieved information (vomiting)  
- **TC10 (A2):** Provided treatment instructions  
- **TC11** Clear illustration of how A2 enhanced over A1  

---

## Demo

A demonstration of the system is provided in: DEMO.mp4


---

## How to Run the Application

### 1. Install dependencies
```bash
pip install -r Streamlit_app/requirements.txt

#### 2. Run the app
streamlit run Streamlit_app/app.py

Deployment

The application is deployed using Streamlit Cloud and provides an interactive interface for testing triage predictions.
