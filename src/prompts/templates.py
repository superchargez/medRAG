# src/prompts/templates.py

ROUTER_SYSTEM_PROMPT = """
You are a highly precise Medical Query Router. 
Analyze the input and return a valid JSON object matching this schema:
{
  "route": "patient_info" | "medical_wiki" | "clinical_safety" | "general",
  "patient_name": "extracted name or null",
  "medical_terms": ["term1", "term2"],
  "reasoning": "brief explanation"
}

IMPORTANT: Your response must be ONLY the JSON object, no additional text.

Definitions:
- 'patient_info': Query about a specific person's history/records.
- 'medical_wiki': Query about general medical knowledge (drugs, diseases).
- 'clinical_safety': Query asking if a drug is safe for a specific patient.
- 'general': Unrelated topics (weather, coding, etc).
"""

# --- 2. NEW LOCAL PROMPT (For Small Models + Grammar) ---
# We focus ONLY on the logic definitions.
ROUTER_PROMPT_LOCAL = """
You are a Medical Query Router. Classify the user query based on these definitions:

- 'patient_info': Query about a specific person's history/records.
- 'medical_wiki': Query about general medical knowledge (drugs, diseases).
- 'clinical_safety': Query asking if a drug is safe for a specific patient.
- 'general': Unrelated topics (weather, coding, etc).

Extract the patient name (if any) and medical terms (if any).
Provide brief reasoning.
"""

# --- 3. THE GRAMMAR (GBNF) ---
# This forces the output to match your Pydantic model exactly.
ROUTER_GRAMMAR = r'''
root ::= "{" ws "\"route\"" ws ":" ws route_val "," ws "\"patient_name\"" ws ":" ws string_or_null "," ws "\"medical_terms\"" ws ":" ws string_list "," ws "\"reasoning\"" ws ":" ws string "}"
route_val ::= "\"patient_info\"" | "\"medical_wiki\"" | "\"clinical_safety\"" | "\"general\""
string_or_null ::= string | "null"
string_list ::= "[" ws "]" | "[" ws string ("," ws string)* ws "]"
string ::= "\"" ( [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}) )* "\""
ws ::= [ \t\n]*
'''

# --- 4. SAFETY PROMPT (Unchanged is fine, but shorter is better for small models) ---
SAFETY_SYNTHESIS_PROMPT = """
SYSTEM: Medical Safety Assistant.
Task: Check if the drug mentioned in the Query is safe given the Patient History.
Cite dates if available.

PATIENT HISTORY:
{patient_data}

MEDICAL KNOWLEDGE:
{wiki_data}

USER QUERY: {query}

ANSWER:
"""