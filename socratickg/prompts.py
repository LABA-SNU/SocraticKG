"""Prompt templates used by the SocraticKG pipeline.

- WHOLE_DOC_QA_PROMPT: 5W1H-guided QA generation from a full document.
- EXTRACTION_PROMPT:   Triple extraction from a single QA pair.
- RESOLUTION_PROMPT:   Entity / relation canonicalization via LLM synonym resolution.
"""


WHOLE_DOC_QA_PROMPT = """## ROLE
You are a **Comprehensive Knowledge Archivist** who converts the [Full Document] into detailed, document-grounded QA pairs.

## OBJECTIVE
Extract as many meaningful Question-Answer pairs as possible from the document.
Use the 5W1H perspectives (Who, What, When, Where, Why, How) **as analytical lenses** to help you identify and expand potential questions, but do NOT restrict yourself to producing only 5W1H-type questions.
Your goal is to maximize informational coverage, capturing every explicit fact, relation, event, definition, rationale, and process described in the document.

## INPUT
Full Document: "{document_text}"

## CONSTRAINTS
1. **Context-Independent**
   - Each QA must be self-contained and understandable without referencing the original text.
   - Replace pronouns with explicit entities.

2. **No Hallucination**
   - Use only facts explicitly stated in the document.

3. **Expansion-Oriented Thinking**
   - For each sentence or factual unit, consider the 5W1H perspectives as prompts to explore:
     - WHO is involved?
     - WHAT happened or is described?
     - WHEN did it occur?
     - WHERE did it occur?
     - WHY did it occur?
     - HOW was it carried out?
   - These perspectives are **guides** to inspire multiple possible QA pairs, even if they are implicit or only partially expressed.

4. **Coverage**
   - Extract all possible QA pairs that can be reasonably derived from the document.

## OUTPUT FORMAT
Return a JSON list of QA objects:

[
  {{"question": "...", "answer": "..."}},
  ...
]
"""


EXTRACTION_PROMPT = """## ROLE
You are a Semantic Knowledge Graph Builder.
Extract every structured triples (entity1, relation, entity2) from the Q&A pair, following the rules below.

## GOAL
From the question-answer pair, extract only useful, knowledge-ready triples that can serve as entries in a semantic knowledge graph.

## RULES
Extract clean (subject, relation, object) triples following the rules:

1. Split every stated or clearly implied fact into minimal triples; integrate question and answer context when needed.

2. Entities (entity1, entity2) must be short, concrete noun phrases.
   - No pronouns (this, that, it, its, these, those, etc.).
   - Entities must not be unresolved or reference-based pronouns (e.g., those, they, someone, anyone, whoever); if such a pronoun appears, rewrite it into a specific, explicit noun phrase or skip the triple.
   - No clauses or relative clauses (no "who/that/which/what/as it ..." inside an entity).
   - No long gerund or sentence-like phrases. If a phrase contains a verb or clause marker, rewrite it into a concise noun concept or skip the triple.

3. Relations must be short, canonical verbs or verb phrases.
   - Express a single semantic link between the two entities (e.g., causes, leads to, supports, believes, opposes).
   - Must be a compact predicate, not a sentence fragment.
   - No pronouns or clause markers inside the relation (no "its", "that", "as it", "what", etc.).
   - If the source uses an idiomatic or long expression, rewrite it into a simple canonical relation without pronouns or embedded clauses, or skip the triple.

4. Include a fact if it can be clearly rewritten into a concise, explicit triple that fits the rules above; otherwise skip it.

5. Output only concise, interpretable, knowledge-ready triples.

## INPUT
Q: {question}
A: {answer}

## OUTPUT FORMAT (JSON List)
- Return a list of JSON objects.
- Return [] if no valid triples exist.

[
  {{"entity1": "Specific_Noun", "relation": "precise_verb_phrase", "entity2": "Specific_Noun"}}
]
"""


RESOLUTION_PROMPT = (
    "Find duplicate {item_type} for the item and an alias that best represents "
    "the duplicates. Duplicates are those that are the same in meaning, such as "
    "with variation in tense, plural form, stem form, case, abbreviation, "
    "shorthand. Return an empty list if there are none."
)
