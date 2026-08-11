Markdown Document RAG source rules
==================================

1. Place only UTF-8 .md files in data/documents/source.
2. Use a unique lowercase filename such as delivery_policy.md.
3. Start every document with exactly one H1 title, for example:

   # Çatdırılma qaydaları

4. Use H2-H6 headings to divide policy sections.
5. After adding or changing files, run:

   .\.venv\Scripts\python.exe -m app.indexing.documents index
   .\.venv\Scripts\python.exe -m app.indexing.documents status

6. Create data/evals/document_retrieval.json with at least 30 manually labelled cases,
   then run the document evaluator before enabling DOCUMENT_SEARCH_ENABLED.

