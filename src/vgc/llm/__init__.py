"""Advisory LLM layer (step 2: harness + fake LLM; no network calls anywhere by default).

The LLM only proposes up to three of OUR joint move pairs by ID from a numbered option
list the engine provides. The engine always has a legal fallback and has the last word;
the LLM never does arithmetic and only ever sees public information. The real OpenAI
client imports the `openai` package lazily, so importing anything here needs no extras.
"""
