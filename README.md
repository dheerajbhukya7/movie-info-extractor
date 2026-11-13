# Movie Information Extractor

Extract structured movie metadata from supplied text using LangChain, Mistral and Pydantic. The schema includes title, release year, genres, director, cast, rating and summary.

## Run locally

```bash
python -m venv .venv
# Activate the environment
pip install -r requirements.txt
```

Set `MISTRAL_API_KEY` in `.env`, then run either interface:

```bash
python core.py
streamlit run ui_structured_extractor.py
```

Validate extracted fields against the supplied text; schema validation alone does not establish accuracy. Model availability depends on your account.

## Next steps

- Separate CLI input from reusable extraction logic.
- Test missing fields, malformed output and unsupported facts.
- Add a labeled extraction evaluation set.
