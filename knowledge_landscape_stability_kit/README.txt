knowledge landscape stability kit
=================================

what it is
- a no-api tool that flags unstable topics in internal docs that make chatbots fail
- scans .md and .txt files, perturbs questions, runs a stability score, and ranks results
- outputs a csv and an html report your team can review and fix

install
```
pip install -U numpy scikit-learn matplotlib
```

run with the default offline backend
```
python knowledge_landscape_scan.py --docs /path/to/docs --out ./scan_out
```

run with your own model backend
```
python knowledge_landscape_scan.py --docs /path/to/docs --out ./scan_out --backend backends/openai_backend.py
```

what the report means
- lower score = stable answers across rephrasings
- higher score = likely disagreement or unclear wording in docs
- fix the top items first, re-run, and watch the score drop

how to wire your backend
- edit `backends/openai_backend.py` and put your model call in `generate(prompt)`
- or create your own backend file with the same function signature
- pass it via `--backend /path/to/your_backend.py`

outputs
- `stability_scan.csv` with columns: file, concept, score, curvature, spread, entropy
- `stability_scan.html` with a visual table and colored bars

notes
- this kit ships offline by default so you can demo it without credentials
- accuracy improves when you route `generate(prompt)` to your production llm