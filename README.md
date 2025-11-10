<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">

</head>
<body>

<h1>knowledge landscape stability kit</h1>

<p>
compression-aware intelligence (CAI) states that Hallucination is a representational conflict problem. Compression strain measures it. CAI is the framework for evaluating it. this toolkit highlights where internal documentation causes genai systems to give inconsistent answers. it works by checking how stable the model’s response is when the same question is phrased in slightly different ways.
</p>

<p>
low stability = documentation is unclear or conflicting.  
high stability = documentation is consistent.
</p>

<hr>

<h2>60 second demo</h2>

<pre>
pip install -r requirements.txt
python knowledge_landscape_scan.py --docs ./sample_docs --out ./scan_out
open ./scan_out/stability_scan.html
</pre>

<p>the highest scoring items are the sections most likely to cause chatbot inconsistency.</p>

<hr>

<h2>use on client knowledge bases</h2>

<p>export pages from sharepoint, confluence, or servicenow as <code>.md</code> or <code>.txt</code>, then run:</p>

<pre>
python knowledge_landscape_scan.py --docs /path/to/exported_docs --out ./scan_out
open ./scan_out/stability_scan.html
</pre>

<p>fix the top 20 highest score rows first, then re-run to confirm stability improves.</p>

<hr>

<h2>use your own llm (optional)</h2>

<p>edit this file:</p>
<pre>backends/openai_backend.py</pre>

<p>replace the body of:</p>
<pre>def generate(prompt: str) -> str:</pre>

<p>with your model call, then run:</p>

<pre>
export OPENAI_API_KEY=yourkey
python knowledge_landscape_scan.py --backend backends/openai_backend.py --docs /path/to/docs --out ./scan_out
</pre>

<hr>

<h2>output artifacts</h2>

<pre>
stability_scan.csv     # sortable spreadsheet of topics and scores
stability_scan.html    # visual report with color indicators
</pre>

<p>columns:</p>
<ul>
  <li><strong>file</strong> = source document</li>
  <li><strong>concept</strong> = extracted topic or question</li>
  <li><strong>score</strong> = instability score (0 to 1)</li>
  <li><strong>curvature</strong>, <strong>spread</strong>, <strong>entropy</strong> = score components</li>
</ul>

<hr>

<h2>interpretation guide</h2>

<table style="border-collapse: collapse; width:100%; margin-top: 10px;">
<tr><th style="text-align:left;">score</th><th style="text-align:left;">meaning</th><th style="text-align:left;">action</th></tr>
<tr><td>0.00 - 0.30</td><td>stable</td><td>no change needed</td></tr>
<tr><td>0.30 - 0.60</td><td>fuzzy or partially conflicting</td><td>clarify wording</td></tr>
<tr><td>0.60 - 1.00</td><td>internally conflicting</td><td>rewrite or consolidate content</td></tr>
</table>

<hr>

<h2>license</h2>
<p>MIT</p>

</body>
</html>
