<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>knowledge landscape stability kit</title>
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    line-height: 1.55;
    max-width: 820px;
    margin: 40px auto;
    padding: 0 18px;
    color: #222;
  }
  code {
    background: #f3f3f3;
    padding: 3px 5px;
    border-radius: 4px;
    font-size: 14px;
  }
  pre {
    background: #f3f3f3;
    padding: 12px;
    border-radius: 6px;
    overflow-x: auto;
  }
  h1, h2 {
    font-weight: 600;
  }
</style>
</head>
<body>

<h1>knowledge landscape stability kit</h1>

<p>
this toolkit helps identify where internal documentation causes genai assistants to give inconsistent answers. it measures how stable a model's response is when the same question is phrased in slightly different ways.
</p>

<p>
low stability suggests the underlying documentation is unclear or conflicting. high stability suggests the knowledge is consistent.
</p>

<hr>

<h2>quick start (60 seconds)</h2>

<pre>
git clone https://github.com/michelejoseph1/knowledge_landscape_stability_kit.git
cd knowledge_landscape_stability_kit
pip install -r requirements.txt
python knowledge_landscape_scan.py --docs ./sample_docs --out ./scan_out
open ./scan_out/stability_scan.html
</pre>

<p>
what you will see:
</p>

<ul>
  <li>low score = consistent meaning across variations</li>
  <li>high score = unclear or conflicting documentation</li>
</ul>

<p>the highest scoring items are where chatbots tend to flip answers.</p>

<hr>

<h2>use on client knowledge bases</h2>

<p>export sharepoint, confluence, or servicenow pages as <code>.md</code> or <code>.txt</code> then run:</p>

<pre>
python knowledge_landscape_scan.py --docs /path/to/exported_docs --out ./scan_out
open ./scan_out/stability_scan.html
</pre>

<p>fix the top 20 highest scoring rows first, then re-run to confirm the score drops.</p>

<hr>

<h2>use your own llm (optional)</h2>

<p>edit this file:</p>
<pre>backends/openai_backend.py</pre>

<p>replace the body of:</p>
<pre>def generate(prompt: str) -> str:</pre>

<p>with your model call. then run:</p>

<pre>
export OPENAI_API_KEY=yourkey
python knowledge_landscape_scan.py \
  --backend backends/openai_backend.py \
  --docs /path/to/docs \
  --out ./scan_out
</pre>

<hr>

<h2>output artifacts</h2>

<pre>
stability_scan.csv     # sortable spreadsheet of topics and scores
stability_scan.html    # visual report with colored indicators
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
<tr><td>0.30 - 0.60</td><td>fuzzy / loosely defined</td><td>clarify language</td></tr>
<tr><td>0.60 - 1.00</td><td>internally conflicting</td><td>rewrite or consolidate</td></tr>
</table>

<hr>

<h2>license</h2>
<p>mit</p>

</body>
</html>
