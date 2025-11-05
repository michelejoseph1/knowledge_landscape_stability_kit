knowledge landscape stability kit

this toolkit helps identify where internal documentation causes genai assistants to give inconsistent answers. it checks how stable a model's response is when the same question is phrased in slightly different ways.

low stability = the documentation is unclear or conflicting
high stability = the knowledge is consistent

quick start (60 seconds)

run the sample demo:

git clone https://github.com/michelejoseph1/knowledge_landscape_stability_kit.git
cd knowledge_landscape_stability_kit
pip install -r requirements.txt
python knowledge_landscape_scan.py --docs ./sample_docs --out ./scan_out
open ./scan_out/stability_scan.html


the html report ranks topics by stability:

low score → consistent meaning

high score → conflicting or vague documentation

the highest scoring rows are where chatbots tend to flip answers.

use on client knowledge bases

export sharepoint / confluence / servicenow pages as .md or .txt files and place them in a folder. then run:

python knowledge_landscape_scan.py --docs /path/to/exported_docs --out ./scan_out
open ./scan_out/stability_scan.html


review the top 20 highest scoring rows first. fix those pages. re-run to confirm the score drops.

use your own llm (optional)

to score using your internal model:

open:

backends/openai_backend.py


replace the body of:

def generate(prompt: str) -> str:


with your model call (must return a string)

run:

export OPENAI_API_KEY=yourkey
python knowledge_landscape_scan.py --backend backends/openai_backend.py --docs /path/to/docs --out ./scan_out

outputs
stability_scan.csv     # sortable rows of topics and scores
stability_scan.html    # visual report with colored bars


columns:

file = source document

concept = extracted topic or question

score = instability score (0 to 1)

score interpretation
score	meaning	action
0.00–0.30	stable	no change needed
0.30–0.60	loosely defined	clarify wording
0.60–1.00	conflicting meaning	consolidate or rewrite docs

fixing the highest scoring items usually stabilizes chatbot behavior immediately.

license

mit
