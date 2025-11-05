# backends/openai_backend.py
# drop-in backend example. replace the body of generate() with your call.

def generate(prompt: str) -> str:
    # example shape only. replace with your client code.
    # return client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}]).choices[0].message.content
    raise NotImplementedError("wire your openai or internal model call here and return a string")