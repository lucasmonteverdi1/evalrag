# Notes and remarks on the project for better personal understanding during development

This project is a tool that measures if a RAG system answers correctly. 

**RAG system flow:** question → get relevant documents (chunks) → generate answer.

What this project aims to determine if that generated answer is **reliable**, initially measured
by 3 metrics: faithfulness, 

## Faithfulness
"Is everything that the answer claims backed by the documents, or did the model invent anything?"

### Example
* Retrieved document: "Paris is the capital city of France."
* RAG response: "Paris is the capital city of France. It has a population of 5 million people."
* The first phrase is backed by the document, the second one was "invented" (not contained in the document).
* Faithfulness: 1/2 phrases = 0.50

### How is it measured
1. `decompose_claims()`: the answer is split into individual phrases, each one being an affirmation to verify.
2. The affirmations and the documents are sent to the judge LLM, and it's asked, 
for each affirmation, "Is it backed by these documents? Yes/no." 
3. `parse_verdict() + score`: We read the judge's response on the claim (the `supported` boolean field). 
Based on that, we calculate the score as the sum of supported claim divided by the total claims. 

### Stubs, FakeProvider y labels
Before spending money on calling LLMs, we want to test if the code actually works. 
**Stubs:** 4/5 handwritten examples (question + answer + documents) that we know how they'd have to score.

**FakeProvider:** A "fake judge" that doesn't go on the internet, returns predefined answers. 
This way, tests are free, fast and consistent.

**Labels:** on each stub, we type by hand the "truth" (which phrases are backed and which ones are not), 
and from there we generate what the fake judge should answer. This way, they are never out of sync.
