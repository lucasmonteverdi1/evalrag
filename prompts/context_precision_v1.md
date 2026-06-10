You are a strict judge evaluating **context precision**: for a given question, which of
the retrieved chunks are actually relevant to answering it.

You are given:
1. The QUESTION that was asked.
2. A numbered list of retrieved CHUNKS.

For EACH chunk, decide whether it is **relevant** to answering the question:
- `true`  — the chunk contains information that helps answer the question.
- `false` — the chunk is off-topic or contributes nothing toward answering it.

Judge each chunk independently and ONLY by its usefulness for THIS question. A chunk that
is true in general but unrelated to the question is NOT relevant.

## QUESTION
{question}

## CHUNKS
{chunks}

## Instructions
First, reason step by step about each chunk's relevance to the question. Then output your
final verdict as a single fenced JSON block with one entry per chunk, using the chunk's
number as its `index`:

    ```json
    {{"chunks": [{{"index": 0, "relevant": true}}, {{"index": 1, "relevant": false}}]}}
    ```

The JSON block MUST be the last thing in your response and MUST contain exactly one entry
per chunk above.
