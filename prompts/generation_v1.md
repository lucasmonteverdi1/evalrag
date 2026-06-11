You are generating an evaluation dataset for a RAG system. Given a source document
CHUNK, write {n} question/answer pair(s) that can be answered **using only this chunk**.

Requirements for each pair:
- The QUESTION must be answerable from the chunk alone — not from outside knowledge.
- The EXPECTED_ANSWER must be correct and fully supported by the chunk.
- Prefer specific, factual questions over vague or yes/no ones.
- Vary the questions; do not ask the same thing twice.

## CHUNK
{chunk}

## Instructions
First, think briefly about what this chunk can support. Then output your final result as
a single fenced JSON block with exactly {n} pair(s):

    ```json
    {{"pairs": [{{"question": "...", "expected_answer": "..."}}]}}
    ```

The JSON block MUST be the last thing in your response.
