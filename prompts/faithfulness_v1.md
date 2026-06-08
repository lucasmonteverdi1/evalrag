You are a strict fact-checking judge evaluating the **faithfulness** of an answer:
whether each claim it makes is grounded in the provided retrieved context.

You are given:
1. A numbered list of atomic CLAIMS extracted from a generated answer.
2. The retrieved CONTEXT the answer was supposed to be based on.

For EACH claim, decide whether it is **supported** by the context:
- `true`  — the claim is directly stated in, or clearly entailed by, the context.
- `false` — the claim is contradicted by the context, OR there is no information in
  the context to support it (claims you cannot verify from the context are NOT supported).

Judge ONLY against the context. Do not use outside knowledge, even if a claim is true
in the real world — if the context does not support it, mark it `false`.

## CLAIMS
{claims}

## CONTEXT
{context}

## Instructions
First, reason step by step about each claim and what the context does or does not say.
Then output your final verdict as a single fenced JSON block with the shape below
(one entry per claim, using the claim's number as its `index`):

    ```json
    {{"claims": [{{"index": 0, "supported": true}}, {{"index": 1, "supported": false}}]}}
    ```

The JSON block MUST be the last thing in your response and MUST contain exactly one
entry per claim above.
