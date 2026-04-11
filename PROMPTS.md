# LLM Validator Prompts

This document contains the two main prompts used by the LLM validator in the IPO Mine project. These prompts are used to assess whether extracted sections from SEC S-1 filings are structurally complete or truncated.

---

## 1. Binary Validation Prompt

**Purpose:**
Determine whether the extracted section appears structurally complete (not truncated or cut off).

**Prompt:**

```
Below is text extracted from the "{section_name}" section of an SEC S-1 filing.
{metadata_context}

Task:
Determine whether this extraction appears STRUCTURALLY COMPLETE — i.e.,
it does NOT appear to be truncated, cut off mid-thought, or prematurely ended.

IMPORTANT:
- Ignore whether unrelated, adjacent, or extraneous material appears.
- Ignore whether the section seems "long enough" or "comprehensive".
- Do NOT penalize the presence of other section content unless it proves truncation.

Answer "No" ONLY if you observe clear evidence of truncation, such as:
1. Text ends mid-sentence or mid-clause (e.g., "the Company will", "as described in")
2. An unfinished cross-reference (e.g., "See", "as discussed in Section" with no continuation)
3. Explicit continuation markers ("[continued]", "continued on next page")
4. The text is extremely short and contains almost no substantive content

If none of the above are present, answer "Yes".
If the answer is "No," clearly justify your answer by providing the exact sentences that are truncated.

{examples}

Now evaluate the following extracted text using the same criteria demonstrated in the examples above:
{parsed_text}

Respond ONLY with valid JSON in the following format:
{
  "Answer": "Yes" or "No",
  "Justification": "Brief explanation citing specific textual evidence"
}
```

---

## 2. Likert Scale Validation Prompt

**Purpose:**
Rate your confidence (1-5) that the extraction is not truncated or prematurely cut off.

**Prompt:**

```
Below is text extracted from the "{section_name}" section of an SEC S-1 filing.
{metadata_context}

Task:
Rate your confidence that this extraction is NOT truncated or prematurely cut off.

Ignore:
- Whether the content matches modern expectations
- Whether unrelated or adjacent section material appears
- Whether the section feels "complete" thematically

Use this 5-point scale:

5 = Very High Confidence (Structurally Complete)
    - No evidence of truncation
    - Text ends naturally (even if abruptly)
    - May contain other section material or administrative text

4 = High Confidence (Likely Complete)
    - Minor ambiguity, but no concrete truncation signals

3 = Moderate Confidence (Uncertain)
    - Some ambiguity (e.g., odd ending), but no direct truncation evidence

2 = Low Confidence (Likely Incomplete)
    - Strong signs of cutoff or missing continuation

1 = Very Low Confidence (Clearly Incomplete)
    - Definite mid-sentence cutoff or explicit continuation marker

IMPORTANT:
- For historical filings, assign 5 unless there is direct textual evidence of truncation.

{examples}

Now rate the following extracted text using the same criteria and scale demonstrated in the examples above:
{parsed_text}

Respond ONLY with valid JSON in the following format:
{
  "Answer": 1-5,
  "Justification": "Brief explanation citing specific textual evidence"
}
```

---

**Note:**
- The placeholders `{section_name}`, `{metadata_context}`, `{examples}`, and `{parsed_text}` are dynamically filled during runtime.

