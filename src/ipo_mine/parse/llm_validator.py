"""
LLM-based validation of parsed S1 filing sections.

This module provides functionality to validate section parser output using
Large Language Models (Claude, Gemini, OpenAI, etc.) with two validation approaches:
1. Binary validation (Yes/No): Does the section appear complete?
2. Likert scale (1-5): Confidence in section completeness.
"""

import json
import os
from typing import Dict, Optional, Literal
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Literal

@dataclass
class FilingMetadata:
    """Metadata about a filing to provide context for validation."""
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    cik: Optional[str] = None
    filing_date: Optional[str] = None
    year: Optional[int] = None
    
    def to_context_string(self) -> str:
        """Convert metadata to a formatted context string for LLM."""
        parts = []
        if self.company_name:
            parts.append(f"Company: {self.company_name}")
        if self.ticker:
            parts.append(f"Ticker: {self.ticker}")
        if self.cik:
            parts.append(f"CIK: {self.cik}")
        if self.filing_date:
            parts.append(f"Filing Date: {self.filing_date}")
        if self.year:
            parts.append(f"Year: {self.year}")
        return " | ".join(parts) if parts else ""

@dataclass
class ValidationExample:
    """A full section example for few-shot learning."""
    text: str
    is_complete: bool
    company_name: str
    justification: str
    likert_score: Optional[int] = None

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from google import genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from huggingface_hub import InferenceClient
    HAS_HUGGINGFACE = True
except ImportError:
    HAS_HUGGINGFACE = False


class ValidationMode(Enum):
    """Validation modes for section completeness assessment."""
    BINARY = "binary"  # Yes/No answer
    LIKERT = "likert"  # 1-5 scale


@dataclass
class BinaryValidationResult:
    """Result from binary (Yes/No) validation."""
    answer: str  # "Yes" or "No"
    justification: str
    raw_response: str


@dataclass
class LikertValidationResult:
    """Result from Likert scale (1-5) validation."""
    answer: int  # 1-5
    justification: str
    raw_response: str


# Prompts for validation

VALIDATION_PROMPT = """Below is text extracted from the "{section_name}" section of an SEC S-1 filing.
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
{{
  "Answer": "Yes" or "No",
  "Justification": "Brief explanation citing specific textual evidence"
}}
"""

LIKERT_PROMPT = """Below is text extracted from the "{section_name}" section of an SEC S-1 filing.
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
{{
  "Answer": 1-5,
  "Justification": "Brief explanation citing specific textual evidence"
}}
"""

def load_example_sections(file_paths: list[tuple[str, bool, str, str, int]]) -> list[ValidationExample]:
    """
    Load full section examples from files for few-shot learning.
    
    Args:
        file_paths: List of tuples (path, is_complete, company_name, justification, likert_score)
    
    Returns:
        List of ValidationExample objects with full section text
    """
    examples = []
    for path, is_complete, company, justification, likert in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            examples.append(ValidationExample(
                text=text,
                is_complete=is_complete,
                company_name=company,
                justification=justification,
                likert_score=likert
            ))
        except Exception as e:
            print(f"Warning: Could not load example from {path}: {e}")
    return examples


class LLMValidator:
    """Validate parsed section content using LLMs."""

    def __init__(
        self,
        provider: Literal["anthropic", "google", "openai", "huggingface"] = "anthropic",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        example_sections: Optional[list[ValidationExample]] = None,
    ):
        """
        Initialize the LLM validator.

        Args:
            provider: LLM provider ("anthropic", "google", "openai", or "huggingface")
            model: Model name (e.g., "claude-sonnet-4-5-20250929", "gemini-2-flash", "gpt-4o", "meta-llama/Llama-2-70b-chat-hf")
            api_key: API key for the provider. If not provided, will try to load from environment.
            example_sections: Optional list of ValidationExample objects for few-shot learning
        """
        self.provider = provider
        self.client = None
        self.example_sections = example_sections or []

        if provider == "anthropic":
            if not HAS_ANTHROPIC:
                raise ImportError(
                    "anthropic package not installed. Install with: pip install anthropic"
                )
            api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not provided and not found in environment"
                )
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model or "claude-sonnet-4-5-20250929"

        elif provider == "google":
            if not HAS_GOOGLE:
                raise ImportError(
                    "google-genai package not installed. Install with: pip install google-genai"
                )
            api_key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not provided and not found in environment")
            self.client = genai.Client(api_key=api_key)
            self.model = model or "gemini-2-flash"

        elif provider == "openai":
            if not HAS_OPENAI:
                raise ImportError(
                    "openai package not installed. Install with: pip install openai"
                )
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not provided and not found in environment")
            self.client = OpenAI(api_key=api_key)
            self.model = model or "gpt-4o"

        elif provider == "huggingface":
            if not HAS_HUGGINGFACE:
                raise ImportError(
                    "huggingface_hub package not installed. Install with: pip install huggingface_hub"
                )
            api_key = api_key or os.environ.get("HUGGINGFACE_API_KEY")
            if not api_key:
                raise ValueError("HUGGINGFACE_API_KEY not provided and not found in environment")
            self.client = InferenceClient(api_key=api_key)
            self.model = model or "meta-llama/Llama-2-70b-chat-hf"

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _parse_json_response(self, text: str) -> Dict:
        """
        Parse JSON from response text with multiple fallback strategies.

        Args:
            text: Response text that should contain JSON

        Returns:
            Parsed JSON object
        """
        import re
        
        original_text = text
        text = text.strip()
        
        # Strategy 1: Remove markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # Strategy 2: Try direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Extract JSON object using regex
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                # Validate it has expected keys
                if "Answer" in parsed or "Justification" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # Strategy 4: Try to find JSON between first { and last }
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            try:
                return json.loads(text[first_brace:last_brace+1])
            except json.JSONDecodeError:
                pass
        
        # Strategy 5: Remove any text before first { and after last }
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if '{' in line:
                text = '\n'.join(lines[i:])
                break
        
        for i in range(len(lines)-1, -1, -1):
            if '}' in lines[i]:
                text = '\n'.join(lines[:i+1])
                break
        
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            # Last resort: raise with helpful context
            preview = original_text[:500] + "..." if len(original_text) > 500 else original_text
            raise ValueError(
                f"Failed to parse JSON response after trying multiple strategies.\n"
                f"Error: {e}\n"
                f"Response preview: {preview}"
            )

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API with JSON mode enforcement."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system="You are a precise analyzer that ONLY outputs valid JSON. Never include explanatory text before or after the JSON.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
        )
        return message.content[0].text

    def _call_google(self, prompt: str) -> str:
        """Call Google Gemini API."""
        from google.genai import types

        text_part = types.Part.from_text(text=prompt)
        contents = [
            types.Content(
                role="user",
                parts=[text_part],
            )
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=1024,
            response_mime_type="application/json",
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT", threshold="OFF"
                ),
            ],
        )

        full_response = ""
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates
                and chunk.candidates[0].content
                and chunk.candidates[0].content.parts
            ):
                full_response += chunk.text

        return full_response

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API with JSON mode enforcement."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with valid JSON only. Do not include any text outside the JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content

    def _call_huggingface(self, prompt: str) -> str:
        """Call HuggingFace Inference API."""
        response = self.client.text_generation(
            prompt=prompt,
            model=self.model,
            max_new_tokens=1000,
            temperature=0.7,
        )
        return response

    def _format_binary_examples(self) -> str:
        """Format example sections for binary validation prompt."""
        if not self.example_sections:
            return ""
        
        examples_text = "\n\n---\nEXAMPLES (from actual 1995 S-1 filings):\n\n"
        for i, ex in enumerate(self.example_sections[:4], 1):  # Limit to 4 examples
            answer = "Yes" if ex.is_complete else "No"
            status = "Complete" if ex.is_complete else "Truncated"
            # Truncate text if too long (keep last 1000 chars to show ending)
            display_text = ex.text if len(ex.text) <= 1500 else "..." + ex.text[-1500:]
            examples_text += f'Example {i} ({status} - {ex.company_name}):\n'
            examples_text += f'Text: "{display_text}"\n'
            examples_text += f'Answer: "{answer}"\n'
            examples_text += f'Justification: "{ex.justification}"\n\n'
        
        return examples_text + "---\n"

    def _format_likert_examples(self) -> str:
        """Format example sections for Likert scale validation prompt."""
        if not self.example_sections:
            return ""
        
        examples_text = "\n\n---\nEXAMPLES (from actual 1995 S-1 filings):\n\n"
        for i, ex in enumerate(self.example_sections[:4], 1):  # Limit to 4 examples
            score = ex.likert_score or (5 if ex.is_complete else 1)
            # Truncate text if too long (keep last 1000 chars to show ending)
            display_text = ex.text if len(ex.text) <= 1500 else "..." + ex.text[-1500:]
            examples_text += f'Example {i} (Score {score} - {ex.company_name}):\n'
            examples_text += f'Text: "{display_text}"\n'
            examples_text += f'Answer: {score}\n'
            examples_text += f'Justification: "{ex.justification}"\n\n'
        
        return examples_text + "---\n"

    def validate_binary(
        self,
        parsed_text: str,
        section_name: str,
        metadata: Optional[FilingMetadata] = None,
    ) -> BinaryValidationResult:
        """
        Validate using binary (Yes/No) prompt.

        Args:
            parsed_text: The extracted section text to validate
            section_name: Name of the section (e.g., "Risk Factors")
            metadata: Optional filing metadata for context

        Returns:
            BinaryValidationResult with Yes/No answer and justification
        """
        metadata_context = ""
        if metadata:
            context_str = metadata.to_context_string()
            if context_str:
                metadata_context = f"\n\nFiling Context: {context_str}"
        
        # Format examples
        examples = self._format_binary_examples()
        
        prompt = VALIDATION_PROMPT.format(
            parsed_text=parsed_text,
            section_name=section_name,
            metadata_context=metadata_context,
            examples=examples,
        )

        if self.provider == "anthropic":
            raw_response = self._call_anthropic(prompt)
        elif self.provider == "google":
            raw_response = self._call_google(prompt)
        elif self.provider == "openai":
            raw_response = self._call_openai(prompt)
        elif self.provider == "huggingface":
            raw_response = self._call_huggingface(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        try:
            parsed = self._parse_json_response(raw_response)
        except ValueError as e:
            # Log the error and provide a default response
            print(f"Warning: JSON parsing failed. Error: {e}")
            print(f"Raw response: {raw_response[:200]}...")
            # Return a conservative default
            return BinaryValidationResult(
                answer="No",
                justification="JSON parsing error - marking as incomplete for safety",
                raw_response=raw_response,
            )
        
        answer = parsed.get("Answer", "").strip("'\"")
        justification = parsed.get("Justification", "")
        
        # Validate answer format
        if answer not in ["Yes", "No"]:
            print(f"Warning: Invalid binary answer '{answer}', defaulting to 'No'")
            answer = "No"
            justification = f"Invalid response format. Original: {answer}. {justification}"

        return BinaryValidationResult(
            answer=answer,
            justification=justification,
            raw_response=raw_response,
        )

    def validate_likert(
        self,
        parsed_text: str,
        section_name: str,
        metadata: Optional[FilingMetadata] = None,
    ) -> LikertValidationResult:
        """
        Validate using Likert scale (1-5) prompt.

        Args:
            parsed_text: The extracted section text to validate
            section_name: Name of the section (e.g., "Risk Factors")
            metadata: Optional filing metadata for context

        Returns:
            LikertValidationResult with rating (1-5) and justification
        """
        metadata_context = ""
        if metadata:
            context_str = metadata.to_context_string()
            if context_str:
                metadata_context = f"\n\nFiling Context: {context_str}"
        
        # Format examples
        examples = self._format_likert_examples()
        
        prompt = LIKERT_PROMPT.format(
            parsed_text=parsed_text,
            section_name=section_name,
            metadata_context=metadata_context,
            examples=examples,
        )

        if self.provider == "anthropic":
            raw_response = self._call_anthropic(prompt)
        elif self.provider == "google":
            raw_response = self._call_google(prompt)
        elif self.provider == "openai":
            raw_response = self._call_openai(prompt)
        elif self.provider == "huggingface":
            raw_response = self._call_huggingface(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        try:
            parsed = self._parse_json_response(raw_response)
        except ValueError as e:
            # Log the error and provide a default response
            print(f"Warning: JSON parsing failed. Error: {e}")
            print(f"Raw response: {raw_response[:200]}...")
            # Return a conservative default (uncertain)
            return LikertValidationResult(
                answer=3,
                justification="JSON parsing error - assigning neutral score",
                raw_response=raw_response,
            )
        
        try:
            answer = int(parsed.get("Answer", 3))
        except (ValueError, TypeError):
            print(f"Warning: Could not parse Likert answer, defaulting to 3")
            answer = 3
        
        justification = parsed.get("Justification", "")

        # Validate answer is in range
        if not (1 <= answer <= 5):
            raise ValueError(f"Invalid Likert answer: {answer}. Must be 1-5.")

        return LikertValidationResult(
            answer=answer,
            justification=justification,
            raw_response=raw_response,
        )

    def validate(
        self,
        parsed_text: str,
        section_name: str,
        mode: ValidationMode = ValidationMode.BINARY,
        metadata: Optional[FilingMetadata] = None,
    ):
        """
        Validate parsed section content.

        Args:
            parsed_text: The extracted section text to validate
            section_name: Name of the section (e.g., "Risk Factors")
            mode: Validation mode (BINARY or LIKERT)
            metadata: Optional filing metadata for context

        Returns:
            BinaryValidationResult or LikertValidationResult depending on mode
        """
        if mode == ValidationMode.BINARY:
            return self.validate_binary(parsed_text, section_name, metadata)
        elif mode == ValidationMode.LIKERT:
            return self.validate_likert(parsed_text, section_name, metadata)
        else:
            raise ValueError(f"Unknown validation mode: {mode}")



def validate_section_binary(
    parsed_text: str,
    section_name: str,
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BinaryValidationResult:
    """
    Quick validation of a section using binary (Yes/No) prompt.

    Args:
        parsed_text: The extracted section text
        section_name: Name of the section
        provider: LLM provider ("anthropic", "google", "openai", or "huggingface")
        model: Model name
        api_key: API key

    Returns:
        BinaryValidationResult
    """
    validator = LLMValidator(provider=provider, model=model, api_key=api_key)
    return validator.validate_binary(parsed_text, section_name)


def validate_section_likert(
    parsed_text: str,
    section_name: str,
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LikertValidationResult:
    """
    Quick validation of a section using Likert scale (1-5) prompt.

    Args:
        parsed_text: The extracted section text
        section_name: Name of the section
        provider: LLM provider ("anthropic", "google", "openai", or "huggingface")
        model: Model name
        api_key: API key

    Returns:
        LikertValidationResult
    """
    validator = LLMValidator(provider=provider, model=model, api_key=api_key)
    return validator.validate_likert(parsed_text, section_name)
