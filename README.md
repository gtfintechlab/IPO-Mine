# IPO-Mine: A Toolkit and Dataset for Section-Structured Analysis of Long, Multimodal IPO Documents

[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg)](https://huggingface.co/datasets/gtfintechlab/ipo-images) [![PyPI - ipo-mine](https://img.shields.io/pypi/v/ipo-mine.svg)](https://pypi.org/project/ipo-mine/) [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

## Dataset Construction Pipelines
<table>
    <tr>
        <td>
            Image Dataset Pipeline
        </td>
        <td>
            Text Dataset Pipeline
        </td>
    </tr>
    <tr>
        <td>
            <img src="images/ipo-image-pipeline.png" alt="Description 1" width="300"/>
        </td>
        <td>
            <img src="images/ipo-text-pipeline.png" alt="Description 2" width="300"/>
        </td>
    </tr>
</table>

## Quickstart

### Install from PyPI

```bash
pip install ipo-mine
```

## Using `ipo-mine` to Download an IPO Filing (Python API)

```python
from download import IPODownloader, Company

downloader = IPODownloader(
    email="example@gmail.com",
    company="Your Example Organization"
)

company = Company.from_ticker("SNOW")

company_filings = downloader.download_ipo(
    company,
    limit=1,
    save_filing=True,
    save_images=False,
    verbose=True
)

filing = company_filings.filings[0]
```

## Parsing the  Table of Contents

```python
results = parser.parse_company(
    ticker="SNOW",
    validate=False
)
```

## CLI Usage

You can use the command-line interface to download and parse filings without writing Python code.

### Download

Download the latest S-1 filing for a company:

```bash
ipo-mine download SNOW --email your@email.com --org "Your Org"
```

Options:
- `--limit N`: Download previous N filings (default: 1)
- `--images`: Download and extract images from the filing
- `--all`: Download all available IPO filings for the ticker

### Parse

Parse a downloaded filing into section-specific files:

```bash
ipo-mine parse SNOW
```

Options:
- `--validate`: Enable LLM-based validation of extracted sections
- `--provider`: LLM provider (anthropic, openai, google, huggingface)
- `--mode`: Validation mode (binary, likert)

### Validate

Run LLM validation on existing parsed text files to check for truncation or completeness.

```bash
ipo-mine validate SNOW --provider anthropic
```

#### Supported Providers

You can choose from the following providers (requires API keys):

| Provider | Argument | Env Variable |
| :--- | :--- | :--- |
| **Anthropic** (Claude) | `--provider anthropic` | `ANTHROPIC_API_KEY` |
| **OpenAI** (GPT-4o) | `--provider openai` | `OPENAI_API_KEY` |
| **Google** (Gemini) | `--provider google` | `GOOGLE_API_KEY` |
| **HuggingFace** | `--provider huggingface` | `HUGGINGFACE_API_KEY` |

#### Validation Modes

- **Binary (`--mode binary`)**: Returns "Yes" (Valid) or "No" (Truncated/Incomplete). Default.
- **Likert (`--mode likert`)**: Returns a confidence score from 1 (Incomplete) to 5 (Complete).

#### Authentication

The CLI will look for API keys in this order:
1.  **Command Line Argument**: `--api-key "sk-..."`
2.  **Environment Variable**: e.g., `export OPENAI_API_KEY="sk-..."`
3.  **Interactive Prompt**: If neither is found, the CLI will securely prompt you to enter the key (input is hidden).

#### Examples

**Validate using OpenAI with Likert scale:**
```bash
ipo-mine validate TSLA --provider openai --mode likert
```

**Validate using Google Gemini with explicit key:**
```bash
ipo-mine validate TSLA --provider google --api-key "your-api-key"
```

## Notes

- The SEC requires a descriptive User-Agent. Provide a real organization name and your email.
- `download_ipo` returns a `CompanyFilings` object; use `company_filings.filings[0]` to pass a `Filing` into the parser.
- The parser automatically chooses HTML or text parsing based on the filing URL.



[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
