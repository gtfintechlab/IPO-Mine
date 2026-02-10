# IPO-Mine: A Toolkit and Dataset for Section-Structured Analysis of Long, Multimodal IPO Documents

[![Image Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg)](https://huggingface.co/datasets/gtfintechlab/ipo-images) [![PyPI - ipo-mine](https://img.shields.io/pypi/v/ipo-mine.svg)](https://pypi.org/project/ipo-mine/) [![CC BY 4.0][cc-by-shield]][cc-by]

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

## Using `ipo-mine` to Download an IPO Filing

```python
from ipo_mine import IPODownloader, Company

dl = IPODownloader("<email>", "<company>")
uber = Company.from_ticker("UBER")
filing = dl.download_ipo(uber)
```

## Parsing the Risk Factors section

```python
from ipo_mine import IPOParser

parser = IPOParser(filing)
risk_factors = parser.parse_section("Risk Factors")
```

## Notes

- The SEC requires a descriptive User-Agent. Provide a real organization name and your email.
- `download_ipo` returns a `CompanyFilings` object; use `company_filings.filings[0]` to pass a `Filing` into the parser.
- The parser automatically chooses HTML or text parsing based on the filing URL.

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
