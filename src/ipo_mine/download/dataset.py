import json
from typing import Dict, Any, List, Union, Optional, Tuple
from collections import defaultdict
from ..entities import S1Filing, S1FilingImage, CompanyFilings, Filing, FilingImage, FormType
import random

class Dataset:
    """
    Manages the collection of CompanyFilings objects, handling JSON loading and sampling.
    """
    def __init__(self, raw_data_or_path: Union[str, Dict[str, Dict[str, Any]]]):
        """
        Initializes the dataset by either loading from a file path or processing 
        an existing dictionary of raw data.
        """
        if isinstance(raw_data_or_path, str):
            raw_data = self._load_json(raw_data_or_path)
        else:
            raw_data = raw_data_or_path
            
        self.companies: Dict[str, CompanyFilings] = self._parse_data(raw_data)

    def _load_json(self, file_path: str) -> Dict[str, Dict[str, Any]]:
        """Handles opening and parsing the JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def _parse_data(self, raw_data: Dict[str, Dict[str, Any]]) -> Dict[str, CompanyFilings]:
        """Iterates through the raw data dictionary and creates CompanyFilings objects."""
        parsed_companies = {}
        
        for cik, company_data in raw_data.items():
            # Extract and parse filings
            raw_filings = company_data.get("filings", [])
            filings_list = []
            
            for filing_data in raw_filings:
                # Parse images for this filing
                raw_images = filing_data.get("images", [])
                images_list = [
                    FilingImage(
                        img_name=img.get('img_name'),
                        url=img.get('url'),
                        local_path=img.get('local_path')
                    ) for img in raw_images
                ]
                
                # Parse form_type enum
                form_type_str = filing_data.get('form_type', '')
                try:
                    form_type = FormType(form_type_str)
                except ValueError:
                    print(f"Warning: Unknown form type '{form_type_str}' for CIK {cik}")
                    continue
                
                # Create Filing object
                filing = Filing(
                    form_type=form_type,
                    acession_number=filing_data.get('acession_number', ''),
                    filing_date=filing_data.get('filing_date', ''),
                    primary_document=filing_data.get('primary_document', ''),
                    filing_url=filing_data.get('filing_url', ''),
                    local_path=filing_data.get('local_path'),
                    images=images_list,
                    raw_content=filing_data.get('raw_content')
                )
                filings_list.append(filing)
            
            # Create CompanyFilings object
            company = CompanyFilings(
                tickers=company_data.get('tickers', []),
                cik=company_data.get('cik', cik),
                name=company_data.get('name', ''),
                sic=company_data.get('sic'),
                industry=company_data.get('industry'),
                office=company_data.get('office'),
                exchanges=company_data.get('exchanges'),
                filings=filings_list
            )
            
            parsed_companies[cik] = company
                
        return parsed_companies

    def sample_filings_by_year(self, num_samples: int, start_year: Optional[int] = None, end_year: Optional[int] = None) -> 'Dataset':
        """
        Samples up to 'num_samples' filings per year and returns a new 
        Dataset instance containing only the sampled filings.

        :param num_samples: The maximum number of filings to sample per year.
        :param start_year: The start year to sample (inclusive)
        :param end_year: The end year to sample (exclusive)
        :return: A new Dataset instance with the sampled data.
        """
        
        # Collect all filings across all companies, grouped by year
        filings_by_year: Dict[int, List[Tuple[str, Filing]]] = defaultdict(list)
        
        for cik, company in self.companies.items():
            for filing in company.filings:
                # Extract year from filing_date (format: "YYYY-MM-DD")
                try:
                    year = int(filing.filing_date.split('-')[0])
                except (ValueError, IndexError, AttributeError):
                    print(f"Warning: Could not parse filing_date for CIK {cik}")
                    continue
                
                if (start_year is not None and year < start_year):
                    continue
                if (end_year is not None and year >= end_year):
                    continue
                
                filings_by_year[year].append((cik, filing))
        
        # Sample filings per year
        sampled_filings: Dict[str, List[Filing]] = defaultdict(list)
        sorted_years = sorted(filings_by_year.keys())
        
        for year in sorted_years:
            filing_list = filings_by_year[year]
            k = min(num_samples, len(filing_list))
            yearly_sample = random.sample(filing_list, k)
            
            for cik, filing in yearly_sample:
                sampled_filings[cik].append(filing)
        
        # Reconstruct raw data structure with only sampled filings
        new_raw_data = {}
        
        for cik, filing_list in sampled_filings.items():
            company = self.companies[cik]
            
            # Convert filings to dictionaries
            filings_data = []
            for filing in filing_list:
                filing_dict = {
                    'form_type': filing.form_type.value,
                    'acession_number': filing.acession_number,
                    'filing_date': filing.filing_date,
                    'primary_document': filing.primary_document,
                    'filing_url': filing.filing_url,
                    'local_path': filing.local_path,
                    'images': [
                        {
                            'img_name': img.img_name,
                            'url': img.url,
                            'local_path': img.local_path
                        } for img in filing.images
                    ],
                    'raw_content': filing.raw_content
                }
                filings_data.append(filing_dict)
            
            # Reconstruct company data
            new_raw_data[cik] = {
                'tickers': company.tickers,
                'cik': company.cik,
                'name': company.name,
                'sic': company.sic,
                'industry': company.industry,
                'office': company.office,
                'exchanges': company.exchanges,
                'filings': filings_data
            }
        
        total_sampled = sum(len(filings) for filings in sampled_filings.values())
        print(f"Sampled {total_sampled} filings across {len(filings_by_year)} years from {len(sampled_filings)} companies.")
        
        return Dataset(new_raw_data)

    def get_company_by_cik(self, cik: str) -> Optional[CompanyFilings]:
        """Retrieves the CompanyFilings object for a given CIK."""
        return self.companies.get(cik)

    def get_company_by_ticker(self, ticker: str) -> Optional[CompanyFilings]:
        """Retrieves the CompanyFilings object for a given ticker."""
        ticker_upper = ticker.upper()
        for company in self.companies.values():
            if ticker_upper in [t.upper() for t in company.tickers]:
                return company
        return None

    def __len__(self) -> int:
        """Returns the number of companies in the dataset."""
        return len(self.companies)

    def __iter__(self):
        """Allows iteration over the CompanyFilings objects in the dataset."""
        return iter(self.companies.values())

    def __repr__(self):
        total_filings = sum(len(company.filings) for company in self.companies.values())
        return f"Dataset(total_companies={len(self.companies)}, total_filings={total_filings})"