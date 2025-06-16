import csv
import requests
import json
from typing import List, Dict, Any
import argparse
from pathlib import Path
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CaseImporter:
    def __init__(self, api_url: str = "http://localhost:8000/api/cases"):
        """
        Initialize the CaseImporter with the API URL.
        
        Args:
            api_url: Base URL of the API (default: http://localhost:8000/api/cases)
        """
        self.api_url = api_url
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def read_csv_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Read cases from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of dictionaries containing case data
        """
        cases = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Clean up the row data
                    case = {k.strip(): v.strip() if isinstance(v, str) else v 
                           for k, v in row.items()}
                    cases.append(case)
            logger.info(f"Successfully read {len(cases)} cases from {file_path}")
            return cases
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            raise

    def create_case(self, case_data: Dict[str, Any]) -> bool:
        """
        Create a case using the API.
        
        Args:
            case_data: Dictionary containing case data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure all required fields are present and not empty
            case_task_number = case_data.get("case_task_number") or case_data.get("number")
            if not case_task_number:
                logger.warning("Skipping case: Missing required field 'case_task_number'")
                return False
                
            # Prepare the request data with required fields
            steps_support = case_data.get("steps_support", "").strip()
            if not steps_support:
                steps_support = "No support steps provided."  # Default message when empty
                
            payload = {
                "case_task_number": str(case_task_number).strip(),
                "parent_case": str(case_data.get("parent", "")).strip() or None,
                "issue": str(case_data.get("issue", "")).strip(),
                "root_cause": str(case_data.get("root_cause", "")).strip(),
                "resolution": str(case_data.get("resolution", "")).strip(),
                "steps_support": steps_support,
            }
            
            # Validate required fields
            required_fields = ["issue", "root_cause", "resolution"]
            missing_fields = [field for field in required_fields if not payload[field]]
            if missing_fields:
                logger.warning(f"Skipping case {case_task_number}: Missing required fields: {', '.join(missing_fields)}")
                return False
            
            # Add metadata
            payload["metadata"] = {
                "source": "csv_import",
                "original_data": {k: str(v) for k, v in case_data.items() 
                                if k not in ["case_task_number", "number", "parent_case", "parent",
                                           "issue", "root_cause", "resolution", "steps_support"]
                                and v is not None and str(v).strip()}
            }
            
            # Filter out None values from metadata
            payload["metadata"] = {k: v for k, v in payload["metadata"].items() if v is not None}
            
            # Log the case being processed
            logger.debug(f"Creating case: {case_task_number}")
            
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self.headers,
                timeout=30  # Add timeout to prevent hanging
            )
            
            if response.status_code == 201:
                logger.debug(f"Successfully created case: {case_task_number}")
                return True
            else:
                logger.error(
                    f"Failed to create case {case_task_number}. "
                    f"Status: {response.status_code}, Response: {response.text}, "
                    f"Payload: {json.dumps(payload, indent=2)}"
                )
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error creating case {case_task_number}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating case {case_task_number}: {str(e)}", exc_info=True)
            return False

    def import_cases(self, file_path: str, batch_size: int = 10) -> None:
        """
        Import cases from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            batch_size: Number of cases to process in each batch
        """
        try:
            # Read cases from CSV
            cases = self.read_csv_file(file_path)
            if not cases:
                logger.warning("No cases found in the CSV file")
                return
                
            logger.info(f"Starting import of {len(cases)} cases...")
            
            # Process cases in batches
            success_count = 0
            for i in tqdm(range(0, len(cases), batch_size), desc="Importing cases"):
                batch = cases[i:i + batch_size]
                for case in batch:
                    if self.create_case(case):
                        success_count += 1
            
            logger.info(
                f"Import completed. Successfully imported {success_count} of {len(cases)} cases."
            )
            if success_count < len(cases):
                logger.warning(f"Failed to import {len(cases) - success_count} cases")
                
        except Exception as e:
            logger.error(f"Error during import: {str(e)}")
            raise


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Import cases from CSV to the vector database")
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the CSV file containing cases"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/api/cases",
        help="Base URL of the API (default: http://localhost:8000/api/cases)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of cases to process in each batch (default: 10)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set log level
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(log_level)
    
    # Check if file exists
    if not Path(args.csv_file).is_file():
        logger.error(f"File not found: {args.csv_file}")
        return
    
    try:
        # Initialize the importer and start the import
        importer = CaseImporter(api_url=args.api_url)
        importer.import_cases(args.csv_file, batch_size=args.batch_size)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())