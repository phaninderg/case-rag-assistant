#!/usr/bin/env python3
"""
Script to import cases from ServiceNow into the vector database using Selenium.
"""
import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('import.log')
    ]
)
logger = logging.getLogger(__name__)

from src.services.case_service import CaseService
from src.services.web_crawler import WebCrawler, CaseImporter
from src.services.llm_service import LLMService
from src.config.settings import settings

class CaseImportLogger:
    """Helper class to log and save case import data."""
    
    def __init__(self, output_file: str = 'cases.json'):
        self.output_file = output_file
        self.cases = []
        self.stats = {
            'total_imported': 0,
            'total_processed': 0,
            'errors': 0,
            'skipped': 0,
            'start_time': datetime.utcnow().isoformat(),
            'end_time': None,
            'duration_seconds': None
        }
    
    def add_case(self, case_data: Dict[str, Any], status: str, error: Optional[str] = None):
        """Add a case to the log with its status."""
        # Clean up the description and close_notes by replacing newlines with spaces
        if 'description' in case_data and case_data['description']:
            case_data = case_data.copy()  # Don't modify the original
            # Replace all newlines with spaces and clean up whitespace
            case_data['description'] = case_data['description'].replace('\n', ' ').replace('\r', ' ')
            case_data['description'] = ' '.join(case_data['description'].split())
        
        # Also clean up close_notes if it exists
        if 'close_notes' in case_data and case_data['close_notes']:
            case_data['close_notes'] = case_data['close_notes'].replace('\n', ' ').replace('\r', ' ').strip()
        
        case_log = {
            'case_number': case_data.get('case_number'),
            'subject': case_data.get('subject'),
            'status': status,
            'timestamp': datetime.utcnow().isoformat(),
            'data': case_data
        }
        
        if error:
            case_log['error'] = str(error)
        
        self.cases.append(case_log)
        self.stats['total_processed'] += 1
        
        if status == 'imported':
            self.stats['total_imported'] += 1
        elif status == 'skipped':
            self.stats['skipped'] += 1
        elif status == 'error':
            self.stats['errors'] += 1
    
    def save(self):
        """Save the case data and statistics to a JSON file."""
        self.stats['end_time'] = datetime.utcnow().isoformat()
        start = datetime.fromisoformat(self.stats['start_time'])
        end = datetime.fromisoformat(self.stats['end_time'])
        self.stats['duration_seconds'] = (end - start).total_seconds()
        
        output = {
            'metadata': {
                'generated_at': datetime.utcnow().isoformat(),
                'total_cases': len(self.cases),
                'stats': self.stats
            },
            'cases': self.cases
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.cases)} cases to {self.output_file}")

def setup_debug_environment():
    """Set up environment for debugging."""
    # Enable debug logging
    logging.getLogger().setLevel(logging.DEBUG)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.DEBUG)
    
    # Enable HTTP debugging
    import http.client as http_client
    http_client.HTTPConnection.debuglevel = 1
    
    # Enable requests logging
    logging.getLogger('urllib3').setLevel(logging.DEBUG)
    requests_log = logging.getLogger("urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True

def process_case(case_data: Dict[str, Any], case_service: CaseService, import_logger: CaseImportLogger) -> bool:
    """
    Process a single case and save it to the database.
    
    Args:
        case_data: Dictionary containing case data
        case_service: Instance of CaseService
        import_logger: Instance of CaseImportLogger
        
    Returns:
        bool: True if case was processed successfully, False otherwise
    """
    case_number = case_data.get('case_number', 'N/A')
    
    try:
        # Log the extracted case data
        logger.debug(f"Processing case {case_number}:")
        logger.debug(f"  Subject: {case_data.get('subject')}")
        logger.debug(f"  Description length: {len(case_data.get('description', ''))} chars")
        logger.debug(f"  Tags: {case_data.get('tags', [])}")
        
        # Check if case already exists
        existing_case = case_service.get_case(case_number) if case_number != 'N/A' else None
        if existing_case:
            logger.info(f"Case {case_number} already exists, skipping...")
            import_logger.add_case(case_data, 'skipped', 'Case already exists')
            return True
        
        # Create the case
        case = case_service.create_case(
            subject=case_data.get('subject', 'Untitled Case'),
            description=case_data.get('description', ''),
            case_number=case_number,
            parent_case=case_data.get('parent_case'),
            close_notes=case_data.get('close_notes'),
            tags=case_data.get('tags', []),
            **case_data.get('metadata', {})
        )
        
        if case:
            logger.info(f"Successfully imported case: {case_number}")
            import_logger.add_case(case_data, 'imported')
            return True
        
        error_msg = f"Failed to import case: {case_number}"
        logger.error(error_msg)
        import_logger.add_case(case_data, 'error', error_msg)
        return False
        
    except Exception as e:
        error_msg = f"Error processing case {case_number}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        import_logger.add_case(case_data, 'error', error_msg)
        return False

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Import cases from ServiceNow')
    parser.add_argument('url', help='ServiceNow case list URL')
    parser.add_argument('--max-pages', type=int, default=10, help='Maximum number of pages to process')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-browser', action='store_true', help='Run in headless mode (no browser GUI)')
    parser.add_argument('--output', default='cases.json', help='Output file for cases (JSON)')
    
    args = parser.parse_args()
    
    if args.debug:
        setup_debug_environment()
    
    logger.info(f"Starting import from {args.url}")
    logger.info(f"Max pages: {args.max_pages}")
    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"Headless mode: {args.no_browser}")
    logger.info(f"Output file: {args.output}")
    
    # Initialize the import logger
    import_logger = CaseImportLogger(output_file=args.output)
    
    try:
        # Initialize services
        llm_service = LLMService()
        case_service = CaseService(llm_service=llm_service)
        
        # Initialize importer with the import_logger
        importer = CaseImporter(case_service, debug=args.debug)
        
        # Start import and get the crawler
        crawler = WebCrawler(
            base_url=args.url,
            max_pages=args.max_pages,
            debug=args.debug
        )
        
        # Process each case using our enhanced function
        for case_data in crawler.crawl_servicenow_cases(args.url):
            if not case_data or not isinstance(case_data, dict):
                logger.warning("Skipping invalid case data")
                continue
                
            # Process the case with our enhanced function
            process_case(case_data, case_service, import_logger)
        
        # Save the import results
        import_logger.save()
        
        # Print summary
        logger.info("\n=== Import Summary ===")
        logger.info(f"Total processed: {import_logger.stats['total_processed']}")
        logger.info(f"Total imported: {import_logger.stats['total_imported']}")
        logger.info(f"Total errors: {import_logger.stats['errors']}")
        logger.info(f"Total skipped: {import_logger.stats['skipped']}")
        logger.info(f"Duration: {import_logger.stats['duration_seconds']:.2f} seconds")
        
        # Return success status
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nImport cancelled by user")
        import_logger.save()  # Save progress before exiting
        return 1
    except Exception as e:
        logger.error(f"Fatal error during import: {str(e)}", exc_info=args.debug)
        import_logger.save()  # Save progress before exiting
        return 1

if __name__ == "__main__":
    sys.exit(main())