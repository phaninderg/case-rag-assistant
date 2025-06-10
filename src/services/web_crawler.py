import re
import logging
from typing import List, Dict, Optional, Set, Generator, Any
from urllib.parse import urljoin, urlparse, urlunparse
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from trafilatura import extract
from pathlib import Path
import json
from datetime import datetime
import uuid
from selenium.webdriver.common.by import By

# Update imports to use absolute paths
from src.models.embeddings import EmbeddingService
from src.config.settings import settings
from src.utils.helpers import generate_id, format_timestamp

logger = logging.getLogger(__name__)

class WebCrawler:
    def __init__(self, base_url: str, max_pages: int = 50, debug: bool = False, verify_ssl: bool = True):
        """
        Initialize the web crawler.
        
        Args:
            base_url: The base URL to start crawling from
            max_pages: Maximum number of pages to crawl
            debug: Enable debug mode (saves HTML, logs more details)
            verify_ssl: Verify SSL certificates
        """
        self.base_url = base_url
        self.max_pages = max_pages
        self.visited_urls: Set[str] = set()
        self.to_visit: List[str] = [base_url]
        self.session = requests.Session()
        self.session.verify = verify_ssl
        self.debug = debug
        self.debug_dir = Path('debug')
        self.authenticated = False  # Track authentication status
        self.driver = None  # Will hold the Selenium WebDriver instance
        self.browser_type = 'google'  # Default browser type
        
        if self.debug:
            self.debug_dir.mkdir(exist_ok=True)
            
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Configure retry strategy with compatibility for older requests versions
        try:
            from requests.adapters import HTTPAdapter
            from requests.packages.urllib3.util.retry import Retry
            
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount('http://', adapter)
            self.session.mount('https://', adapter)
        except ImportError:
            # Fallback for older versions of requests
            from requests.adapters import HTTPAdapter
            adapter = HTTPAdapter(max_retries=3)
            self.session.mount('http://', adapter)
            self.session.mount('https://', adapter)
    
    def _setup_webdriver(self):
        """Set up the Selenium WebDriver with proper configuration."""
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service as ChromeService
        from webdriver_manager.chrome import ChromeDriverManager
        from webdriver_manager.core.os_manager import ChromeType
        from selenium.webdriver.chrome.options import Options
        
        try:
            # Configure Chrome options
            chrome_options = Options()
            # Run in non-headless mode for debugging
            # chrome_options.add_argument("--headless")  # Commented out for debugging
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            # Set up ChromeDriver with WebDriver Manager - simplified version
            driver_path = ChromeDriverManager().install()
            
            # Initialize the WebDriver
            self.driver = webdriver.Chrome(
                service=ChromeService(driver_path),
                options=chrome_options
            )
            
            # Set page load timeout
            self.driver.set_page_load_timeout(30)
            
            logger.info("Successfully initialized Chrome WebDriver")
            
        except Exception as e:
            logger.error(f"Failed to initialize Chrome: {str(e)}")
            # Try alternative approach if the first one fails
            try:
                # Try with ChromeType
                driver_path = ChromeDriverManager(chrome_type=ChromeType.GOOGLE).install()
                
                self.driver = webdriver.Chrome(
                    service=ChromeService(driver_path),
                    options=chrome_options
                )
                logger.info("Successfully initialized Chrome WebDriver (fallback method)")
                
            except Exception as fallback_error:
                logger.error(f"Fallback Chrome initialization failed: {str(fallback_error)}")
                raise RuntimeError(f"Could not initialize Chrome WebDriver: {str(fallback_error)}")
    
    def _authenticate_with_browser(self) -> bool:
        """
        Authenticate by opening Chrome browser for manual login.
        
        Returns:
            bool: True if authentication was successful, False otherwise
        """
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        import time
        
        try:
            if self.driver is not None:
                logger.info("Browser already open, reusing existing session")
                return True
                
            logger.info("Launching Chrome for authentication...")
            self._setup_webdriver()
            
            # Navigate to the ServiceNow login page
            login_url = "https://support.servicenow.com/"
            logger.info(f"Opening login page: {login_url}")
            self.driver.get(login_url)
            
            # Wait for manual login
            logger.info("Please log in to ServiceNow in the Chrome window...")
            logger.info("You have 5 minutes to complete the login")
            
            # Wait for successful login (check for dashboard or home page)
            start_time = time.time()
            timeout = 300  # 5 minutes
            logged_in = False
            
            while time.time() - start_time < timeout:
                try:
                    current_url = self.driver.current_url.lower()
                    if any(term in current_url for term in ['navpage', 'home', 'dashboard', 'ns_user_profile']):
                        logger.info("Detected successful login!")
                        logged_in = True
                        break
                    time.sleep(2)
                except Exception as e:
                    logger.warning(f"Error checking login status: {str(e)}")
                    time.sleep(2)
            
            if not logged_in:
                logger.error("Login timeout: Please try again")
                return False
            
            self.authenticated = True
            logger.info("Successfully authenticated with ServiceNow")
            return True
            
        except Exception as e:
            logger.error(f"Error during authentication: {str(e)}")
            if hasattr(self, 'driver') and self.driver:
                self.close()
            return False

    def close(self):
        """Close the WebDriver and clean up resources."""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
                self.authenticated = False
            except Exception as e:
                logger.error(f"Error closing WebDriver: {str(e)}")

    def __del__(self):
        """Ensure browser is closed when the object is destroyed."""
        self.close()

    def get_case_task_links(self, list_url: str) -> List[str]:
        """
        Get all case task links from the list page.
        
        Args:
            list_url: URL of the case tasks list page
            
        Returns:
            List of case task URLs
        """
        if not self.authenticated:
            if not self._authenticate_with_browser():
                raise Exception("Authentication failed. Cannot proceed.")
        
        try:
            logger.info(f"Navigating to case tasks list: {list_url}")
            self.driver.get(list_url)
            
            # Wait for the case list to load
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='sn_customerservice_task.do?sys_id=']"))
            )
            
            # Find all case task links
            case_links = []
            for link in self.driver.find_elements(By.CSS_SELECTOR, "a[href*='sn_customerservice_task.do?sys_id=']"):
                href = link.get_attribute('href')
                if href and href not in case_links:
                    case_links.append(href)
            
            logger.info(f"Found {len(case_links)} case tasks")
            return case_links
            
        except Exception as e:
            logger.error(f"Error getting case task links: {str(e)}")
            return []
    
    def extract_case_data(self, case_url: str) -> Optional[Dict]:
        """
        Extract case data from a case task page.
        
        Args:
            case_url: URL of the case task page
            
        Returns:
            Dictionary containing extracted case data or None if extraction failed
        """
        if not self.authenticated:
            if not self._authenticate_with_browser():
                raise Exception("Authentication failed. Cannot proceed.")
        
        try:
            logger.info(f"Extracting case data from: {case_url}")
            self.driver.get(case_url)
            
            # Wait for the page to load
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            
            # Wait for the main form to load
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".form-group, .form-field, [role='main']"))
            )
            
            # Get current timestamp in ISO format
            extracted_at = datetime.utcnow().isoformat()
            
            # Extract case data with more robust selectors
            case_data = {
                'case_number': self._safe_find_element('input[id*="number"][readonly]'),
                'parent_case': self._safe_find_element('input[id="sn_customerservice_task.parent_label"]'),
                'subject': self._safe_find_element('input[id*="short_description"], input[id*="subject"]'),
                'description': self._extract_description(),
                'close_notes': self._find_close_notes(),
                'metadata': {
                    'source_url': case_url,
                    'extracted_at': extracted_at
                }
            }
            
            # Generate a temporary case number if none exists
            if not case_data['case_number']:
                case_number = f"TEMP-{generate_id()[:8]}"
                case_data['case_number'] = case_number
            else:
                case_number = case_data['case_number']
            
            logger.info(f"Extracted case: {case_data.get('case_number')} - {case_data.get('subject')}")
            return case_data
            
        except Exception as e:
            logger.error(f"Error extracting case data from {case_url}: {str(e)}")
            return None
    
    def _extract_description(self) -> str:
        """
        Extract the case description using multiple fallback strategies.
        
        Returns:
            Extracted description text or empty string if not found
        """
        selectors = [
            'textarea#sn_customerservice_task\\.description',
            'div[data-stream-text-value*="description"]',
            'div[class*="description"], div[data-element*="description"]',
            'div[role="textbox"], div[contenteditable="true"]',
            'div[class*="form-field"], div[class*="form-group"]',
        ]
        
        for selector in selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    try:
                        # Try to get the value attribute first
                        value = element.get_attribute('value')
                        if value and value.strip():
                            return value.strip()
                        
                        # Then try text content
                        text = element.text
                        if text and text.strip():
                            return text.strip()
                        
                        # Finally try innerHTML
                        html = element.get_attribute('innerHTML')
                        if html and html.strip():
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(html, 'html.parser')
                            text = soup.get_text(separator='\n', strip=True)
                            if text and text.strip():
                                return text.strip()
                                
                    except Exception:
                        continue
                        
            except Exception:
                continue
                
        logger.warning("Could not extract description using any selector")
        return ''
    
    def _find_close_notes(self) -> str:
        """
        Find and return the close notes text from the textarea.
        
        Returns:
            The close notes text or empty string if not found
        """
        try:
            from selenium.webdriver.common.by import By
            
            # Try direct ID match first
            element = self.driver.find_element(
                By.CSS_SELECTOR, 
                'textarea#sn_customerservice_task\\.close_notes'
            )
            
            # Get the value attribute (for textarea/input)
            value = element.get_attribute('value')
            if value is not None:
                return value.strip()
                
            # Fallback to text content if value is empty
            text = element.text
            if text and text.strip():
                return text.strip()
                
            return ''
            
        except Exception as e:
            logger.debug(f"Error finding close notes: {str(e)}")
            return ''
    
    def _safe_find_element(self, selector: str) -> str:
        """
        Safely find and return the value or text of an element.
        
        Args:
            selector: CSS selector for the element
            
        Returns:
            Element value/text or empty string if not found
        """
        try:
            from selenium.webdriver.common.by import By
            
            # Handle dots in IDs by escaping them
            if '#' in selector and '.' in selector.split('#')[1]:
                parts = selector.split('#')
                parts[1] = parts[1].replace('.', '\\.')
                selector = '#'.join(parts)
            
            element = self.driver.find_element(By.CSS_SELECTOR, selector)
            
            # Try to get the value attribute first (for input/textarea)
            value = element.get_attribute('value')
            if value is not None:
                return value.strip()
                
            # If no value, try to get the text content
            text = element.text
            if text:
                return text.strip()
                
            # If still no content, try innerHTML as last resort
            html = element.get_attribute('innerHTML')
            if html:
                from bs4 import BeautifulSoup
                return BeautifulSoup(html, 'html.parser').get_text().strip()
                
            return ''
            
        except Exception as e:
            logger.debug(f"Element not found or error: {selector} - {str(e)}")
            return ''
    
    def process_case_task_list(self, list_url: str) -> Generator[Dict, None, None]:
        """
        Process a list of case tasks and yield extracted case data.
        
        Args:
            list_url: URL of the case tasks list page
            
        Yields:
            Dictionaries containing extracted case data
        """
        # Get all case task links
        case_links = self.get_case_task_links(list_url)
        
        # Process each case
        for link in case_links:
            case_data = self.extract_case_data(link)
            if case_data:
                yield case_data
                
    def crawl_servicenow_cases(self, list_url: str) -> Generator[Dict, None, None]:
        """
        Crawl ServiceNow case tasks starting from the list URL.
        
        Args:
            list_url: URL of the case tasks list page
            
        Yields:
            Dictionaries containing extracted case data
        """
        # Track visited pages to prevent duplicates
        visited_pages = set()
        current_url = list_url
        
        # Authenticate if not already authenticated
        if not self.authenticated:
            if not self._authenticate_with_browser():
                logger.error("Authentication failed. Cannot proceed with crawling.")
                return
        
        page_count = 0
        
        while current_url and page_count < self.max_pages:
            # Skip if we've already visited this page
            if current_url in visited_pages:
                logger.debug(f"Skipping already visited page: {current_url}")
                break
                
            # Mark current page as visited
            visited_pages.add(current_url)
            logger.info(f"Processing page {page_count + 1}: {current_url}")
            
            # Process the current page
            yield from self.process_case_task_list(current_url)
            page_count += 1
            
            # Handle pagination
            try:
                next_button = self.driver.find_element(
                    By.CSS_SELECTOR,
                    'button[data-original-title^="Next"], a[data-original-title^="Next"]'
                )
                
                # Check if next button is disabled
                if 'disabled' in next_button.get_attribute('class'):
                    logger.debug("Reached the last page")
                    break
                    
                # Store current URL before clicking to detect page change
                previous_url = self.driver.current_url
                
                # Click the next button
                next_button.click()
                
                # Wait for the page to change
                from selenium.webdriver.support.ui import WebDriverWait
                from selenium.webdriver.support import expected_conditions as EC
                
                WebDriverWait(self.driver, 20).until(
                    lambda d: d.current_url != previous_url
                )
                
                # Update current URL for the next iteration
                current_url = self.driver.current_url
                
            except Exception as e:
                logger.debug(f"No more pages or error during pagination: {str(e)}")
                break


class CaseImporter:
    def __init__(self, case_service, debug: bool = False):
        """
        Initialize the CaseImporter.
        
        Args:
            case_service: Instance of CaseService for saving cases
            debug: Enable debug mode for more verbose logging
        """
        self.case_service = case_service
        self.debug = debug
        self.stats = {
            'total_imported': 0,
            'total_processed': 0,
            'errors': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None,
            'processed_case_numbers': set()  # Track processed case numbers
        }
    
    def import_servicenow_cases(self, list_url: str, max_pages: int = 10) -> Dict[str, Any]:
        """
        Import cases from a ServiceNow case list URL.
        
        Args:
            list_url: URL of the ServiceNow case list page
            max_pages: Maximum number of pages to process
            
        Returns:
            Dictionary with import statistics and results
        """
        self.stats['start_time'] = datetime.utcnow().isoformat()
        logger.info(f"Starting import of up to {max_pages} pages from {list_url}")
        
        try:
            # Initialize crawler with debug options
            crawler = WebCrawler(
                base_url=list_url,
                max_pages=max_pages,
                debug=self.debug
            )
            
            # Process each case
            for case_data in crawler.crawl_servicenow_cases(list_url):
                if not case_data or not isinstance(case_data, dict):
                    logger.warning("Skipping invalid case data")
                    continue
                    
                case_number = case_data.get('case_number')
                if not case_number:
                    logger.warning("Skipping case with missing case number")
                    self.stats['skipped'] += 1
                    continue
                    
                self.stats['total_processed'] += 1
                
                # Skip if we've already processed this case
                if case_number in self.stats['processed_case_numbers']:
                    if self.debug:
                        logger.debug(f"Skipping duplicate case: {case_number}")
                    self.stats['skipped'] += 1
                    continue
                
                try:
                    # Process the case
                    case = self.case_service.create_case(
                        subject=case_data.get('subject', 'Untitled Case'),
                        description=case_data.get('description', ''),
                        case_number=case_number,
                        parent_case=case_data.get('parent_case'),
                        close_notes=case_data.get('close_notes'),
                        tags=case_data.get('tags', []),
                        **case_data.get('metadata', {})
                    )
                    
                    if case:
                        self.stats['total_imported'] += 1
                        self.stats['processed_case_numbers'].add(case_number)
                        if self.debug or self.stats['total_imported'] % 10 == 0:
                            logger.info(f"Imported {self.stats['total_imported']} cases... (Latest: {case_number})")
                    else:
                        self.stats['errors'] += 1
                        logger.error(f"Failed to import case: {case_number}")
                        
                except ValueError as e:
                    # Handle case already exists error
                    if "already exists" in str(e):
                        if self.debug:
                            logger.debug(f"Case {case_number} already exists, skipping...")
                        self.stats['skipped'] += 1
                        self.stats['processed_case_numbers'].add(case_number)
                    else:
                        self.stats['errors'] += 1
                        logger.error(f"Error processing case {case_number}: {str(e)}")
                        if self.debug:
                            logger.exception("Detailed error:")
                except Exception as e:
                    self.stats['errors'] += 1
                    logger.error(f"Unexpected error processing case {case_number}: {str(e)}")
                    if self.debug:
                        logger.exception("Detailed error:")
            
            # Update end time and return results
            self.stats['end_time'] = datetime.utcnow().isoformat()
            
            # Calculate duration
            start_time = datetime.fromisoformat(self.stats['start_time'])
            end_time = datetime.fromisoformat(self.stats['end_time'])
            duration = end_time - start_time
            
            result = {
                'status': 'completed',
                'stats': {
                    'total_imported': self.stats['total_imported'],
                    'total_processed': self.stats['total_processed'],
                    'errors': self.stats['errors'],
                    'skipped': self.stats['skipped'],
                    'start_time': self.stats['start_time'],
                    'end_time': self.stats['end_time'],
                    'duration_seconds': duration.total_seconds()
                },
                'message': f"Successfully imported {self.stats['total_imported']} " \
                          f"of {self.stats['total_processed']} cases in {duration}"
            }
            
            if self.stats['errors'] > 0:
                result['status'] = 'completed_with_errors'
                result['message'] += f" with {self.stats['errors']} errors"
            
            return result
            
        except Exception as e:
            self.stats['end_time'] = datetime.utcnow().isoformat()
            
            # Calculate duration
            start_time = datetime.fromisoformat(self.stats['start_time'])
            end_time = datetime.fromisoformat(self.stats['end_time'])
            duration = end_time - start_time
            
            return {
                'status': 'failed',
                'error': str(e),
                'stats': {
                    'total_imported': self.stats['total_imported'],
                    'total_processed': self.stats['total_processed'],
                    'errors': self.stats['errors'],
                    'skipped': self.stats['skipped'],
                    'start_time': self.stats['start_time'],
                    'end_time': self.stats['end_time'],
                    'duration_seconds': duration.total_seconds()
                },
                'message': f"Import failed after {duration}: {str(e)}"
            }
    
    def import_from_website(self, base_url: str, max_pages: int = 20) -> Dict[str, Any]:
        """
        Generic website importer (legacy method).
        For ServiceNow imports, use import_servicenow_cases instead.
        """
        logger.warning("import_from_website is deprecated. Use import_servicenow_cases for ServiceNow imports.")
        
        try:
            crawler = WebCrawler(base_url=base_url, max_pages=max_pages, debug=self.debug)
            
            for case_data in crawler.crawl():
                if not case_data or not isinstance(case_data, dict):
                    continue
                    
                case_number = case_data.get('case_number')
                if not case_number:
                    case_number = f"temp_{uuid.uuid4().hex[:8]}"  # Generate a temporary ID if none exists
                    case_data['case_number'] = case_number
                
                self.stats['total_processed'] += 1
                
                # Skip if we've already processed this case
                if case_number in self.stats['processed_case_numbers']:
                    if self.debug:
                        logger.debug(f"Skipping duplicate case: {case_number}")
                    self.stats['skipped'] += 1
                    continue
                
                try:
                    # Create the case
                    case = self.case_service.create_case(
                        subject=case_data.get('title', 'Untitled Case'),
                        description=case_data.get('description', ''),
                        case_number=case_number,
                        tags=case_data.get('tags', []),
                        **case_data.get('metadata', {})
                    )
                    
                    if case:
                        self.stats['total_imported'] += 1
                        self.stats['processed_case_numbers'].add(case_number)
                        if self.debug or self.stats['total_imported'] % 10 == 0:
                            logger.info(f"Imported {self.stats['total_imported']} cases... (Latest: {case_number})")
                
                except ValueError as e:
                    if "already exists" in str(e):
                        if self.debug:
                            logger.debug(f"Case {case_number} already exists, skipping...")
                        self.stats['skipped'] += 1
                        self.stats['processed_case_numbers'].add(case_number)
                    else:
                        self.stats['errors'] += 1
                        logger.error(f"Error processing case {case_number}: {str(e)}")
                except Exception as e:
                    self.stats['errors'] += 1
                    logger.error(f"Unexpected error processing case {case_number}: {str(e)}")
            
            # Prepare results
            result = {
                'status': 'completed',
                'stats': {
                    'total_imported': self.stats['total_imported'],
                    'total_processed': self.stats['total_processed'],
                    'errors': self.stats['errors'],
                    'skipped': self.stats['skipped']
                },
                'message': f"Successfully imported {self.stats['total_imported']} cases"
            }
            
            if self.stats['errors'] > 0:
                result['status'] = 'completed_with_errors'
                
            return result
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'stats': {
                    'total_imported': self.stats['total_imported'],
                    'total_processed': self.stats['total_processed'],
                    'errors': self.stats['errors'],
                    'skipped': self.stats['skipped']
                },
                'message': 'Failed to import cases'
            }
