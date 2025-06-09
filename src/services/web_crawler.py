import re
import logging
from typing import List, Dict, Optional, Set, Generator
from urllib.parse import urljoin, urlparse, urlunparse
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from trafilatura import extract
from pathlib import Path
import json
from datetime import datetime

# Update imports to use absolute paths
from src.models.embeddings import EmbeddingService
from src.config.settings import settings
from src.utils.helpers import generate_id, format_timestamp

logger = logging.getLogger(__name__)

class WebCrawler:
    def __init__(self, base_url: str, max_pages: int = 50):
        """
        Initialize the web crawler.
        
        Args:
            base_url: The base URL to start crawling from
            max_pages: Maximum number of pages to crawl
        """
        self.base_url = base_url
        self.max_pages = max_pages
        self.visited_urls: Set[str] = set()
        self.to_visit: List[str] = [base_url]
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and query parameters."""
        parsed = urlparse(url)
        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            '',  # params
            '',  # query
            ''   # fragment
        )).rstrip('/')
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and belongs to the same domain."""
        if not url or url.startswith('mailto:') or url.startswith('tel:'):
            return False
            
        parsed = urlparse(url)
        base_parsed = urlparse(self.base_url)
        
        # Only crawl same domain
        if parsed.netloc and parsed.netloc != base_parsed.netloc:
            return False
            
        # Only crawl http/https
        if parsed.scheme not in ('http', 'https', ''):
            return False
            
        # Ignore common file types
        if any(parsed.path.lower().endswith(ext) for ext in 
              ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.tar.gz', '.exe', '.dmg']):
            return False
            
        return True
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all valid links from a page."""
        links = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Handle relative URLs
            full_url = urljoin(base_url, href)
            normalized = self.normalize_url(full_url)
            
            if self.is_valid_url(normalized) and normalized not in self.visited_urls:
                links.add(normalized)
        
        return list(links)
    
    def extract_case_data(self, url: str, html: str) -> Optional[Dict]:
        """Extract case data from a web page."""
        try:
            # Try to extract main content using trafilatura
            text_content = extract(html, include_comments=False, include_tables=False)
            
            if not text_content:
                return None
                
            # Use newspaper3k for metadata
            article = Article(url)
            article.set_html(html)
            article.parse()
            
            title = article.title or "Untitled Case"
            
            # Try to find case ID in URL or content
            case_id_match = re.search(r'case[_-]?(\d+)', url.lower())
            case_id = f"imported_{case_id_match.group(1)}" if case_id_match else generate_id('imported')
            
            return {
                'case_id': case_id,
                'title': title,
                'description': text_content[:5000],  # Limit description length
                'source_url': url,
                'created_at': format_timestamp(),
                'updated_at': format_timestamp(),
                'metadata': {
                    'source': 'web_crawler',
                    'publish_date': str(article.publish_date) if article.publish_date else None,
                    'authors': article.authors,
                    'keywords': article.meta_keywords,
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting case data from {url}: {str(e)}")
            return None
    
    def crawl(self) -> Generator[Dict, None, None]:
        """Crawl the website and yield case data from each page."""
        while self.to_visit and len(self.visited_urls) < self.max_pages:
            current_url = self.to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
                
            try:
                logger.info(f"Crawling: {current_url}")
                response = self.session.get(current_url, timeout=10)
                response.raise_for_status()
                
                if 'text/html' not in response.headers.get('Content-Type', ''):
                    continue
                
                soup = BeautifulSoup(response.text, 'lxml')
                
                # Extract links for further crawling
                new_links = self.extract_links(soup, current_url)
                self.to_visit.extend(link for link in new_links if link not in self.visited_urls)
                
                # Extract case data
                case_data = self.extract_case_data(current_url, response.text)
                if case_data:
                    yield case_data
                
            except Exception as e:
                logger.error(f"Error crawling {current_url}: {str(e)}")
            finally:
                self.visited_urls.add(current_url)
                
            # Be nice to the server
            import time
            time.sleep(1)


class CaseImporter:
    def __init__(self, case_service):
        self.case_service = case_service
    
    def import_from_website(self, base_url: str, max_pages: int = 20) -> Dict:
        """Import cases from a website."""
        crawler = WebCrawler(base_url, max_pages)
        imported_count = 0
        
        for case_data in crawler.crawl():
            try:
                # Check if case already exists
                existing_case = self.case_service.get_case(case_data['case_id'])
                if existing_case:
                    logger.info(f"Case {case_data['case_id']} already exists, skipping...")
                    continue
                
                # Create the case
                self.case_service.create_case(
                    title=case_data['title'],
                    description=case_data['description'],
                    tags=[],
                    **case_data.get('metadata', {})
                )
                imported_count += 1
                logger.info(f"Imported case: {case_data['title']}")
                
            except Exception as e:
                logger.error(f"Error importing case {case_data.get('case_id')}: {str(e)}")
        
        return {
            "status": "completed",
            "imported_count": imported_count,
            "total_visited": len(crawler.visited_urls)
        }
