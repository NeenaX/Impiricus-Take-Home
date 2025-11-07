import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import re
import pandas as pd

class BrownBulletinScraper:
    """
    Scrapes the Brown Bulletin website by searching the links on a page and finding the
    .courseblock tags
    """
    def __init__(self):
        self.base_url = "https://bulletin.brown.edu"
        self.session = requests.Session()
        self.visited_urls = set()
        self.all_course_blocks = []
        
    def is_valid_url(self, url):
        """Check if URL is valid to scrape"""
        parsed = urlparse(url)
        # Only follow links within the brown.edu domain
        if not parsed.netloc.endswith('brown.edu'):
            return False
        
        # Avoid non-HTML content
        if any(parsed.path.endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.doc', '.docx']):
            return False
        return True
    
    def extract_links(self, soup, current_url):
        """Extract all valid links from a page"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(current_url, href)
            
            if self.is_valid_url(full_url) and full_url not in self.visited_urls:
                links.append(full_url)
                
        return links
    
    def extract_course_blocks(self, soup: BeautifulSoup, url: str):
        """Extract course blocks from the current page"""
        course_blocks = []
        
        # Find all courseblock tags
        blocks = soup.select('.courseblock')
        if blocks:
            print(f"Found {len(blocks)} course blocks with selector '{'.courseblock'}' on {url}")
            for block in blocks:
                course_info = self.parse_course_block(block, url)
                if course_info:
                    course_blocks.append(course_info)
        
        return course_blocks
    
    def parse_course_block(self, block, source_url):
        """Parse individual course block to extract information"""
        try:
            # Get the text content
            text = block.get_text(strip=True)
            
            # Extract course code (pattern like "CSCI 0150", "MATH 0100")
            code_match = re.search(r'([A-Z]{2,8}\s+\d{4}[A-Z]?)', text)
            course_code = code_match.group(1) if code_match else "UNKNOWN"
            
            # Extract everything after the course code
            parts = text[code_match.end():].strip()
            # Clean up the second part
            parts = re.sub(r'^[-:\s]*', '', parts)
            parts = parts.split('.', 2)
            
            # Title, decription, and code seperated by '.'
            title = parts[1]
            description = parts[2]
            
            # Use part of text as description in case of no description
            if not description and len(text) > len(course_code) + 20:
                description = text[len(course_code):200].strip()
            
            return {
                'course_code': course_code,
                'title': title,
                'description': description,
                'source': 'Bulletin',
            }
            
        except Exception as e:
            print(f"Error parsing course block: {e}")
            return None
    
    def scrape_site(self, max_pages: int = 0, max_depth: int = 3):
        """Scrape the site by following links (BFS)"""
        queue = deque([(self.base_url, 0)])  # (url, depth)
        self.visited_urls.add(self.base_url)
        
        page_count = 0
        
        while queue and page_count < max_pages:
            current_url, depth = queue.popleft()
            
            if depth > max_depth:
                continue                
            print(f"Scraping ({depth}): {current_url}")
            
            try:
                response = self.session.get(current_url)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract course blocks from current page
                course_blocks = self.extract_course_blocks(soup, current_url)
                self.all_course_blocks.extend(course_blocks)
                
                # Look for more links on current page
                if depth < max_depth:
                    new_links = self.extract_links(soup, current_url)
                    
                    for link in new_links:
                        if link not in self.visited_urls:
                            self.visited_urls.add(link)
                            queue.append((link, depth + 1))
                
                page_count += 1
                
            except Exception as e:
                print(f"Error scraping {current_url}: {e}")
                continue
        
        print(f"Scraping complete. Visited {page_count} pages, found {len(self.all_course_blocks)} course blocks.")
    
    def save_results(self, filename: str ='../data/bulletin_courses.csv'):
        """Save results to CSV"""
        if not self.all_course_blocks:
            print("No course blocks found to save.")
            return
        
        df = pd.DataFrame(self.all_course_blocks)
        
        # Remove duplicates based on course code
        df = df.drop_duplicates(subset=['course_code'], keep='first')
        
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} unique courses to {filename}")
        return df

def main():
    """Main function to run the link-following scraper"""
    scraper = BrownBulletinScraper()
    
    print("Starting link-following scrape from: https://bulletin.brown.edu/")
    scraper.scrape_site(max_pages=100, max_depth=2)
    scraper.save_results()

if __name__ == "__main__":
    main()