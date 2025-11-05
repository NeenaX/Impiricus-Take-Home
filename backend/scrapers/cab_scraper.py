import grequests
import requests
import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
from tqdm import tqdm
import json

class CABScraper:
    """
    CAB scraper class with code that calls the API taken from 
    https://github.com/karipov/brunosearch/blob/main/scraper/src/scrape_courses.py
    I could not find documentation for the API to get the courses on my own
    """

    def __init__(self):
        self.base_url = "https://cab.brown.edu"
        self.course_search_url = self.base_url + "/api/?page=fose&route=search&is_ind_study=N&is_canc=N"
        self.detail_search_url = self.base_url + "/api/?page=fose&route=details"
        
    def construct_db_string(self, season: str, year: str):
        """
        Converts a school year and season to the corresponding CAB database string, e.g.
            spring 2023 => 202220
            fall 2022 => 202210
        """
        database_year = int(year) if season == "fall" else int(year) - 1
        suffix = "20" if season == "spring" else "10"

        return str(database_year) + suffix


    def get_course_metadata(self, season: str, year: str) -> list:
        """
        Fetches course metadata from CAB for a given term
        """
        print(f"[INFO] Requesting course metadata for {season}, {year}...")
        cab_search_payload = {
            "other": { "srcdb": self.construct_db_string(season, year)},
            "criteria": [
                { "field": "is_ind_study", "value": "N" },
                { "field": "is_canc", "value": "N" },
            ],
        }
        return requests.post(self.course_search_url, json=cab_search_payload).json()["results"]



    def parallel_get_course_details(self, courses: list) -> dict:
        """
        Fetches all course details from CAB in parallel
        """
        print(f"[INFO] Getting details for {len(courses)} courses in parallel...")
        get_details_payload = lambda r: {
            "group": f"code:{r['code']}",
            "key": f"crn:{r['crn']}",
            "srcdb": r["srcdb"],
            "matched": f"crn:{r['crn']}",
        }
        rs = (grequests.post(self.detail_search_url, json=get_details_payload(r)) for r in courses)
        raw_parallel = grequests.map(rs)
        
        # construct a dictionary of course details by course code
        details = {}
        for response in raw_parallel:
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                print(f"[ERROR] Error fetching course details: {e}")
                continue
            
            response_json = response.json()
            details[response_json["code"]] = response_json
        
        return details

    def clean_description(self, description: str) -> str:
        """
        Cleans the description of each course by removing HTML tags
        """
        return BeautifulSoup(description, 'html.parser').get_text()

    
def main(): 
    """Main function to run the CABScraper"""   
    tqdm.pandas()
    scraper = CABScraper()
    f25 = scraper.get_course_metadata("fall", "2025") # Scraping specifically for fall 2025
    rows =  []
    details = scraper.parallel_get_course_details(f25)

    for k in tqdm(details.keys()):
        a = details[k]
        code = a['code']
        title = a['title']
        instructor = a['instructordetail_html']
        prereqs = a['suggestedcourses_html']
        department = a['code'].split(' ')[0] # dept is the first section of the class code
        source = 'CAB'
        meetings = a['meeting_html']


        # Clean HTML from all HTML fields
        description = scraper.clean_description(a['description'])
        prereqs = scraper.clean_description(prereqs)
        meetings = scraper.clean_description(meetings)
        instructor = scraper.clean_description(instructor)
        
        rows.append({
            'course_code': code,
            'title': title,
            'instructor': instructor,
            'meeting_times': meetings,
            'prerequisites': prereqs,
            'department': department,
            'description': description,
            'source': source,
        })

    json_str = json.dumps(rows, indent=6)
    with open("../data/cab_courses.json", "w") as f:
        f.write(json_str)

if __name__ == "__main__":
    main()