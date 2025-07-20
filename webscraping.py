import pandas as pd
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import os
import json
from bs4 import BeautifulSoup
import re
import random
from urllib.parse import urljoin, urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealFashionScraper:
    def __init__(self, headless=True, delay_range=(1, 3)):
        self.headless = headless
        self.delay_range = delay_range
        self.scraped_data = []
        self.session = requests.Session()
        
        # Set up session headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        self.driver = None
        self.init_driver()
        
    def init_driver(self):
        """Initialize Chrome WebDriver with anti-detection measures"""
        try:
            chrome_options = Options()
            
            if self.headless:
                chrome_options.add_argument('--headless')
            
            # Anti-detection measures
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument('--disable-web-security')
            chrome_options.add_argument('--allow-running-insecure-content')
            chrome_options.add_argument('--disable-features=VizDisplayCompositor')
            
            # Set window size
            chrome_options.add_argument('--window-size=1920,1080')
            
            # User agent
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Execute script to hide webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            logger.info("WebDriver initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise
    
    def random_delay(self):
        """Add random delay between requests"""
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)
    
    def safe_find_element(self, by, value, timeout=10):
        """Safely find element with timeout"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return element
        except TimeoutException:
            return None
    
    def safe_find_elements(self, by, value, timeout=10):
        """Safely find elements with timeout"""
        try:
            elements = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_all_elements_located((by, value))
            )
            return elements
        except TimeoutException:
            return []
    
    def scrape_amazon_fashion(self, search_term="mens shirts", max_pages=3):
        """Scrape Amazon fashion products"""
        logger.info(f"Starting Amazon scraping for: {search_term}")
        
        base_url = "https://www.amazon.in"
        search_url = f"{base_url}/s?k={search_term.replace(' ', '+')}&rh=n%3A1571271031"
        
        try:
            for page in range(1, max_pages + 1):
                url = f"{search_url}&page={page}"
                logger.info(f"Scraping Amazon page {page}: {url}")
                
                self.driver.get(url)
                self.random_delay()
                
                # Wait for products to load
                products = self.safe_find_elements(By.CSS_SELECTOR, "[data-component-type='s-search-result']")
                
                if not products:
                    logger.warning(f"No products found on page {page}")
                    continue
                
                for product in products:
                    try:
                        # Extract product name
                        name_elem = product.find_element(By.CSS_SELECTOR, "h2 a span")
                        name = name_elem.text.strip() if name_elem else "N/A"
                        
                        # Extract price
                        price_elem = product.find_elements(By.CSS_SELECTOR, ".a-price-whole")
                        price = f"₹{price_elem[0].text}" if price_elem else "N/A"
                        
                        # Extract rating
                        rating_elem = product.find_elements(By.CSS_SELECTOR, ".a-icon-alt")
                        rating = rating_elem[0].get_attribute("textContent") if rating_elem else "N/A"
                        
                        # Extract image URL
                        img_elem = product.find_elements(By.CSS_SELECTOR, "img.s-image")
                        image_url = img_elem[0].get_attribute("src") if img_elem else "N/A"
                        
                        # Extract product URL
                        link_elem = product.find_elements(By.CSS_SELECTOR, "h2 a")
                        product_url = urljoin(base_url, link_elem[0].get_attribute("href")) if link_elem else "N/A"
                        
                        if name != "N/A":
                            self.scraped_data.append({
                                'name': name,
                                'price': price,
                                'rating': rating,
                                'image_url': image_url,
                                'product_url': product_url,
                                'source': 'Amazon'
                            })
                            
                    except Exception as e:
                        logger.error(f"Error extracting product data: {e}")
                        continue
                
                logger.info(f"Amazon page {page} completed. Total products so far: {len(self.scraped_data)}")
                
        except Exception as e:
            logger.error(f"Error scraping Amazon: {e}")
    
    def scrape_flipkart_fashion(self, search_term="mens shirts", max_pages=3):
        """Scrape Flipkart fashion products"""
        logger.info(f"Starting Flipkart scraping for: {search_term}")
        
        base_url = "https://www.flipkart.com"
        search_url = f"{base_url}/search?q={search_term.replace(' ', '%20')}"
        
        try:
            for page in range(1, max_pages + 1):
                url = f"{search_url}&page={page}"
                logger.info(f"Scraping Flipkart page {page}: {url}")
                
                self.driver.get(url)
                self.random_delay()
                
                # Wait for products to load
                products = self.safe_find_elements(By.CSS_SELECTOR, "div[data-id]")
                
                if not products:
                    logger.warning(f"No products found on page {page}")
                    continue
                
                for product in products:
                    try:
                        # Extract product name
                        name_elem = product.find_elements(By.CSS_SELECTOR, "a[title]")
                        name = name_elem[0].get_attribute("title") if name_elem else "N/A"
                        
                        # Extract price
                        price_elem = product.find_elements(By.CSS_SELECTOR, "div._30jeq3, div._1_WHN1")
                        price = price_elem[0].text if price_elem else "N/A"
                        
                        # Extract rating
                        rating_elem = product.find_elements(By.CSS_SELECTOR, "div._3LWZlK, div._3LWZlK span")
                        rating = rating_elem[0].text if rating_elem else "N/A"
                        
                        # Extract image URL
                        img_elem = product.find_elements(By.CSS_SELECTOR, "img")
                        image_url = img_elem[0].get_attribute("src") if img_elem else "N/A"
                        
                        # Extract product URL
                        link_elem = product.find_elements(By.CSS_SELECTOR, "a[title]")
                        product_url = urljoin(base_url, link_elem[0].get_attribute("href")) if link_elem else "N/A"
                        
                        if name != "N/A":
                            self.scraped_data.append({
                                'name': name,
                                'price': price,
                                'rating': rating,
                                'image_url': image_url,
                                'product_url': product_url,
                                'source': 'Flipkart'
                            })
                            
                    except Exception as e:
                        logger.error(f"Error extracting product data: {e}")
                        continue
                
                logger.info(f"Flipkart page {page} completed. Total products so far: {len(self.scraped_data)}")
                
        except Exception as e:
            logger.error(f"Error scraping Flipkart: {e}")
    
    def scrape_myntra_fashion(self, search_term="mens shirts", max_pages=3):
        """Scrape Myntra fashion products"""
        logger.info(f"Starting Myntra scraping for: {search_term}")
        
        base_url = "https://www.myntra.com"
        search_url = f"{base_url}/{search_term.replace(' ', '-')}"
        
        try:
            self.driver.get(search_url)
            self.random_delay()
            
            # Wait for products to load
            products = self.safe_find_elements(By.CSS_SELECTOR, "li.product-base")
            
            if not products:
                logger.warning("No products found on Myntra")
                return
                
            for product in products:
                try:
                    # Extract product name
                    name_elem = product.find_elements(By.CSS_SELECTOR, "h3.product-brand, h4.product-product")
                    name = name_elem[0].text if name_elem else "N/A"
                    
                    # Extract price
                    price_elem = product.find_elements(By.CSS_SELECTOR, "span.product-discountedPrice")
                    price = price_elem[0].text if price_elem else "N/A"
                    
                    # Extract rating
                    rating_elem = product.find_elements(By.CSS_SELECTOR, "span.product-rating")
                    rating = rating_elem[0].text if rating_elem else "N/A"
                    
                    # Extract image URL
                    img_elem = product.find_elements(By.CSS_SELECTOR, "img.product-image")
                    image_url = img_elem[0].get_attribute("src") if img_elem else "N/A"
                    
                    # Extract product URL
                    link_elem = product.find_elements(By.CSS_SELECTOR, "a")
                    product_url = urljoin(base_url, link_elem[0].get_attribute("href")) if link_elem else "N/A"
                    
                    if name != "N/A":
                        self.scraped_data.append({
                            'name': name,
                            'price': price,
                            'rating': rating,
                            'image_url': image_url,
                            'product_url': product_url,
                            'source': 'Myntra'
                        })
                        
                except Exception as e:
                    logger.error(f"Error extracting product data: {e}")
                    continue
            
            logger.info(f"Myntra scraping completed. Total products so far: {len(self.scraped_data)}")
            
        except Exception as e:
            logger.error(f"Error scraping Myntra: {e}")
    
    def scrape_vogue_fashion(self, max_articles=20):
        """Scrape Vogue India fashion articles"""
        logger.info("Starting Vogue India scraping")
        
        base_url = "https://www.vogue.in"
        fashion_url = f"{base_url}/fashion"
        
        try:
            self.driver.get(fashion_url)
            self.random_delay()
            
            # Wait for articles to load
            articles = self.safe_find_elements(By.CSS_SELECTOR, "article, .article-item, .story-card")
            
            if not articles:
                logger.warning("No articles found on Vogue")
                return
            
            count = 0
            for article in articles[:max_articles]:
                try:
                    # Extract article title
                    title_elem = article.find_elements(By.CSS_SELECTOR, "h1, h2, h3, .headline, .title")
                    title = title_elem[0].text.strip() if title_elem else "N/A"
                    
                    # Extract image URL
                    img_elem = article.find_elements(By.CSS_SELECTOR, "img")
                    image_url = img_elem[0].get_attribute("src") if img_elem else "N/A"
                    
                    # Extract article URL
                    link_elem = article.find_elements(By.CSS_SELECTOR, "a")
                    article_url = urljoin(base_url, link_elem[0].get_attribute("href")) if link_elem else "N/A"
                    
                    if title != "N/A" and "fashion" in title.lower():
                        self.scraped_data.append({
                            'name': title,
                            'price': 'N/A',
                            'rating': 'N/A',
                            'image_url': image_url,
                            'product_url': article_url,
                            'source': 'Vogue'
                        })
                        count += 1
                        
                except Exception as e:
                    logger.error(f"Error extracting article data: {e}")
                    continue
            
            logger.info(f"Vogue scraping completed. Scraped {count} articles")
            
        except Exception as e:
            logger.error(f"Error scraping Vogue: {e}")
    
    def scrape_h_and_m_fashion(self, search_term="mens shirts", max_pages=2):
        """Scrape H&M fashion products"""
        logger.info(f"Starting H&M scraping for: {search_term}")
        
        base_url = "https://www2.hm.com"
        search_url = f"{base_url}/en_in/search-results.html?q={search_term.replace(' ', '%20')}"
        
        try:
            self.driver.get(search_url)
            self.random_delay()
            
            # Wait for products to load
            products = self.safe_find_elements(By.CSS_SELECTOR, "article.item")
            
            if not products:
                logger.warning("No products found on H&M")
                return
            
            for product in products:
                try:
                    # Extract product name
                    name_elem = product.find_elements(By.CSS_SELECTOR, "h3, .item-heading")
                    name = name_elem[0].text.strip() if name_elem else "N/A"
                    
                    # Extract price
                    price_elem = product.find_elements(By.CSS_SELECTOR, ".price")
                    price = price_elem[0].text.strip() if price_elem else "N/A"
                    
                    # Extract image URL
                    img_elem = product.find_elements(By.CSS_SELECTOR, "img")
                    image_url = img_elem[0].get_attribute("src") if img_elem else "N/A"
                    
                    # Extract product URL
                    link_elem = product.find_elements(By.CSS_SELECTOR, "a")
                    product_url = urljoin(base_url, link_elem[0].get_attribute("href")) if link_elem else "N/A"
                    
                    if name != "N/A":
                        self.scraped_data.append({
                            'name': name,
                            'price': price,
                            'rating': 'N/A',
                            'image_url': image_url,
                            'product_url': product_url,
                            'source': 'H&M'
                        })
                        
                except Exception as e:
                    logger.error(f"Error extracting product data: {e}")
                    continue
            
            logger.info(f"H&M scraping completed. Total products so far: {len(self.scraped_data)}")
            
        except Exception as e:
            logger.error(f"Error scraping H&M: {e}")
    
    def scrape_all_sites(self, search_terms=["mens shirts", "womens dresses", "casual wear"]):
        """Scrape all fashion sites with multiple search terms"""
        logger.info("Starting comprehensive fashion scraping")
        
        for search_term in search_terms:
            logger.info(f"Scraping for search term: {search_term}")
            
            # Scrape e-commerce sites
            self.scrape_amazon_fashion(search_term, max_pages=2)
            self.scrape_flipkart_fashion(search_term, max_pages=2)
            self.scrape_myntra_fashion(search_term, max_pages=1)
            self.scrape_h_and_m_fashion(search_term, max_pages=1)
            
            # Add delay between different search terms
            self.random_delay()
        
        # Scrape fashion magazines/blogs
        self.scrape_vogue_fashion(max_articles=15)
        
        logger.info(f"All scraping completed. Total products: {len(self.scraped_data)}")
    
    def save_to_csv(self, filename="real_fashion_data.csv"):
        """Save scraped data to CSV file"""
        if not self.scraped_data:
            logger.warning("No data to save")
            return None
        
        df = pd.DataFrame(self.scraped_data)
        
        # Clean the data
        df = self.clean_data(df)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}")
        logger.info(f"Total products saved: {len(df)}")
        
        # Print summary
        print("\n=== SCRAPING SUMMARY ===")
        print(f"Total products scraped: {len(df)}")
        print(f"Sources distribution:")
        print(df['source'].value_counts())
        
        return df
    
    def clean_data(self, df):
        """Clean and standardize the scraped data"""
        # Remove duplicates
        df = df.drop_duplicates(subset=['name', 'source'])
        
        # Clean price column
        df['price'] = df['price'].astype(str).str.replace(',', '')
        
        # Clean rating column
        df['rating'] = df['rating'].astype(str).str.extract('(\d+\.?\d*)', expand=False)
        
        # Remove rows with missing names
        df = df[df['name'] != 'N/A']
        
        return df
    
    def close_driver(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
            logger.info("WebDriver closed")

def main():
    """Main function to run the scraper"""
    scraper = RealFashionScraper(headless=False)  # Set to True for headless mode
    
    try:
        # Define search terms for different categories
        search_terms = [
            "mens shirts",
            "womens dresses", 
            "casual wear",
            "formal wear",
            "summer collection"
        ]
        
        # Scrape all sites
        scraper.scrape_all_sites(search_terms)
        
        # Save data
        df = scraper.save_to_csv()
        
        if df is not None:
            print("\nSample scraped data:")
            print(df.head())
            
            # Save additional metadata
            with open("scraping_metadata.json", "w") as f:
                json.dump({
                    "total_products": len(df),
                    "sources": df['source'].value_counts().to_dict(),
                    "scraping_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "search_terms": search_terms
                }, f, indent=2)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        
    finally:
        scraper.close_driver()

if __name__ == "__main__":
    main()
