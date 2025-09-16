import os
import shutil
import glob
import sqlite3
from playwright.sync_api import sync_playwright
import logging
import json
import requests
import time
import re
from urllib.parse import urljoin
from datetime import datetime
import coloredlogs

# Setup colored logging
logger = logging.getLogger('PMC_Scraper')
logger.setLevel(logging.DEBUG)

coloredlogs.install(
    logger=logger,
    level='INFO',
    fmt='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level_styles={
        'debug': {'color': 'blue'},
        'info': {'color': 'green'},
        'warning': {'color': 'yellow'},
        'error': {'color': 'red'},
        'critical': {'color': 'red', 'bold': True}
    }
)

def extract_pmc_data(source_profile, target_profile):
    """Extract PMC-related data from existing profile"""
    logger.info("🔄 Setting up PMC access...")
    
    source_cookies = os.path.join(source_profile, "cookies.sqlite")
    target_cookies = os.path.join(target_profile, "cookies.sqlite")
    
    if os.path.exists(source_cookies):
        if os.path.exists(target_cookies):
            os.chmod(target_cookies, 0o777)
            os.remove(target_cookies)
        
        try:
            shutil.copy2(source_cookies, target_cookies)
            conn = sqlite3.connect(target_cookies)
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM moz_cookies 
                WHERE host NOT LIKE '%ncbi.nlm.nih.gov%' 
                AND host NOT LIKE '%nih.gov%'
                AND name NOT LIKE '%session%'
                AND name NOT LIKE '%auth%'
            """)
            conn.commit()
            conn.close()
            logger.info("✅ PMC cookies configured")
        except Exception as e:
            logger.warning("⚠️  Cookie setup failed: %s", e)
    
    essential_files = ["prefs.js", "user.js", "permissions.sqlite"]
    for file_name in essential_files:
        source_file = os.path.join(source_profile, file_name)
        target_file = os.path.join(target_profile, file_name)
        
        if os.path.exists(source_file):
            try:
                if os.path.exists(target_file):
                    os.chmod(target_file, 0o777)
                    os.remove(target_file)
                shutil.copy2(source_file, target_file)
            except Exception as e:
                logger.warning("⚠️  Failed to configure %s: %s", file_name, e)

def create_pmc_firefox_browser():
    """Create PMC-optimized Firefox browser"""
    try:
        snap_firefox = os.path.expanduser("~/snap/firefox/common/.mozilla/firefox")
        profiles = glob.glob(os.path.join(snap_firefox, "*.default*"))
        
        if not profiles:
            regular_firefox = os.path.expanduser("~/.mozilla/firefox")
            profiles = glob.glob(os.path.join(regular_firefox, "*.default*"))
        
        source_profile = profiles[0] if profiles else None
        target_profile = os.path.expanduser("~/pmc-firefox-profile")
        
        if os.path.exists(target_profile):
            def handle_readonly(func, path, exc):
                os.chmod(path, 0o777)
                func(path)
            shutil.rmtree(target_profile, onerror=handle_readonly)
        
        os.makedirs(target_profile, mode=0o755)
        
        if source_profile:
            extract_pmc_data(source_profile, target_profile)
        
        playwright = sync_playwright().start()
        browser = playwright.firefox.launch_persistent_context(
            user_data_dir=target_profile,
            headless=False,
            args=['--no-first-run', '--new-instance'],
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0'
        )
        
        return playwright, browser
        
    except Exception as e:
        logger.error("❌ Failed to create PMC Firefox: %s", e)
        raise

def search_pmc_for_article(search_term, max_results=50):
    """Search PMC for articles using NCBI API"""
    try:
        logger.info("🔍 Searching PMC for: %s", search_term)
        
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'pmc',
            'term': search_term,
            'retmode': 'json',
            'retmax': max_results,
            'sort': 'relevance'
        }
        
        response = requests.get(search_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if 'esearchresult' in data and data['esearchresult']['idlist']:
            results = []
            for pmc_id in data['esearchresult']['idlist']:
                pmc_url = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC%s/" % pmc_id
                results.append({'pmc_id': pmc_id, 'url': pmc_url})
            
            logger.info("📄 Found %d articles", len(results))
            return results
        else:
            logger.warning("❌ No PMC articles found")
            return []
            
    except Exception as e:
        logger.error("❌ PMC search error: %s", e)
        return []

class BatchPMCArticleScraper:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None
        self.batch_stats = {
            'total_papers': 0,
            'successful_papers': 0,
            'failed_papers': 0,
            'skipped_no_figures': 0,
            'total_figures': 0,
            'total_claims': 0,
            'total_images_downloaded': 0,
            'failed_urls': [],
            'filtered_tables': 0,
            '15_plus_figures': 0
        }
        
    def start_browser(self):
        """Initialize Firefox browser for PMC"""
        try:
            self.playwright, self.browser = create_pmc_firefox_browser()
            self.page = self.browser.pages[0] if self.browser.pages else self.browser.new_page()
            logger.info("🚀 PMC browser ready for batch processing")
        except Exception as e:
            logger.error("❌ Failed to start PMC browser: %s", e)
            raise

    def extract_paper_id_from_url(self, url):
        """Extract PMC ID from URL"""
        match = re.search(r'PMC(\d+)', url)
        return "PMC%s" % match.group(1) if match else "pmc_%d" % int(time.time())

    def get_all_possible_image_urls(self, figure, img_src):
        """Get all possible image URLs to try"""
        urls_to_try = []
        
        # 1. Original image src (highest priority)
        if img_src:
            urls_to_try.append(("original_src", img_src))
        
        # 2. Look for tileshop link data
        tileshop_link = figure.query_selector("a.tileshop")
        if tileshop_link:
            href = tileshop_link.get_attribute("href")
            if href:
                logger.debug("🔍 Found tileshop link: %s", href)
                # Try to click the link and get the actual high-res URL
                # For now, we'll skip this complex approach
        
        # 3. Try different size variations of the original URL
        if img_src and 'ncbi.nlm.nih.gov' in img_src:
            # Remove size parameters
            base_url = re.sub(r'[?&](size|maxwidth|maxheight)=[^&]*', '', img_src)
            if base_url != img_src:
                urls_to_try.append(("no_size_params", base_url))
            
            # Try size variations
            size_variations = [
                ('large_size', base_url + '?maxwidth=2000'),
                ('original_size', base_url + '?size=original'),
                ('xlarge_size', base_url + '?maxwidth=4000'),
            ]
            urls_to_try.extend(size_variations)
            
            # Try path modifications
            if '/thumb/' in img_src:
                large_url = img_src.replace('/thumb/', '/large/')
                urls_to_try.append(("thumb_to_large", large_url))
            
            if '/medium/' in img_src:
                large_url = img_src.replace('/medium/', '/large/')
                urls_to_try.append(("medium_to_large", large_url))
        
        return urls_to_try

    def is_table_image(self, title_text, caption_text):
        """Enhanced table detection with comprehensive keyword filtering"""
        
        # Comprehensive table-related keywords
        table_keywords = [
            # Direct table references
            'table', 'tabular', 'tabulated', 'tab.', 'tbl',
            
            # Summary/demographic tables
            'summary of', 'characteristics of patients', 'baseline characteristics', 
            'patient demographics', 'clinical characteristics', 'patient characteristics',
            'demographic data', 'demographic characteristics', 'baseline data',
            'study population', 'participant characteristics', 'cohort characteristics',
            
            # Statistical/results tables
            'statistical analysis', 'statistical summary', 'analysis results',
            'univariate analysis', 'multivariate analysis', 'regression analysis',
            'correlation analysis', 'comparison of', 'outcomes by', 'results by',
            
            # Data presentation tables
            'data summary', 'summary statistics', 'descriptive statistics',
            'frequencies', 'percentages', 'distributions', 'cross-tabulation',
            'contingency table', 'frequency table', 'data table',
            
            # Medical/clinical tables
            'treatment outcomes', 'clinical outcomes', 'survival rates',
            'response rates', 'adverse events', 'side effects', 'complications',
            'laboratory values', 'test results', 'biomarker levels',
            
            # Comparison tables
            'before and after', 'pre and post', 'control vs', 'treatment vs',
            'group comparison', 'between groups', 'stratified by',
            
            # Timeline/schedule tables
            'treatment schedule', 'dosing schedule', 'timeline', 'follow-up schedule',
            'study timeline', 'protocol schedule'
        ]
        
        # Medical procedure keywords that should NOT be filtered (keep these images)
        procedure_keywords = [
            'procedure', 'surgical', 'technique', 'method', 'approach', 'steps',
            'illustration', 'diagram', 'schematic', 'flowchart', 'algorithm',
            'oncoplastic', 'mammoplasty', 'mastopexy', 'lumpectomy', 'mastectomy',
            'surgical steps', 'operation', 'operative', 'intraoperative',
            'pattern', 'wise pattern', 'surgical technique', 'reconstruction',
            'imaging', 'mri', 'ct scan', 'ultrasound', 'mammography', 'biopsy',
            'histology', 'pathology', 'microscopy', 'staining', 'immunohistochemistry'
        ]
        
        # Image/figure keywords that should NOT be filtered
        image_keywords = [
            'photograph', 'photo', 'image', 'radiograph', 'x-ray', 'scan',
            'microscopic', 'macroscopic', 'gross', 'specimen', 'tissue',
            'histological', 'cytological', 'molecular', 'genetic'
        ]
        
        combined_text = (title_text + " " + caption_text).lower()
        
        # First priority: Protect medical procedures and images
        for keyword in procedure_keywords + image_keywords:
            if keyword in combined_text:
                logger.debug("✅ Protected content detected: %s", keyword)
                return False
        
        # Check for table indicators
        for keyword in table_keywords:
            if keyword in combined_text:
                logger.debug("🚫 Table keyword detected: %s", keyword)
                return True
        
        # Additional pattern-based detection
        table_patterns = [
            r'\btable\s+\d+',  # "table 1", "table 2", etc.
            r'\btab\.\s+\d+',  # "tab. 1", "tab. 2", etc.
            r'see\s+table',    # "see table"
            r'shown\s+in\s+table',  # "shown in table"
            r'listed\s+in\s+table',  # "listed in table"
            r'n\s*=\s*\d+',    # statistical notation like "n = 50"
            r'p\s*[<>=]\s*0\.',  # p-values like "p < 0.05"
            r'\d+\s*\(\s*\d+\.?\d*%\s*\)',  # percentages in parentheses
            r'mean\s*±\s*std',  # statistical notation
            r'median\s*\[.*\]',  # median with range
        ]
        
        for pattern in table_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                logger.debug("🚫 Table pattern detected: %s", pattern)
                return True
        
        return False

    def extract_figures_and_captions(self):
        """Extract ALL figures with enhanced table filtering"""
        figures_data = {}
        
        try:
            logger.info("🔍 Starting comprehensive figure extraction...")
            
            # Expanded selectors to catch more figure types
            all_selectors = [
                "figure",
                "figure.fig", 
                "div.fig",
                "div.figure",
                "[id*='fig']",
                "[id*='Fig']",
                "[class*='fig']",
                "[class*='Fig']",
                ".image-container",
                ".figure-container"
            ]
            
            all_figures = []
            for selector in all_selectors:
                elements = self.page.query_selector_all(selector)
                logger.info("📊 Selector '%s' found %d elements", selector, len(elements))
                all_figures.extend(elements)
            
            # Remove duplicates by getting unique elements
            unique_figures = []
            seen_elements = set()
            for fig in all_figures:
                element_id = id(fig)
                if element_id not in seen_elements:
                    unique_figures.append(fig)
                    seen_elements.add(element_id)
            
            logger.info("📊 Total unique figure elements found: %d", len(unique_figures))
            
            processed_ids = set()
            
            for i, figure in enumerate(unique_figures):
                try:
                    logger.debug("🔍 Processing figure element %d", i+1)
                    
                    # Get figure ID - be more flexible
                    fig_id = figure.get_attribute("id")
                    if not fig_id:
                        # Try to create an ID from class or other attributes
                        fig_class = figure.get_attribute("class") or ""
                        if 'fig' in fig_class.lower():
                            fig_id = f"auto_fig_{i}"
                        else:
                            fig_id = f"figure_{i}"
                    
                    logger.debug("📋 Figure ID: %s", fig_id)
                    
                    if fig_id in processed_ids:
                        logger.debug("⚠️ Already processed ID %s, skipping", fig_id)
                        continue
                    
                    processed_ids.add(fig_id)
                    
                    # Extract figure number - be more flexible
                    fig_match = re.search(r'fig(\d+)', fig_id, re.IGNORECASE)
                    if not fig_match:
                        # Try to find figure number in surrounding text
                        figure_text = figure.text_content()
                        fig_match = re.search(r'figure\s+(\d+)', figure_text, re.IGNORECASE)
                    
                    if fig_match:
                        fig_num = fig_match.group(1)
                    else:
                        # Assign sequential number
                        fig_num = str(len(figures_data) + 1)
                    
                    logger.debug("📋 Figure number: %s", fig_num)
                    
                    # Find image element - try multiple approaches
                    img_element = figure.query_selector("img")
                    if not img_element:
                        # Look for images in child elements
                        img_element = figure.query_selector("a img, div img, span img")
                    
                    if not img_element:
                        logger.debug("⚠️ No img element found in figure %s", fig_num)
                        continue
                    
                    logger.debug("✅ Found img element in figure %s", fig_num)
                    
                    # Extract title with more selectors
                    title_text = ""
                    title_selectors = ["h4", "h3", "h2", "h1", ".obj_head", "strong", "b", ".title", ".fig-title"]
                    
                    for title_sel in title_selectors:
                        title_elem = figure.query_selector(title_sel)
                        if title_elem:
                            title_text = title_elem.text_content().strip()
                            logger.debug("📋 Title from %s: %s", title_sel, title_text[:50])
                            if title_text:  # Stop at first non-empty title
                                break
                    
                    # Extract caption with expanded search
                    caption_content = ""
                    caption_selectors = ["figcaption", ".caption", ".fig-caption", "p", ".description", ".legend"]
                    
                    for cap_sel in caption_selectors:
                        caption_elems = figure.query_selector_all(cap_sel)
                        if caption_elems:
                            caption_parts = []
                            for elem in caption_elems:
                                text = elem.text_content().strip()
                                # More refined text filtering
                                if (text and len(text) > 10 and 
                                    not text.startswith("Download") and 
                                    not text.startswith("Open in") and
                                    not text.startswith("View larger") and
                                    not text.startswith("Copyright")):
                                    caption_parts.append(text)
                            if caption_parts:
                                caption_content = " ".join(caption_parts)
                                logger.debug("📋 Caption: %s", caption_content[:100])
                                break
                    
                    # Get image URLs
                    img_src = None
                    for attr in ['src', 'data-src', 'data-original', 'data-lazy-src']:
                        img_src = img_element.get_attribute(attr)
                        if img_src:
                            logger.debug("📋 Found image src via %s: %s", attr, img_src)
                            break
                    
                    if not img_src:
                        logger.debug("⚠️ No image src found for figure %s", fig_num)
                        continue
                    
                    # Make URL absolute
                    if img_src.startswith("/"):
                        img_src = "https://www.ncbi.nlm.nih.gov" + img_src
                    elif not img_src.startswith("http"):
                        img_src = urljoin(self.page.url, img_src)
                    
                    logger.debug("📋 Final image URL: %s", img_src)
                    
                    # Enhanced table filtering
                    if self.is_table_image(title_text, caption_content):
                        self.batch_stats['filtered_tables'] += 1
                        logger.info("🚫 Filtered table: Figure %s (Title: %s)", fig_num, title_text[:50])
                        continue
                    
                    # Get all possible URLs to try
                    possible_urls = self.get_all_possible_image_urls(figure, img_src)
                    
                    figure_key = f"figure_{fig_num}"
                    figures_data[figure_key] = {
                        "figure_number": f"Figure {fig_num}",
                        "title": title_text,
                        "caption": caption_content,
                        "possible_urls": possible_urls,
                        "figure_id": fig_id,
                        "is_table": False
                    }
                    
                    logger.info("✅ Added Figure %s: %s", fig_num, title_text[:50] if title_text else "(no title)")
                    
                except Exception as e:
                    logger.error("❌ Error processing figure %d: %s", i+1, e)
                    continue
            
            logger.info("📊 Total figures extracted: %d (filtered %d tables)", 
                    len(figures_data), self.batch_stats['filtered_tables'])
            return figures_data
            
        except Exception as e:
            logger.error("❌ Error in figure extraction: %s", e)
            return {}


    def download_image_with_fallback(self, possible_urls, filepath):
        """Try downloading from multiple URLs until one works"""
        try:
            logger.debug("⬇️ Trying %d URLs for %s", len(possible_urls), os.path.basename(filepath))
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0',
                'Referer': self.page.url,
                'Accept': 'image/webp,image/png,image/jpeg,image/gif,*/*'
            }
            
            for url_type, url in possible_urls:
                try:
                    logger.debug("🔗 Trying %s: %s", url_type, url)
                    
                    response = requests.get(url, headers=headers, stream=True, timeout=15, allow_redirects=True)
                    
                    if response.status_code == 200:
                        content_type = response.headers.get('content-type', '').lower()
                        if 'image' in content_type:
                            # Create directory
                            os.makedirs(os.path.dirname(filepath), exist_ok=True)
                            
                            # Download the file
                            with open(filepath, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                            
                            # Verify the download
                            if os.path.exists(filepath):
                                file_size = os.path.getsize(filepath)
                                
                                if file_size > 500:  # At least 500 bytes
                                    logger.info("✅ Downloaded via %s: %s (%.1f KB)", 
                                               url_type, os.path.basename(filepath), file_size/1024)
                                    return True, url
                                else:
                                    logger.debug("⚠️ File too small via %s: %d bytes", url_type, file_size)
                                    os.remove(filepath)
                        else:
                            logger.debug("⚠️ Not an image via %s: %s", url_type, content_type)
                    else:
                        logger.debug("⚠️ HTTP %d via %s", response.status_code, url_type)
                        
                except Exception as e:
                    logger.debug("⚠️ Failed %s: %s", url_type, e)
                    continue
            
            logger.warning("❌ All URLs failed for %s", os.path.basename(filepath))
            return False, None
                
        except Exception as e:
            logger.error("❌ Download error: %s", e)
            return False, None

    def extract_all_text_with_figure_refs(self):
        """Extract ALL text from the page that contains figure references"""
        claims = []
        
        try:
            # Get all text content from the page
            all_text_elements = self.page.query_selector_all("p, div, span, section")
            
            # Enhanced figure reference patterns
            figure_patterns = [
                re.compile(r'\b(?:Figure|Fig\.?)\s*(\d+)([A-Za-z]*)', re.IGNORECASE),
                re.compile(r'\((?:Figure|Fig\.?)\s*(\d+)([A-Za-z]*)\)', re.IGNORECASE),
                re.compile(r'\((?:Fig\.?)\s*(\d+)([A-Za-z]*)\)', re.IGNORECASE),
            ]
            
            supp_figure_pattern = re.compile(r'\b(?:Figure|Fig\.?)\s+S\d+', re.IGNORECASE)
            
            for element in all_text_elements:
                try:
                    text_content = element.text_content().strip()
                    if not text_content or len(text_content) < 20:
                        continue
                    
                    # Skip supplementary figures
                    if supp_figure_pattern.search(text_content):
                        continue
                    
                    # Check if text contains any figure references
                    has_figure_ref = False
                    for pattern in figure_patterns:
                        if pattern.search(text_content):
                            has_figure_ref = True
                            break
                    
                    if has_figure_ref:
                        # Split into sentences
                        sentences = self.split_into_sentences(text_content)
                        
                        for sentence in sentences:
                            # Check each sentence for figure references
                            sentence_matches = []
                            for pattern in figure_patterns:
                                sent_matches = pattern.findall(sentence)
                                sentence_matches.extend(sent_matches)
                            
                            if sentence_matches:
                                figure_refs = []
                                for match in sentence_matches:
                                    fig_num = match[0]
                                    panel = match[1] if len(match) > 1 and match[1] else ""
                                    figure_refs.append({
                                        "figure_number": "Figure %s" % fig_num,
                                        "panel": "Panel %s" % panel if panel else "",
                                        "figure_key": "figure_%s" % fig_num
                                    })
                                
                                claims.append({
                                    "sentence": sentence.strip(),
                                    "figure_references": figure_refs
                                })
                
                except Exception as e:
                    continue
            
            return claims
            
        except Exception as e:
            return []

    def split_into_sentences(self, text):
        """Split text into sentences with better handling"""
        # Preserve abbreviations
        text = re.sub(r'\b(Fig|Ref|Eq|et al|Dr|Mr|Mrs|vs|etc)\.', lambda m: m.group().replace('.', '_DOT_'), text)
        
        # Split on periods followed by space and capital letter or digit
        sentences = re.split(r'\.(?=\s+[A-Z\d])', text)
        
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.replace('_DOT_', '.').strip()
            if sent and not sent.endswith('.'):
                sent += '.'
            if sent and len(sent) > 15:  # Must be at least 15 characters
                cleaned_sentences.append(sent)
        
        return cleaned_sentences

    def scrape_single_article(self, url, output_dir):
        """Scrape a single article - only process if valid figures exist"""
        paper_id = None
        try:
            paper_id = self.extract_paper_id_from_url(url)
            logger.info("📋 Processing %s...", paper_id)
            
            # Navigate to the article
            self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            time.sleep(3)  # Give page time to fully load
            
            # First, extract figures to see if we have any valid ones
            logger.info("🔍 Extracting figures...")
            figures_data = self.extract_figures_and_captions()
            
            if not figures_data:
                logger.warning("⚠️ No valid figures found in %s - skipping paper", paper_id)
                self.batch_stats['skipped_no_figures'] += 1
                return None
            if len(figures_data) > 15:
                logger.warning("⚠️ 15+ valid figures found in %s - skipping paper", paper_id)
                self.batch_stats['15_plus_figures'] += 1
                return None
            
            # Create temp directory to test image downloads
            temp_images_dir = os.path.join("/tmp", "pmc_temp", paper_id, "images")
            os.makedirs(temp_images_dir, exist_ok=True)
            
            # Test download all images first
            logger.info("⬇️ Testing download of %d images...", len(figures_data))
            valid_figures = {}
            successful_downloads = 0
            
            for fig_key, fig_data in figures_data.items():
                temp_img_path = os.path.join(temp_images_dir, "%s.jpg" % fig_key)
                logger.info("⬇️ Testing download %s...", fig_key)
                
                success, working_url = self.download_image_with_fallback(fig_data["possible_urls"], temp_img_path)
                
                if success:
                    fig_data["working_url"] = working_url
                    fig_data["download_success"] = True
                    valid_figures[fig_key] = fig_data
                    successful_downloads += 1
                    logger.info("✅ Downloaded %s", fig_key)
                else:
                    fig_data["download_success"] = False
                    logger.warning("❌ Failed to download %s", fig_key)
            
            # Clean up temp directory
            try:
                shutil.rmtree(os.path.join("/tmp", "pmc_temp", paper_id))
            except:
                pass
            
            # Check if we have at least one valid figure
            if not valid_figures:
                logger.warning("⚠️ No images successfully downloaded for %s - skipping paper", paper_id)
                self.batch_stats['skipped_no_figures'] += 1
                return None
            
            logger.info("✅ %s has %d valid figures - proceeding with full processing", 
                       paper_id, len(valid_figures))
            
            # Now create the real output directories
            paper_dir = os.path.join(output_dir, paper_id)
            images_dir = os.path.join(paper_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # Download images to final location using working URLs
            logger.info("⬇️ Downloading %d images to final location...", len(valid_figures))
            final_successful_downloads = 0
            
            for fig_key, fig_data in valid_figures.items():
                img_path = os.path.join(images_dir, "%s.jpg" % fig_key)
                
                # Use the working URL we found during testing
                working_url = fig_data["working_url"]
                success, _ = self.download_image_with_fallback([("working", working_url)], img_path)
                
                if success:
                    fig_data["local_image_path"] = img_path
                    fig_data["final_url"] = working_url
                    final_successful_downloads += 1
                else:
                    # Remove from valid figures if final download fails
                    del valid_figures[fig_key]
            
            # Final check - make sure we still have valid figures
            if not valid_figures:
                logger.warning("⚠️ All images failed final download for %s - cleaning up", paper_id)
                shutil.rmtree(paper_dir)
                self.batch_stats['skipped_no_figures'] += 1
                return None
            
            # Only extract claims if we have valid figures
            logger.info("🔍 Extracting claims (paper has %d valid figures)...", len(valid_figures))
            claims = self.extract_all_text_with_figure_refs()
            
            # Save data
            captions_file = os.path.join(paper_dir, "captions.json")
            captions_dict = {fig_key: fig_data["caption"] for fig_key, fig_data in valid_figures.items()}
            with open(captions_file, 'w', encoding='utf-8') as f:
                json.dump(captions_dict, f, indent=2, ensure_ascii=False)
            
            claims_file = os.path.join(paper_dir, "claims.json")
            with open(claims_file, 'w', encoding='utf-8') as f:
                json.dump(claims, f, indent=2, ensure_ascii=False)
            
            complete_data = {
                "paper_id": paper_id,
                "url": url,
                "figures": valid_figures,
                "claims": claims,
                "extraction_stats": {
                    "figures_count": len(valid_figures),
                    "claims_count": len(claims),
                    "images_downloaded": final_successful_downloads,
                    "tables_filtered": self.batch_stats['filtered_tables']
                }
            }
            
            complete_file = os.path.join(paper_dir, "complete_data.json")
            with open(complete_file, 'w', encoding='utf-8') as f:
                json.dump(complete_data, f, indent=2, ensure_ascii=False)
            
            # Update batch statistics
            self.batch_stats['successful_papers'] += 1
            self.batch_stats['total_figures'] += len(valid_figures)
            self.batch_stats['total_claims'] += len(claims)
            self.batch_stats['total_images_downloaded'] += final_successful_downloads
            
            logger.info("✅ %s: %d figures, %d claims, %d images downloaded", 
                       paper_id, len(valid_figures), len(claims), final_successful_downloads)
            
            return complete_data
            
        except Exception as e:
            logger.error("❌ Failed %s: %s", paper_id or "unknown", e)
            self.batch_stats['failed_papers'] += 1
            self.batch_stats['failed_urls'].append(url)
            return None

    def batch_scrape_articles(self, search_term="Brain", max_papers=50, output_dir="./data"):
        """Batch scrape multiple articles"""
        try:
            logger.info("🚀 Starting batch scraping for: '%s'", search_term)
            logger.info("📊 Target: %d papers", max_papers)
            logger.info("📁 Output directory: %s", output_dir)
            logger.info("⚠️ Note: Only papers with valid figures will be processed")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Search for articles
            logger.info("🔍 Searching PMC...")
            results = search_pmc_for_article(search_term, max_results=max_papers)
            
            if not results:
                logger.warning("❌ No articles found")
                return
            
            # Limit to requested number
            results = results[:max_papers]
            self.batch_stats['total_papers'] = len(results)
            
            logger.info("📚 Found %d articles to process", len(results))
            logger.info("=" * 60)
            
            # Process each article
            start_time = time.time()
            
            for i, result in enumerate(results, 1):
                try:
                    logger.info("[%d/%d] Processing PMC%s...", i, len(results), result['pmc_id'])
                    
                    # Scrape the article (will skip if no valid figures)
                    self.scrape_single_article(result['url'], output_dir)
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    avg_time = elapsed / i
                    remaining = (len(results) - i) * avg_time
                    
                    logger.info("⏱️  Progress: %d/%d (%.1f%%) | Elapsed: %.1fmin | ETA: %.1fmin", 
                              i, len(results), (i/len(results)*100), elapsed/60, remaining/60)
                    
                    # Small delay to be respectful to PMC servers
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error("❌ Error processing article %d: %s", i, e)
                    self.batch_stats['failed_papers'] += 1
                    continue
            
            # Save batch summary
            self.save_batch_summary(output_dir, search_term, start_time)
            
            # Print final statistics
            self.print_batch_statistics()
            
            logger.info("🎉 Batch processing completed!")
            logger.info("📁 Results saved to: %s", output_dir)
            
        except Exception as e:
            logger.error("❌ Batch processing error: %s", e)

    def save_batch_summary(self, output_dir, search_term, start_time):
        """Save batch processing summary"""
        try:
            total_time = time.time() - start_time
            
            summary = {
                "batch_info": {
                    "search_term": search_term,
                    "timestamp": datetime.now().isoformat(),
                    "total_processing_time_minutes": round(total_time / 60, 2)
                },
                "statistics": self.batch_stats,
                "failed_urls": self.batch_stats['failed_urls']
            }
            
            summary_file = os.path.join(output_dir, "batch_summary.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info("📊 Batch summary saved to: %s", summary_file)
            
        except Exception as e:
            logger.error("❌ Error saving batch summary: %s", e)

    def print_batch_statistics(self):
        """Print final batch statistics"""
        stats = self.batch_stats
        logger.info("\n" + "="*60)
        logger.info("📊 BATCH PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info("📄 Total papers attempted: %d", stats['total_papers'])
        logger.info("✅ Successfully processed: %d", stats['successful_papers'])
        logger.info("⚠️ Skipped (no valid figures): %d", stats['skipped_no_figures'])
        logger.info("❌ Failed: %d", stats['failed_papers'])
        
        if stats['total_papers'] > 0:
            success_rate = (stats['successful_papers']/stats['total_papers']*100)
            skip_rate = (stats['skipped_no_figures']/stats['total_papers']*100)
            logger.info("📊 Success rate: %.1f%%", success_rate)
            logger.info("📊 Skip rate (no figures): %.1f%%", skip_rate)
        
        logger.info("📸 Total figures extracted: %d", stats['total_figures'])
        logger.info("🚫 Tables filtered out: %d", stats['filtered_tables'])
        logger.info("📝 Total claims extracted: %d", stats['total_claims'])
        logger.info("⬇️ Total images downloaded: %d", stats['total_images_downloaded'])
        
        if stats['successful_papers'] > 0:
            logger.info("📈 Average per successful paper:")
            logger.info("   • Figures: %.1f", stats['total_figures']/stats['successful_papers'])
            logger.info("   • Claims: %.1f", stats['total_claims']/stats['successful_papers'])
            logger.info("   • Images: %.1f", stats['total_images_downloaded']/stats['successful_papers'])

    def close(self):
        """Close browser"""
        try:
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
        except Exception as e:
            logger.error("❌ Cleanup error: %s", e)

def main():
    """Main function with simplified batch processing"""
    logger.info("🔬 PMC Batch Article Scraper - Brain Cancer Research")
    logger.info("=" * 60)
    
    # Hardcoded parameters
    search_term = "eyes" # brain cancer, breast cancer, skin cancer, blood cancer
    max_papers = 100
    output_dir = "./data"
    
    logger.info("🎯 Configuration:")
    logger.info("   • Search term: '%s' (hardcoded)", search_term)
    logger.info("   • Max papers: %d (default)", max_papers)
    logger.info("   • Output directory: %s", output_dir)
    logger.info("   • Requirement: At least 1 valid figure per paper")
    
    # Start batch scraping immediately
    scraper = BatchPMCArticleScraper()
    
    try:
        scraper.start_browser()
        scraper.batch_scrape_articles(search_term, max_papers, output_dir)
        
    except Exception as e:
        logger.error("❌ Error: %s", e)
    
    finally:
        scraper.close()

if __name__ == "__main__":
    main()
