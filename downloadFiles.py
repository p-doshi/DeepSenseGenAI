import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# Function to setup Chrome WebDriver with download preferences
def setup_driver(download_folder):
    chromedriver_path = "chromedriver-mac-arm64/chromedriver"  # Adjust this to your chromedriver path
    service = Service(chromedriver_path)

    options = webdriver.ChromeOptions()
    prefs = {
        "download.default_directory": download_folder,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(service=service, options=options)
    return driver


# Function to click on all <a> tags with the 'download' attribute
def click_download_links(driver):
    # Wait until all <a> tags are visible
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a[download]"))
    )

    # Find all <a> tags with the 'download' attribute
    download_links = driver.find_elements(By.CSS_SELECTOR, "a[download]")

    # Click each download link
    for link in download_links:
        href = link.get_attribute('href')
        if href:
            driver.execute_script("window.open(arguments[0]);", href)
            time.sleep(1)  # Pause to allow the browser to initiate the download


# Load the initial webpage and handle 'Load more files' button
def load_full_page(driver, url):
    driver.get(url)
    try:
        while True:
            load_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Load more files')]"))
            )
            load_more_button.click()  # Click the button
            time.sleep(2)  # Small delay to allow content to load
    except Exception as e:
        print("All files loaded or an error occurred:", e)



def main():
    download_folder = os.path.join(os.getcwd(), 'downloaded_files')
    driver = setup_driver(download_folder)
    url = 'https://huggingface.co/datasets/DORI-SRKW/DORI-Orcasound/tree/main/data'
    load_full_page(driver, url)
    click_download_links(driver)
    time.sleep(5)  # Allow some time for any post-click activity
    driver.quit()


main()
