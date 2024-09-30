from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

# Path to your WebDriver (adjust this path)
webdriver_path = r'(C:\Program Files\Google\Chrome\Application\)

# Create a Service object with the path to your WebDriver
service = Service(webdriver_path)

# Set up Selenium WebDriver (Chrome in this case)
driver = webdriver.Chrome(service=service)


# Function to scrape fund data using Selenium
def scrape_fund_data(fund_code):
    url = f"https://www.tefas.gov.tr/FonAnaliz.aspx?FonKod={fund_code}"

    # Open the webpage
    driver.get(url)

    # Give the page time to load
    time.sleep(3)  # You can adjust this delay as needed

    # Scrape the fund name
    try:
        fund_name = driver.find_element(By.ID, 'ContentPlaceHolder1_lbl_FonUnvani').text
    except:
        fund_name = "Not Available"

    # Scrape the daily NAV
    try:
        daily_nav = driver.find_element(By.ID, 'ContentPlaceHolder1_lbl_FonSonFiyat').text
    except:
        daily_nav = "Not Available"

    # Scrape the NAV change
    try:
        nav_change = driver.find_element(By.ID, 'ContentPlaceHolder1_lbl_FonGunlukGetiri').text
    except:
        nav_change = "Not Available"

    # Scrape the fund size
    try:
        fund_size = driver.find_element(By.ID, 'ContentPlaceHolder1_lbl_FonToplamDeger').text
    except:
        fund_size = "Not Available"

    # Return the scraped data
    return {
        'Fund Name': fund_name,
        'Daily NAV': daily_nav,
        'NAV Change': nav_change,
        'Fund Size': fund_size
    }


# Example usage for a specific fund code (ICA)
fund_code = "ICA"
fund_data = scrape_fund_data(fund_code)

# Close the browser after scraping
driver.quit()

# Display the data
for key, value in fund_data.items():
    print(f"{key}: {value}")

