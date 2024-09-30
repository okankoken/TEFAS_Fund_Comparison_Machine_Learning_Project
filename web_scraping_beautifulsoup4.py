import requests
from bs4 import BeautifulSoup


# Function to scrape fund data from TEFAS
def scrape_fund_data(fund_code):
    # Construct the URL for the fund analysis page
    url = f"https://www.tefas.gov.tr/FonAnaliz.aspx?FonKod={fund_code}"

    # Send a GET request to the webpage
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Scrape the fund name
        try:
            fund_name = soup.find('span', {'id': 'ContentPlaceHolder1_lbl_FonUnvani'}).text.strip()
        except AttributeError:
            fund_name = "Not Available"

        # Scrape other details (performance, asset sizes, etc.)
        try:
            daily_nav = soup.find('span', {'id': 'ContentPlaceHolder1_lbl_FonSonFiyat'}).text.strip()
        except AttributeError:
            daily_nav = "Not Available"

        try:
            nav_change = soup.find('span', {'id': 'ContentPlaceHolder1_lbl_FonGunlukGetiri'}).text.strip()
        except AttributeError:
            nav_change = "Not Available"

        try:
            fund_size = soup.find('span', {'id': 'ContentPlaceHolder1_lbl_FonToplamDeger'}).text.strip()
        except AttributeError:
            fund_size = "Not Available"

        # Add more fields as needed by inspecting the webpage structure

        # Print or return the scraped data
        data = {
            'Fund Name': fund_name,
            'Daily NAV': daily_nav,
            'NAV Change': nav_change,
            'Fund Size': fund_size
        }
        return data
    else:
        return None


# Example usage for a specific fund code (ICA in this case)
fund_code = "ICA"
fund_data = scrape_fund_data(fund_code)

if fund_data:
    for key, value in fund_data.items():
        print(f"{key}: {value}")
else:
    print("Failed to retrieve data.")
