
import requests
from bs4 import BeautifulSoup
import json




game_id_log=[]
THREE_LETTER_TEAM_ABBREVIATION="xxx"

for half in range(1,3):
    url = f"https://www.espn.com/mlb/team/schedule/_/name/{THREE_LETTER_TEAM_ABBREVIATION}/seasontype/2/half/{half}"
    
    # Add headers to mimic a browser so that I'm not blocked from the API
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    
        
# Check if the request was successful
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        for link in soup.find_all('a', href=True):
            if '/gameId/' in link['href']:
                game_id_p1 = link['href'].split('/gameId/')[1]
                game_id=game_id_p1.split('/')[0]
                if game_id not in game_id_log: #prevents accidental duplicates
                    game_id_log.append(game_id)
    else:
        print(f"Failed to retrieve page, status code: {response.status_code}")

print(game_id_log)


    
with open('id_log.json','w') as file:
    json.dump(game_id_log,file)


