#%%
import requests
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#%%

with open('id_log.json') as file:
    game_id_log=file.readlines()


#%%

PITCHER_ID='XXXXX'

#%%
pitches_dict={}

pitches_log=[]
for game_id in game_id_log:
    api_link=f'https://cdn.espn.com/core/mlb/playbyplay?xhr=1&gameId={game_id}'
    response=requests.get(api_link)
    apijson=response.json()
    plays=apijson['gamepackageJSON']['plays']
    
    for index in range(len(plays)):
        if 'participants' in plays[index]:
            if plays[index]['participants'][0]['athlete']['id']==PITCHER_ID and ('pitchCoordinate' in plays[index]):
                try:
                    previous_pitch=int(plays[index-1]['pitchType']['id']) #gets previous pitch
                except KeyError:  #catches if it is the first pitch of an AB
                    previous_pitch=0
                if plays[index]['bats']['abbreviation']=='R': #gets handedness of opposing batter
                    batter_handedness=0
                elif plays[index]['bats']['abbreviation']=='L':
                    batter_handedness=1
                balls = int(plays[index]['pitchCount']['balls'])
                strikes = int(plays[index]['pitchCount']['strikes'])
                runners_on=0
                bases=['onFirst','onSecond','onThird']
                for runner in bases: #counts amount of runners on base
                    if runner in plays[index]:
                        runners_on+=1
                try:
                    pitch_type = int(plays[index]['pitchType']['id'])
                    row={'previous_pitch':previous_pitch,'runners_on':runners_on,'batter_handedness':batter_handedness,'balls':balls,'strikes':strikes,'pitch_type':pitch_type}
                    pitches_log.append(row)
                except KeyError:
                    pass
        #Finds the different pitches that the pitcher throws and converts them from numeric ID to text
                try:
                    if plays[index]['pitchType']['id'] not in pitches_dict.keys():
                        pitches_dict[plays[index]['pitchType']['id']]=plays[index]['pitchType']['text']
                except KeyError:
                    pass
     


#%%


handedness_dict={'0':'Righty','1':'Lefty'}



df = pd.DataFrame(pitches_log)


x = df.drop(columns=['pitch_type'])  # drop classification column and get features
y = df['pitch_type']  # get the target variable



# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize and train Random Forest model
forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
forest_model.fit(X_train, y_train)

# Feature importance plot
feature_importances = forest_model.feature_importances_
features = X_train.columns
indices = np.argsort(feature_importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(features)), feature_importances[indices], align="center")
plt.xticks(range(len(features)), features[indices], rotation=90)
plt.show()

# Initialize and train Decision Tree model
tree_model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
tree_model.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = tree_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Pruned Decision Tree Accuracy:", accuracy)

# Plot the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(
    tree_model,
    feature_names=X.columns,
    class_names=y.unique().astype(str),
    filled=True
)
plt.show()


