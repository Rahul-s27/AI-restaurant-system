import pandas as pd
restaurant_data = pd.read_csv('Dataset.csv')
restaurant_data.head()
restaurant_data=restaurant_data[['Restaurant Name','Cuisines','Average Cost for two','Price range','Aggregate rating']]
restaurant_data.head()
restaurant_data['Cuisines']= restaurant_data['Cuisines'].str.replace(',' , ' ')
restaurant_data['Cuisines']=restaurant_data['Cuisines'].fillna('Unknown')
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english')
cv
cuisine_matrix = cv.fit_transform(restaurant_data['Cuisines'])
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(cuisine_matrix)
similarity
restaurant_data[restaurant_data['Restaurant Name']== 'Ooma'].index[0]
scores = sorted(list(enumerate(similarity[3])), reverse = True,key= lambda vector: vector[1])
for i in scores[0:6]:
    print(restaurant_data.iloc[i[0]]['Restaurant Name'])
    def recommend(restaurant_name):
        index = restaurant_data[restaurant_data['Restaurant Name']== restaurant_name].index[0]
        scores = sorted(list(enumerate(similarity[index])), reverse = True,key= lambda vector: vector[1])
        for i in scores[0:6]:
            print(restaurant_data.iloc[i[0]]['Restaurant Name'])
            recommend('Izakaya Kikufuji')
import pickle
with open('recommendation_model.pkl', 'wb') as file:
    pickle.dump(similarity, file)