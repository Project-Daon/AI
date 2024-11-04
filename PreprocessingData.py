import pandas as pd


data = pd.read_excel("data/Validation.xlsx")

emotions = ['분노', '기쁨', '불안', '슬픔']

data = data[['감정_대분류', '사람문장1']]

data = data[data['감정_대분류'].isin(emotions)]

data.to_csv("data/validation_data.csv")
