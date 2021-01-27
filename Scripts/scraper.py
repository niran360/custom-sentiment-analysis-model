import twitter
import tweepy
import csv
import pandas as pd
####input your credentials here
consumer_key = 'RVTSVkw5sNJQ1Tj0MbWrosxHR'
consumer_secret = 'OJYrU7FsNtq4IU5GovbAD97irppFGSx6wgbvT2ydmLc4jpMcg1'
access_token = '1034383632906047491-Wg6LeSInKC6yLJSDQDKbcWBxGiG3Wa'
access_token_secret = '6TQFyyQXMPQy1AtnB1Zcn8uswDutguZa32LVKWSVxatw4'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
#####United Airlines
# Open/Create a file to append data
csvFile = open('sarcasm.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search,q="#sarcasm",geocode='6.5244,3.3792,15mi',
                           lang="en",
                           until="2020-01-23").items():
    print (tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])