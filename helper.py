# it fetch the data for analysis of unique user
import pandas as pd
from collections import Counter
from urlextract import URLExtract
from wordcloud import WordCloud
import emoji
ext=URLExtract()
def fetch_stats(selected_user,df):
    if selected_user!='Overall':
        df=df[df['user']==selected_user.replace('\n',' ')]
    #number of total messages
    num_messages=df.shape[0]
    #number of total words
    words=[]
    for i in df['message']:
        words.extend(i.split())
    #number of media
    num_media=df[df['message']=='<Media omitted>\n'].shape[0]
    #number of links
    links=[]
    for i in df['message']:
        links.extend(ext.find_urls(i))
        
    return num_messages,len(words),num_media,len(links)

        #finding the most active users in the group (only for group)
def most_busy_user(df):
    x=df['user'].value_counts().head()
    df=round(df['user'].value_counts()/df.shape[0]*100,2).reset_index().rename(columns={'index':'Name','user':'Percentage'})
    return x,df

        #WorldCloud
def create_wordcloud(selected_user,df):
    if selected_user!='Overall':
        df=df[df['user']==selected_user.replace('\n',' ')]
    wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    df_new=df[df['message']!='<Media omitted>\n']['message']
    dfc=[]
    for i in df_new:
        if(i.find('joined using this group')==-1):
            dfc.append(i)
    dfc=pd.Series(dfc)
    # it will broke each word that how much time it come then there size will be bigger
    df_wc=wc.generate(dfc.str.cat(sep=' '))
    return df_wc

        #Most Common Words
def most_common_words(selected_user,df):
    # removing stop words messages like a , the ,etc
    f=open('stop_hinglish.txt','r')
    stop_word=f.read()
    if selected_user!='Overall':
        df=df[df['user']==selected_user.replace('\n',' ')]
        # Removing group notification messages
    temp=df[df['user']!='group_notification']
    # Removing media messages
    temp=temp[temp['message']!='<Media omitted>\n']
    
    words=[]
    
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_word:
                words.append(word)
    return pd.DataFrame(Counter(words).most_common(20))    

        #Emoji Analysis
# def emoji_helper(selected_user,df):
#     if selected_user!='Overall':
#         df=df[df['user']==selected_user.replace('\n',' ')]
    
#     emojis=[]
#     for message in df['message']:
#         emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
#     return pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

            #Timeline Analysis 
def monthly_timeline(selected_user,df):
    if selected_user!='Overall':
        df=df[df['user']==selected_user.replace('\n',' ')]
    timeline=df.groupby(['year','month_num','month']).count()['message'].reset_index()
    time=[]
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i]+'-'+str(timeline['year'][i]))
    timeline['time']=time
    return timeline

def daily_timeline(selected_user,df):
    if selected_user!='Overall':
        df=df[df['user']==selected_user.replace('\n',' ')]
    daily_timeline=df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline

        #Activity Map
def week_activity_map(selected_user,df):
    if selected_user!='Overall':
        df=df[df['user']==selected_user.replace('\n',' ')]
    return df['day_name'].value_counts()
    
def month_activity_map(selected_user,df):
    if selected_user!='Overall':
        df=df[df['user']==selected_user.replace('\n',' ')]
    return df['month'].value_counts()

def activity_heatmap(selected_user,df):
    if selected_user!='Overall':
        df=df[df['user']==selected_user.replace('\n',' ')]
    return df.pivot_table(index='day_name',columns='period',values='message',aggfunc='count').fillna(0)