import matplotlib.pyplot as plt
import streamlit as st
import preprocessor,helper
import seaborn as sns
import altair as alt
import numpy as np
import pandas as pd
import joblib

pipe_lr = joblib.load(open("D:\ml\project\Hawk/chat_emotion.pkl.gz", "rb"))

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}


st.set_page_config(layout="wide")
# This is done for uploading the chat file in sidebar
st.sidebar.title('Hawk')
st.sidebar.header('Chat Analyzer')
# uploading a file
uploaded_file=st.sidebar.file_uploader('Choose a File')
if uploaded_file is not None:
    bytes_data=uploaded_file.getvalue()
    # currently data is in stream form to convert it into string we use decode function
    data=bytes_data.decode('utf-8')
    df=preprocessor.preprocess(data)
    # st.dataframe(df)
    
    #fetching unique users in sidebar nav 
    user_list=df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,'Overall')
    selected_user=st.sidebar.selectbox('Show Analysis of:', user_list)
    if st.sidebar.button('Show Analysis'):
        
        num_messages,words,num_media,links= helper.fetch_stats(selected_user,df)
        st.title('Top Statistics')
        col1,col2,col3,col4=st.columns(4)
        with col1:
            st.header('Total Messages')
            st.title(num_messages)
        with col2:
            st.header('Total Words')
            st.title(words)
        with col3:
            st.header('Media Shared')
            st.title(num_media)
        with col4:
            st.header('Links Shared')
            st.title(links)
            
        #finding the most active users in the group (only for group)
        if selected_user=='Overall':
            st.title('Most Active Users')
            x,df_per=helper.most_busy_user(df)
            fig,ax=plt.subplots()
            col1,col2=st.columns(2)
            with col1:
                ax.bar(x.index,x.values)
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(df_per)
        
        #WorldCloud
        st.title('Most Common WordCloud')
        df_wc=helper.create_wordcloud(selected_user,df)
        fig,ax=plt.subplots()
        ax.imshow(df_wc) 
        st.pyplot(fig)
        
        #Most Common Words
        st.title('Most Common Words')
        col1,col2=st.columns(2)
        most_common_df=helper.most_common_words(selected_user,df)
        with col1 :
            fig,ax=plt.subplots()
            plt.xticks(rotation='vertical')
            ax.barh(most_common_df[0],most_common_df[1])
            st.pyplot(fig)
            
            # sns.barplot(most_common_df[0],most_common_df[1])
            
        with col2:        
            st.dataframe(most_common_df)
            
        #Emoji Analysis
        # emoji_df=helper.emoji_helper(selected_user,df)
        # st.title("Emoji Analysis")

        # col1,col2=st.columns(2)

        # with col1:
        #     st.dataframe(emoji_df)
        # with col2:
        #     fig,ax=plt.subplots()
        #     ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
        #     st.pyplot(fig)
        
        # Emotion detection
        st.title("Emotion Detection")
        predict_emotion,predict_prob = helper.emotion_detection(selected_user,df)
        col1,col2=st.columns(2)
        with col1:
            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[predict_emotion]
            st.write("{}:{}".format(predict_emotion, emoji_icon))
            st.write("Confidence:{}".format(np.max(predict_prob)))
            
        with col2:
            st.success("Prediction Probability")
            #st.write(probability)
            proba_df = pd.DataFrame(predict_prob, columns=pipe_lr.classes_)
            #st.write(proba_df.T)
            proba_df_transposed = proba_df.T.reset_index()
            proba_df_transposed.columns = ["emotions", "probability"]


            fig = alt.Chart(proba_df_transposed).mark_bar().encode(x='emotions', y='probability', color='emotions')

            st.altair_chart(fig, use_container_width=True)


            
        col1,col2=st.columns(2)
        with col1:
            #Timeline Analysis (monthly)
            st.title('Monthly Timeline')
            timeline=helper.monthly_timeline(selected_user,df)
            fig,ax=plt.subplots()
            ax.plot(timeline['time'],timeline['message'],color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            #Timeline Analysis (daily)
            st.title('Daily Timeline')
            daily_timeline=helper.daily_timeline(selected_user,df)
            fig,ax=plt.subplots()
            ax.plot(daily_timeline['only_date'],daily_timeline['message'],color='black')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        
        
        #Activity Map
        st.title('Activity Map')
        col1,col2=st.columns(2)
        with col1:
            week_activity_df=helper.week_activity_map(selected_user,df)
            st.header('Most Busy Day')
            fig,ax=plt.subplots()
            ax.bar(week_activity_df.index,week_activity_df.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            month_activity_df=helper.month_activity_map(selected_user,df)
            st.header('Most Busy Month')
            fig,ax=plt.subplots()
            ax.bar(month_activity_df.index,month_activity_df.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        
        st.title('Weekly Activity Map')    
        user_heatmap=helper.activity_heatmap(selected_user,df)
        fig,ax=plt.subplots()
        ax=sns.heatmap(user_heatmap)
        st.pyplot(fig)
        
        


        