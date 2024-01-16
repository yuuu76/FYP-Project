import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
import plotly.express as px
import requests
import json
import re
import time
import streamlit.components.v1 as components
import matplotlib.image as mpimg

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bertopic import BERTopic
from umap import UMAP
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from streamlit_timeline import timeline

#========================================#
# Load the BERTopic model and data incase
with open('final_bertopic_model.pkl', 'rb') as file:
    bertopic_model = pickle.load(file)

bertopic_config = pickle.load(open('bertopic_config_model.pkl','rb'))
    
CNN = pd.read_csv('CNN_Articles_modified.csv')

merged_CNN = pickle.load(open('merged_cnn.pkl','rb'))

TOT = pickle.load(open('TOT_model.pkl','rb'))

trained_data = pickle.load(open('infomative.pkl','rb'))

with open('bertopic_visuals.pkl', 'rb') as file:
    loaded_visuals = pickle.load(file)

# Access individual objects from the loaded dictionary
distance_map_graph = loaded_visuals['distance_map']
bar_chart = loaded_visuals['bar_chart']
topic_similarity_graph = loaded_visuals['topic_similarity']
term_scoring_graph = loaded_visuals['term_scoring']
topic_over_time = loaded_visuals['topic_over_time']

#========================================#

CNN['tokenized_cleaned_bert'] = CNN['tokenized_cleaned_bert'].astype(str)

#================Functions========================#


#graph of topic models comparison
def model_compare():
    img = mpimg.imread('models_comparison.png')

    # Create the figure without setting figsize directly
    fig, ax = plt.subplots()
    
    # Set the size of the figure using set_figwidth and set_figheight
    fig.set_figwidth(5)
    fig.set_figheight(3)

    ax.imshow(img)
    ax.axis('off') 

    st.pyplot(fig)


#pre-processing technique for BERTopic model
def cleaned_for_bert(text):
    lemmatizer = WordNetLemmatizer()
    
    # Ensure the input is a string
    if not isinstance(text, str):
        return ""
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Convert tokens to lowercase and lemmatize
    tokens = [lemmatizer.lemmatize(token.lower(), pos='v') for token in tokens]
    
    # Join tokens back into a single string
    cleaned_text = ' '.join(tokens)
    
    # Remove numbers, punctuations, and single characters
    cleaned_text = re.sub(r'/d+', '', cleaned_text)
    cleaned_text = re.sub(r'[^/w/s]', '', cleaned_text)
    cleaned_text = ' '.join([word for word in cleaned_text.split() if len(word) > 1])
    
    return cleaned_text


# function for user to seacrh topics
# def find_relevant_articles_based_on_topics(model, data, dataframe):
#     # Find similar topics
#     similar_topics, _ = model.find_topics(data, top_n=10)
    
#     # Create a structured format (list of dictionaries) from the similar topics output
#     structured_similar_topics = [{'Similar term': topic[0], 'Weight': topic[1]} for topic in model.get_topic(similar_topics[0])]
    
#     # Create a list to store matched rows
#     matched_rows = []
    
#     for topic in structured_similar_topics:
#         similar_term = topic['Similar term']
#         matched_row = merged_CNN[merged_CNN['Name'].str.contains(similar_term, case=False, na=False)]  # Match based on 'Name' column
#         if not matched_row.empty:
#             matched_rows.append({
#                 'Similar term': similar_term,
#                 'Weight': topic['Weight'],
#                 'Representation': matched_row['Name'].iloc[0],
#                 'Topic': matched_row['Topic'].iloc[0],
#                 'Section': matched_row['Representative_Section'].iloc[0]
#             })

#     # Display the matched rows in a table
#     if matched_rows:
#         df_matched = pd.DataFrame(matched_rows)
#         st.table(df_matched)
#     else:
#         st.write("No matching information found in the merged_CNN dataframe for the similar topics.")

def find_relevant_articles_based_on_topics(model, data, dataframe):
    # Find similar topics
    similar_topics, _ = model.find_topics(data, top_n=10)

    # Create a structured format (list of dictionaries) from the similar topics output
    structured_similar_topics = [{'Similar term': topic[0], 'Weight': topic[1]} for topic in model.get_topic(similar_topics[0])]

    # Create a list to store matched rows
    matched_rows = []

    for topic in structured_similar_topics:
        similar_term = topic['Similar term']
        matched_row = merged_CNN[merged_CNN['Name'].str.contains(similar_term, case=False, na=False)]  # Match based on 'Name' column
        if not matched_row.empty:
            matched_rows.append({
                'Similar term': similar_term,
                'Weight': topic['Weight'],
                'Representation': matched_row['Name'].iloc[0],
                'Topic': matched_row['Topic'].iloc[0],
                'Section': matched_row['Representative_Section'].iloc[0]
            })

    # Display the matched rows in a horizontally scrollable table
    if matched_rows:
        df_matched = pd.DataFrame(matched_rows)
        st.dataframe(df_matched.style.set_sticky(axis="both", max_elements_to_sticky=1))
    else:
        st.write("No matching information found in the merged_CNN dataframe for the similar topics.")



#defining the json file structure
def dataframe_to_timeline_json(df):
    """
    Convert dataframe to a JSON structure suitable for the timeline visualization.
    """
    events_list = []
    for index, row in df.iterrows():
        # try:
            event_date = pd.to_datetime(row['Date published'])
            event = {
                "start_date": {
                    "year": str(event_date.year),
                    "month": str(event_date.month),
                    "day": str(event_date.day)
                },
                "text": {
                    "headline": row['Section'],
                    "text": row['Description'],
                    "full_article_text": row['Article text']
                }
            }
            events_list.append(event)
        # except Exception as e:
        #     st.error(f"Error processing row: {index}")

    json_items = {
        "title": {
            "media": {
                "url": "",
                "caption": "",
                "credit": ""
            },
            "text": {
                "headline": f"Timeline for Topic {df['Topic'].iloc[0]}",
                "text": ""  # You can add additional description here if needed
            }  
        },
        "events": events_list
    }
    return json_items


#filtering the data based on the section and topic
def filter_data_by_section_and_topic(df, section, topic):
    return df[(df['Representative_Section'] == section) & (df['Topic'] == topic)]


# the trend analysis and visualization for tab1
def trend_func(selected_section):

    filtered_section = merged_CNN[merged_CNN['Representative_Section'] == selected_section]

    unique_topic = filtered_section['Topic'].unique()

    topic_number_input = st.text_input(f'Enter a Topic Number within {selected_section} (Valid values: {", ".join(map(str, unique_topic))}):')

    if topic_number_input:
        try:
            selected_topic = int(topic_number_input)
            if selected_topic in unique_topic:
                
                # Filter DataFrame further based on selected topic
                filtered_df = filter_data_by_section_and_topic(merged_CNN, selected_section, selected_topic)
                total_articles = len(filtered_df)
                selected_representation = filtered_df['Representation'].iloc[0] 

                # Display filtered DataFrame (For simplicity, only showing a subset of columns)
                st.subheader(f"Articles under {selected_section} for Topic Number: {selected_topic}")
                st.write(f"#### Representation, Hidden terms in the topic: {selected_representation}")
                
                filtered_df['Date published'] = pd.to_datetime(filtered_df['Date published'], errors='coerce')
                filtered_df['year'] = filtered_df['Date published'].dt.year.astype(str)
                filtered_df['month'] = filtered_df['Date published'].dt.month.astype(str)
                filtered_df['day'] = filtered_df['Date published'].dt.day.astype(str)
    
                    # st.write(filtered_df['Representation'])
                # st.dataframe(filtered_df[['Date','Description']],use_container_width = True)

                # filtered_df['Date published'] = pd.to_datetime(filtered_df['Date published'])

                #testing for timeline
                json_data_timeline = dataframe_to_timeline_json(filtered_df)
                with open('output_file_timeline.json', 'w') as f:
                    json.dump(json_data_timeline, f)
                with open('output_file_timeline.json', 'r') as f:
                    json_data = f.read()
                st.write(f"Total number of articles: {total_articles}")
                st.write('Timeline for relevant articles:')
                coo1,coo2 = st.columns([4,1])
                with coo1:
                    timeline(json_data, height=600)

                # st.write(test)

                # Additional interactivity: Show full article text when an article is selected
                selected_article_index = st.selectbox('Select an article to view full text:', filtered_df.index, index = None)
                if st.button('Show Full Article'):
                    full_article_text = filtered_df.loc[selected_article_index, 'Article text']
                    st.write(f"Full Article Text:/n{full_article_text}")


            else:
                st.warning(f"Invalid topic number entered. Please enter a valid topic number.")
        except ValueError:
            st.warning(f"Please enter a valid integer topic number.")


#================get the total number of trained topic========================#
def topic_length(model):
    return len(model.get_topics())

#================visuals, graph, charts, table========================#
#get bertopic info
def trained_info(model):
    trained_info_data = model.get_topic_info()
    return trained_info_data

#get bertopic document info
def trained_docinfo(model, data):
    return model.get_document_info(data)


#================visuals, graph, charts, table(default)========================#
# default bertopic visual (hierarchy visualization)
def visual_hierarchy():
#    renamed_topics = new_topic_list()
    hierar_topics = bertopic_model.hierarchical_topics(CNN['tokenized_cleaned_bert'])

    st.plotly_chart(bertopic_model.visualize_hierarchy(hierarchical_topics=hierar_topics, orientation = 'left'))


#default bertopic Topic Over Time
def trend_analysis():
        st.plotly_chart(bertopic_model.visualize_topics_over_time(TOT, width = 1400, height = 550, title = '<b>BERTopic over time</b>'))

#distance map
def distance_map():
    st.plotly_chart(bertopic_model.visualize_topics(width = 1000))

#topic bar chart
def topic_barchart():
    st.plotly_chart(bertopic_model.visualize_barchart(top_n_topics = 20, width = 200, height = 200))

#visualize doc and topic in 2D
def clus_doctopic():
    st.plotly_chart(bertopic_model.visualize_documents(CNN['tokenized_cleaned_bert']))

#topic similarity (heatmap)
def heatmap_topic():
    st.plotly_chart(bertopic_model.visualize_heatmap(top_n_topics = 20, width = 1000, height = 1000)
)
#ranks of all terms across all topics
def scoring_topic():
    st.plotly_chart(bertopic_model.visualize_term_rank(width = 1300))


#the original CNN dataset
def cnn_datatable():
    columns_to_display = ['Date published', 'Day_of_Week', 'Section', 'Headline', 'Description', 'Article text']
    st.dataframe(CNN[columns_to_display].head(100))



#================Discovery(trained new article using the model)========================#

#get the news article domain
def get_domain_name(link):
    # Check if the input is empty
    if not link:
        return "Please Enter a URL"
    
    try:
        # Parse the URL
        parsed_url = urlparse(link)

        # Check if the netloc (domain) is empty or not
        if not parsed_url.netloc:
            return 'Invalid URL Format'

        # Split the domain by dots and get the first part
        domain_name = parsed_url.netloc.split('.')[0]

        # If the Website starts with 'www', extract the subsequent part
        if domain_name == 'www' and len(parsed_url.netloc.split('.')) > 1:
            domain_name = parsed_url.netloc.split('.')[1]

        return domain_name.capitalize()

    except ValueError:
        return 'Invalid URL Format, Please enter a correct URL'
    
# Send an HTTP request to the URL of the webpage we want to access
# Defining Function for Requesting Access to link/Establishing Connection
def establish_Connection(link):

    try:
        # Connecting to Website
        r = requests.get(link)
        # Create a BeautifulSoup object and parse the HTML content
        # lxml is capable of parsing both HTML and XML Content
        soup = BeautifulSoup(r.content, 'lxml')
        # Returning Soup object to use it later
        return soup

    except:
            pass

# Scraping Text data from Website 
def save_to_file(text, fname):

    if text is not None:
        st.download_button(
        label="Download Text File",
        data='/n'.join(text),
        file_name=fname,
        key="download_button",
        )
    else:
        st.write("Website has No Data!!")


# Remove any invalid control characters from the JSON string
def sanitize_json(json_str):
    return ''.join(char for char in json_str if char.isprintable())


# function to scrape the data
def extract_article_info(url):
    try:
        soup = establish_Connection(url)

        # Extracting date_published, article_text, category, and headline using appropriate CSS selectors
        date_published = soup.select_one('meta[property="article:published_time"]')['content'] if soup.select_one('meta[property="article:published_time"]') else "Date not found"
        
        article_text = soup.select_one('div.article-body').text.strip() if soup.select_one('div.article-body') else "Article text not found"

        category = soup.select_one('meta[property="article:section"]')['content'] if soup.select_one('meta[property="article:section"]') else "Category not found"
        
        headline = soup.select_one('meta[property="og:title"]')['content'] if soup.select_one('meta[property="og:title"]') else "Headline not found"

        # Fallback to JSON-LD content if data not found using BeautifulSoup
        json_ld_script = soup.select_one('script[type="application/ld+json"]')
        
        if json_ld_script:
            sanitized_json_str = sanitize_json(json_ld_script.string)
            json_ld_content = json.loads(sanitized_json_str)
            
            date_published = json_ld_content.get("datePublished", date_published)
            article_text = json_ld_content.get("articleBody", article_text)
            
            # Use the list directly if available; otherwise, fallback to the single category
            categories_list = json_ld_content.get("articleSection", [])
            category = categories_list if categories_list else [category]

        return {
            'Date published': date_published,
            'Article text': article_text,
            'Section': category,
            'Headline': headline
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# functions to fit the new news data to the trained model
def fitinto_trainedmodel(model, data, merged_CNN):
    # Transform the new article data to get the topics
    topics, prob = model.transform(data)
    
    # Display the identified topics for the new article
    st.write(f'### The article has been identified to fit into Topic(s): {topics}')
    
    # Check if any topics are identified
    if topics:
        # Extract the first topic from the list
        first_topic = topics[0]
        
        # Filter the merged_CNN DataFrame based on the first identified topic
        filtered_df = merged_CNN[merged_CNN['Topic'] == first_topic]
        
        # Extract the representative terms from the filtered DataFrame
        representative_terms = filtered_df['Representation'].tolist()

        unique_terms = list(set([item for sublist in representative_terms for item in sublist]))
        
        # Display the representative terms for the first identified topic
        if representative_terms:
            st.write(f"#### Representative Term(s) for First Identified Topic: {unique_terms}")

            # Display the timeline chart
            # Convert dates to the required format
            filtered_df['Date published'] = pd.to_datetime(filtered_df['Date published'])
            filtered_df['year'] = filtered_df['Date published'].dt.year.astype(str)
            filtered_df['month'] = filtered_df['Date published'].dt.month.astype(str)
            filtered_df['day'] = filtered_df['Date published'].dt.day.astype(str)

            # Create the events list
            events_list = []
            for index, row in filtered_df.iterrows():
                event = {
                    "start_date": {
                        "year": row['year'],
                        "month": row['month'],
                        "day": row['day']
                    },
                    "text": {
                        "headline": row['Section'],
                        "text": row['Description']
                    }
                }
                events_list.append(event)

            # Create the JSON structure
            json_items = {
                "title": {
                    "media": {
                        "url": "",
                        "caption": "",
                        "credit": ""
                    },
                    "text": {
                        "headline": (f"Trend for Topic:{topics}"),
                        "text": unique_terms ,
                    }
                },
                "events": events_list
            }


            with open('output_file.json', 'w') as f:
                json.dump(json_items, f)
            with open('output_file.json', 'r') as f:
                json_data = f.read()

            timeline(json_data, height=500)
            # st.subheader('Selected content')
            # st.write(timeline)

            
        else:
            st.error('No representative terms found for the identified topic, try another link')
    else:
        st.error('No topics identified for the article.')
    
    return


# display of the timeline chart
def display_timeline_chart(df):
    # Assuming your DataFrame has columns 'Time' and 'Description'
    fig = px.timeline(df, x_start='Time', x_end='Time', y='Description', title='Timeline of Articles')
    
    # Customize the appearance of the timeline chart
    fig.update_layout(xaxis_title='Time', yaxis_title='Description', showlegend=False)
    
    # Display the chart
    st.plotly_chart(fig)





