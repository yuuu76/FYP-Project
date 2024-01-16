import streamlit as st
import time
import torch
import matplotlib.pyplot as plt
import plotly.express as px

from functions import topic_length
from functions import *


# Page configuration
st.set_page_config(
    page_title="FYP - BERTopic modeling for News Article",
    page_icon=":sun_with_face:",
    layout="wide",
    initial_sidebar_state="collapsed")

# #========================================#
# # Defining different tabs
def tabs():
    tab1, tab2, tab3= st.tabs(["Home Page","Information","Discovery"])
    with tab1:
        total_topics = topic_length(bertopic_model)

        st.write('# BERTopic model with CNN News Articles')
        st.write(f' <center><h2>{total_topics} Trained Topics Overview:</h2></center>', unsafe_allow_html=True)

        st.dataframe(trained_data)
        # st.dataframe(merged_CNN)

        #next area - showing the visualization by BERTopic model
        st.write('## Default Visuals in BERTopic Model')
        selected_visual = st.multiselect('Choose up to one to visual', ['Bar Chart','Topic Hierarchy Graph','Topic Similarity', 'Topic Over Time','Distance Map', 'Term Score'])
        
        if 'Distance Map' in selected_visual:
            st.write("#### Displaying Distance Map")
            st.write("##### Embedding c-TF-IDF representation of the topics in 2D using Umap")
            distance_map_graph

        if 'Bar Chart' in selected_visual:
            st.write("#### Displaying Top 20-Topic in Bar Chart:")
            st.write("##### Visualize the Top 5 terms for a 20 topics by creating bar charts out of the c-TF-IDF scores,  Insights can be gained from the relative c-TF-IDF scores between and within topics.")
            bar_chart

        if 'Topic Hierarchy Graph' in selected_visual:
            st.write("#### Displaying Topic Hierarchy Graph:")
            st.write("#####  Presenting topic representation at that level of the hierarchy. These representations help you understand the effect of merging certain topics. ")
            visual_hierarchy()

        if 'Topic Similarity' in selected_visual:
            st.write("#### Displaying Topic Similarity in Heatmap:")
            st.write("##### The result will be a matrix indicating how similar certain topics are to each other.")
            topic_similarity_graph

        if 'Topic Over Time' in selected_visual:
            st.write("#### Displaying Topic Over Time:")
            st.write("##### The topics evolution over the period of time")
            topic_over_time
        
        if 'Term Score' in selected_visual:
            st.write("#### Displaying Term Score Decline:")
            st.write("The c-TF-IDF score visualization plots the decline in word importance within topics based on their rank, enabling the determination of the optimal number of words for topic representation using the elbow method.")
            term_scoring_graph

        #next area - let user to find similar topic using keyword
        st.write('## Find Similar Topic')
        term_txt = st.text_input('Search using a term: ')

        if term_txt:
            st.write('#### Top 10 relevant topics will be display below:')
            find_relevant_articles_based_on_topics(bertopic_model, term_txt, merged_CNN)

        #next area
        st.write('## Trend Analysis')
        selected_topic = st.selectbox('Select a Representative Section:', merged_CNN['Representative_Section'].unique(), index = None)

        trend_func(selected_topic)


    with tab2:
        t1col1, t2col2 = st.columns([2,2])
        with t1col1:
            st.markdown("""
                <h2>Aim:</h2>
                <p>By apply topic modelling techniques on news articles and to implement the model, with the objective of easing the problem of information overload that is so prevalent in the vast amount of information available online nowadays.</p>
                        
                <h2>Objectives:</h2>
                <ol>
                    <li>To extract topics from news articles using topic modelling methods and evaluate extracted knowledge.</li>
                    <li>To identify the trend of the news articles based on the extracted topics and terms.</li>
                    <li>To deploy the extracted topics and resulting patterns in an informative and interactive manner.</li>
                </ol>


                <h2>Additional Information:</h2>
                <ul>
                    <li>This model uses the BERTopic algorithm for topic modeling.</li>
                    <li>It was trained using Python and various NLP libraries.</li>
                    <li>Topics were extracted based on semantic similarity between words and phrases.</li>
                </ul>
                        
            """, unsafe_allow_html=True)


        with t2col2:
            st.markdown("<h2>Overview of Original CNN News dataset: </h2> The data used to trained the topic model is available here<a href='https://www.kaggle.com/datasets/hadasu92/cnn-articles-after-basic-cleaning/data'>CNN News Article</a> ", unsafe_allow_html=True)

            cnn_datatable()

        st.write('## Topic Model Trained:')
        st.write ('###### 1. Latent Dirichlet Allocation (LDA)')
        st.write ('###### 2. Latent Semantic Index (LSI)/ Latent Semantic Analysis (LDA)')
        st.write ('###### 3. Non-Negative Matrix Factorization (NMF)')
        st.write ('###### 4. Bidirectional Encoder Representations from Transformers Topic (BERTopic)')

        st.write('## Model Score:')
        st.write ('The final trained and tuned model achieved an coherence score of (c_v ≈ 0.80) and (u_mass ≈ -0.90) on the CNN news articles dataset.')
        model_compare()
           

    with tab3:
        st.write('# Try it with something new')

        t3col1, t3col2 = st.columns([4,1])

        link = st.text_input("### Enter a News Media Website Link Below (recommended test with CNN, MalayMail and SinarDaily):")

        if link:
            domain_name = get_domain_name(link)

        if link:
            article_info = extract_article_info(link)
            if article_info:
                st.write("## Article Information")
                st.write("It comes along with the pre-processed text")
                
                tokenized_cleaned_bert = cleaned_for_bert(article_info['Article text'])
                
                new_data = {
                    'Date published': [article_info['Date published']],
                    'Section': [article_info['Section']],
                    'Headline': [article_info['Headline']],
                    'Article text': [article_info['Article text']],
                    'tokenized_cleaned_bert': [tokenized_cleaned_bert]
                }
                
                the_data = pd.DataFrame(new_data)
                
                st.table(the_data)
                
                if article_info['Article text'] == "Article text not found":
                    st.error("Couldn't fit your data to the model, try another valid link from [CNN], [MalayMail] or [SinarDaily]")

                else:
                    fitdata = st.button('Fit my Article to BERTopic modeling')
                    
                    if fitdata:
                        st.write("Fitting the article to BERTopic model! Please be patient.")
                        progress_bar = st.progress(0)
                        
                        for percent_complete in range(0, 101, 10):
                            time.sleep(0.5)
                            progress_bar.progress(percent_complete) 

                        fitinto_trainedmodel(bertopic_model, the_data['tokenized_cleaned_bert'], merged_CNN)
            else:
                st.error("Failed to extract valid article information. Please enter a valid URL.")
        else:
            st.error("Please enter a URL to fetch the news data.")



#main
def main():
    st.title("Welcome to the page! :sun_with_face:")
    st.markdown(
            """
            This is an interactive topic modeling tool built using [streamlit.io](https://streamlit.io)
            and [BERTopic](https://maartengr.github.io/BERTopic/index.html).
            """)

    tabs()

if __name__ == "__main__":
    main()
