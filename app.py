import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import base64
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="RedEx - AI Supply Chain Assistant", page_icon="images/logo.png", layout="wide")

# Background setup function
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    background_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{image_data}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# Set the background
set_background("images/background.png")

# Function to initialize RAG with the dataset
def initialize_rag(df):
    # Convert dataframe to text for RAG
    text_data = df.to_string()
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_text(text_data)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
    
    # Create conversation chain
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, model_name="gpt-4"),
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    
    return conversation_chain

# Sidebar setup
with st.sidebar:
    st.image('images/logo.png')
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    
    # API key validation
    if not (openai.api_key.startswith('sk-') and len(openai.api_key) == 164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        os.environ["OPENAI_API_KEY"] = openai.api_key
        st.success('Ready to assist with your supply chain queries!', icon='👉')
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV file):", type=["csv"])
    if uploaded_file:
        try:
            # Try different encoding and separator options
            try:
                dataframed = pd.read_csv(uploaded_file, encoding='utf-8')
            except:
                try:
                    dataframed = pd.read_csv(uploaded_file, encoding='latin1')
                except:
                    dataframed = pd.read_csv(uploaded_file, sep=';')  # Try semicolon separator
            
            if dataframed.empty:
                st.error("The uploaded file appears to be empty.")
            else:
                st.success("Dataset uploaded successfully! Here's a preview:")
                # Display basic dataset info
                st.write("Dataset Info:")
                st.write(f"- Number of rows: {len(dataframed)}")
                st.write(f"- Number of columns: {len(dataframed.columns)}")
                st.write("- Columns:", ", ".join(dataframed.columns.tolist()))
                st.dataframe(dataframed.head())
                # Initialize RAG when dataset is uploaded
                st.session_state.conversation_chain = initialize_rag(dataframed)
        except Exception as e:
            st.error(f"Failed to read the dataset. Error details: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted and contains data.")
    else:
        dataframed = None  # No dataset uploaded

    options = option_menu(
        "RedEx", 
        ["Home", "Chat Assistant", "Data Visualization"],
        icons=['house', 'chat-dots', 'bar-chart-fill'],
        menu_icon="truck",
        default_index=0,
        styles={
            "icon": {"color": "#dec960", "font-size": "20px"},
            "nav-link": {"font-size": "17px", "text-align": "left", "margin": "5px", "--hover-color": "#262730"},
            "nav-link-selected": {"background-color": "#262730"}          
        })

# Custom CSS for chat interface
st.markdown("""
    <style>
    .user-message {
        background-color: rgba(255, 255, 255, 0.25) !important;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        margin-left: auto;
        margin-right: 0;
        max-width: 80%;
        text-align: right;
        font-weight: 600;
        color: white;
    }
    .assistant-message {
        background-color: rgba(255, 255, 255, 0.25) !important;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        margin-right: auto;
        margin-left: 0;
        max-width: 80%;
        text-align: left;
        font-weight: 600;
        color: white;
    }
    .stChatMessage > div {
        background-color: transparent !important;
    }
    .stChatMessage [data-testid="UserAvatar"] {
        float: right;
    }
    
    /* Chat input field styling */
    .stChatInput {
        background-color: rgba(255, 255, 255, 0.25) !important;
        border-radius: 10px;
        padding: 10px;
        backdrop-filter: blur(5px);
    }
    
    .stChatInput input {
        color: white !important;
        font-weight: 500 !important;
    }
    
    .stChatInput::placeholder {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Add these custom styles after the existing CSS
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: rgba(251, 251, 251, 0.05);
        border-radius: 10px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #dec960;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #c4b052;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Select box styling */
    .stSelectbox {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
        backdrop-filter: blur(5px);
    }
    
    /* Chat container styling */
    .chat-container {
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    
    /* Message styling improvements */
    .user-message, .assistant-message {
        background-color: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(5px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .user-message:hover, .assistant-message:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Visualization container styling */
    .viz-container {
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    
    /* Chart styling */
    .plotly-chart {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        backdrop-filter: blur(5px);
    }
    
    /* Input field styling */
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
        backdrop-filter: blur(5px);
    }
    
    /* File uploader styling */
    .stFileUploader {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 20px;
        backdrop-filter: blur(5px);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Initialize conversation chain in session state
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None

if options == "Home":
    st.markdown("""
        <div style='background-color: rgba(0,0,0,0.8); 
                    padding: 30px; 
                    border-radius: 15px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    backdrop-filter: blur(10px);
                    margin-bottom: 20px;'>
            <h1 style='text-align: center; 
                       color: #dec960; 
                       margin: 0;
                       font-size: 2.5em;'>
                RedEx - Your AI Supply Chain Assistant
            </h1>
            <p style='text-align: center; color: white; margin-top: 10px;'>
                Empowering Supply Chain Excellence Through AI
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features section with columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image('images/chat.png', width=150, use_column_width=True)
        st.markdown("""
            <div style='background-color: rgba(0,0,0,0.7); padding: 20px; border-radius: 10px;'>
                <h3 style='color: #dec960;'>Interactive AI Assistant</h3>
                <p style='color: white;'>Get instant insights and answers about your supply chain data through our AI-powered chat interface.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image('images/analytics.png', width=150, use_column_width=True)
        st.markdown("""
            <div style='background-color: rgba(0,0,0,0.7); padding: 20px; border-radius: 10px;'>
                <h3 style='color: #dec960;'>Data Visualization</h3>
                <p style='color: white;'>Transform complex supply chain data into clear, actionable visualizations for better decision-making.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.image('images/insights.png', width=150, use_column_width=True)
        st.markdown("""
            <div style='background-color: rgba(0,0,0,0.7); padding: 20px; border-radius: 10px;'>
                <h3 style='color: #dec960;'>Smart Insights</h3>
                <p style='color: white;'>Receive AI-driven recommendations for optimizing inventory, reducing costs, and improving efficiency.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Getting Started section
    st.markdown("---")
    st.markdown("""
        <div style='background-color: rgba(0,0,0,0.7); padding: 20px; border-radius: 10px;'>
            <h3 style='color: #dec960;'>Getting Started</h3>
            <ol style='color: white;'>
                <li>Enter your OpenAI API key in the sidebar</li>
                <li>Upload your supply chain dataset (CSV format)</li>
                <li>Use the Chat Assistant for detailed analysis and insights</li>
                <li>Explore Data Visualization for visual analytics</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    # Key Features section
    st.markdown("""
        <div style='background-color: rgba(0,0,0,0.7); padding: 20px; border-radius: 10px; margin-top: 20px;'>
            <h3 style='color: #dec960;'>Key Features</h3>
            <ul style='color: white;'>
                <li>Supply chain optimization recommendations</li>
                <li>Inventory management insights</li>
                <li>Cost reduction strategies</li>
                <li>Performance analysis and benchmarking</li>
                <li>Demand forecasting assistance</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Pro Tip with matching background
    st.markdown("""
        <div style='background-color: rgba(0,0,0,0.7); padding: 20px; border-radius: 10px; margin-top: 20px;'>
            <p style='color: white; margin: 0;'>💡 Pro Tip: For the best experience, ensure your dataset includes key supply chain metrics such as inventory levels, lead times, and shipping data.</p>
        </div>
    """, unsafe_allow_html=True)

elif options == "Chat Assistant":
    # Title container for Chat Assistant
    st.markdown("""
        <div style='background-color: rgba(0,0,0,0.8); 
                    padding: 30px; 
                    border-radius: 15px; 
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    backdrop-filter: blur(10px);
                    margin-bottom: 20px;'>
            <h1 style='text-align: center; 
                       color: #dec960; 
                       margin: 0;
                       font-size: 2.5em;'>
                RedEx - Your AI Supply Chain Assistant
            </h1>
        </div>
    """, unsafe_allow_html=True)

    # Introduction text
    st.markdown("""
        <div style='background-color: rgba(0,0,0,0.8); 
                    padding: 30px; 
                    border-radius: 15px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    backdrop-filter: blur(10px);
                    margin-bottom: 20px;'>
            <p style='font-size: 1.2em; color: #ffffff;'>
                Welcome to RedEx, your advanced Supply Chain and Logistics Intelligence Assistant. 
                I'm here to help you with:
            </p>
            <ul style='font-size: 1.1em; color: #dec960;'>
                <li>Supply chain optimization</li>
                <li>Logistics planning and execution</li>
                <li>Inventory management</li>
                <li>Cost reduction strategies</li>
                <li>Performance analysis</li>
            </ul>
            <p style='font-size: 1.2em; color: #ffffff; margin-top: 20px;'>
                How can I assist you today?
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Display chat messages
    for message in st.session_state.messages:
        message_class = "user-message" if message["role"] == "user" else "assistant-message"
        with st.chat_message(message["role"], avatar="images/user.png" if message["role"] == "user" else "images/red.png"):
            st.markdown(f'<div class="{message_class}">{message["content"]}</div>', unsafe_allow_html=True)

    # Update the chat response handling
    if prompt := st.chat_input("Ask RedEx about your supply chain challenges..."):
        if not openai.api_key:
            st.error("Please enter your OpenAI API key in the sidebar first!")
        elif dataframed is None:
            st.error("Please upload a dataset first!")
        else:
            with st.chat_message("user", avatar="images/user.png"):
                st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
            
            st.session_state.messages.append({"role": "user", "content": prompt})

            try:
                # Use RAG for response generation
                if st.session_state.conversation_chain is None:
                    st.session_state.conversation_chain = initialize_rag(dataframed)
                
                response = st.session_state.conversation_chain({"question": prompt})
                response_text = response['answer']

                with st.chat_message("assistant", avatar="images/red.png"):
                    st.markdown(f'<div class="assistant-message">{response_text}</div>', unsafe_allow_html=True)
                
                st.session_state.messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.warning("Please check your API key and try again.")

# Data Visualization Section
elif options == "Data Visualization":
    if not openai.api_key:
        st.error("Please enter your OpenAI API key in the sidebar first!")
    elif dataframed is None:
        st.error("Please upload a dataset first!")
    else:
        # Title container with consistent styling
        st.markdown("""
            <div style='background-color: rgba(0,0,0,0.8); 
                        padding: 30px; 
                        border-radius: 15px; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        backdrop-filter: blur(10px);
                        margin-bottom: 20px;'>
                <h2 style='text-align: center; 
                           color: #dec960; 
                           margin: 0;
                           font-size: 2em;'>
                    Data Visualization Dashboard
                </h2>
            </div>
        """, unsafe_allow_html=True)

        # Wrap tabs in a container with backdrop blur
        st.markdown("""
            <div style='
                background-color: rgba(0,0,0,0.7);
                padding: 20px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
                margin-bottom: 20px;
            '>
        """, unsafe_allow_html=True)
        
        viz_tabs = st.tabs(["Overview", "Time Series", "Relationships"])

        with viz_tabs[0]:
            st.markdown("""
                <div style='
                    background-color: rgba(0,0,0,0.6);
                    padding: 20px;
                    border-radius: 10px;
                    backdrop-filter: blur(5px);
                    margin-bottom: 15px;
                '>
                    <h3 style='color: #dec960; margin-bottom: 15px;'>Dataset Overview</h3>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    <div style='
                        background-color: rgba(0,0,0,0.6);
                        padding: 20px;
                        border-radius: 10px;
                        backdrop-filter: blur(5px);
                    '>
                        <h4 style='color: #dec960; margin-bottom: 15px;'>Dataset Statistics</h4>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                            <div style='
                                background-color: rgba(0,0,0,0.4);
                                padding: 8px;
                                margin: 5px 0;
                                border-radius: 5px;
                                backdrop-filter: blur(5px);
                            '>
                                <p style='margin: 0; color: white; font-weight: bold;'>📊 Total Records: {len(dataframed)}</p>
                            </div>
                            <div style='
                                background-color: rgba(0,0,0,0.4);
                                padding: 8px;
                                margin: 5px 0;
                                border-radius: 5px;
                                backdrop-filter: blur(5px);
                            '>
                                <p style='margin: 0; color: white; font-weight: bold;'>📋 Total Features: {len(dataframed.columns)}</p>
                            </div>
                            <div style='
                                background-color: rgba(0,0,0,0.4);
                                padding: 8px;
                                margin: 5px 0;
                                border-radius: 5px;
                                backdrop-filter: blur(5px);
                            '>
                                <p style='margin: 0; color: white; font-weight: bold;'>🔢 Numeric Columns: {len(dataframed.select_dtypes(include=['int64', 'float64']).columns)}</p>
                            </div>
                            <div style='
                                background-color: rgba(0,0,0,0.4);
                                padding: 8px;
                                margin: 5px 0;
                                border-radius: 5px;
                                backdrop-filter: blur(5px);
                            '>
                                <p style='margin: 0; color: white; font-weight: bold;'>📝 Categorical Columns: {len(dataframed.select_dtypes(include=['object']).columns)}</p>
                            </div>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                    <div style='
                        background-color: rgba(0,0,0,0.6);
                        padding: 20px;
                        border-radius: 10px;
                        backdrop-filter: blur(5px);
                    '>
                        <h4 style='color: #dec960; margin-bottom: 15px;'>Data Quality</h4>
                """, unsafe_allow_html=True)
                missing_data = dataframed.isnull().sum()
                if missing_data.any():
                    st.markdown("""
                        <div style='
                            background-color: rgba(0,0,0,0.4);
                            padding: 8px;
                            margin: 5px 0;
                            border-radius: 5px;
                            backdrop-filter: blur(5px);
                        '>
                            <p style='margin: 0; color: white; font-weight: bold;'>Missing Values:</p>
                        </div>
                    """, unsafe_allow_html=True)
                    for col, count in missing_data[missing_data > 0].items():
                        st.markdown(f"""
                            <div style='
                                background-color: rgba(0,0,0,0.4);
                                padding: 8px;
                                margin: 5px 0;
                                border-radius: 5px;
                                backdrop-filter: blur(5px);
                            '>
                                <p style='margin: 0; color: white; font-weight: bold;'>❗ {col}: {count} missing values</p>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style='
                            padding: 8px;
                            margin: 5px 0;
                            border-radius: 5px;
                        '>
                            <p style='margin: 0; color: white;'>✅ No missing values found in the dataset!</p>
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with viz_tabs[1]:
            st.markdown(f"""
                <div style='
                    background-color: rgba(0,0,0,0.6);
                    padding: 20px;
                    border-radius: 10px;
                    backdrop-filter: blur(5px);
                    margin-bottom: 15px;
                '>
                    <h3 style='color: #dec960; margin-bottom: 15px;'>Time Series Analysis</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Detect datetime columns
            datetime_cols = dataframed.select_dtypes(include=['datetime64']).columns
            if not datetime_cols.empty:
                date_col = st.selectbox("Select Date Column:", datetime_cols)
            else:
                # Try to identify potential date columns from object types
                potential_date_cols = [col for col in dataframed.columns if any(date_keyword in col.lower() 
                    for date_keyword in ['date', 'time', 'year', 'month'])]
                if potential_date_cols:
                    date_col = st.selectbox("Select Date Column:", potential_date_cols)
                    # Convert to datetime
                    try:
                        dataframed[date_col] = pd.to_datetime(dataframed[date_col])
                    except:
                        st.warning("Selected column could not be converted to date format.")
                        date_col = None
                else:
                    st.warning("No datetime columns detected in the dataset.")
                    date_col = None

            if date_col:
                # Select numeric column for time series
                numeric_cols = dataframed.select_dtypes(include=['int64', 'float64']).columns
                if not numeric_cols.empty:
                    value_col = st.selectbox("Select Value to Plot:", numeric_cols)
                    
                    # Aggregate data by date
                    time_series_data = dataframed.groupby(date_col)[value_col].sum().reset_index()
                    
                    # Create time series plot
                    fig = px.line(time_series_data, x=date_col, y=value_col,
                                title=f'{value_col} Over Time',
                                template='plotly_dark')
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0.8)',
                        paper_bgcolor='rgba(0,0,0,0.8)',
                        font_color='#ffffff'
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with viz_tabs[2]:
            st.markdown(f"""
                <div style='
                    background-color: rgba(0,0,0,0.6);
                    padding: 20px;
                    border-radius: 10px;
                    backdrop-filter: blur(5px);
                    margin-bottom: 15px;
                '>
                    <h3 style='color: #dec960; margin-bottom: 15px;'>Relationship Analysis</h3>
                </div>
            """, unsafe_allow_html=True)
            
            numeric_cols = dataframed.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_axis = st.selectbox("Select X-axis:", numeric_cols)
                with col2:
                    y_axis = st.selectbox("Select Y-axis:", [col for col in numeric_cols if col != x_axis])
                
                # Create scatter plot
                fig = px.scatter(dataframed, x=x_axis, y=y_axis,
                               title=f'Relationship between {x_axis} and {y_axis}',
                               template='plotly_dark')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0.8)',
                    paper_bgcolor='rgba(0,0,0,0.8)',
                    font_color='#ffffff'
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)
