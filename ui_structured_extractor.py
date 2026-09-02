import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
import streamlit as st
import json

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🎬 AI Movie Data Extractor",
    page_icon="🎬",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .info-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .field-label {
        font-weight: 600;
        color: #667eea;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    .field-value {
        color: #333;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.75rem;
        border: none;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
    .genre-tag {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .cast-member {
        background: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.3rem;
        display: inline-block;
        color: #333;
    }
    .example-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .example-card:hover {
        border-color: #667eea;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎬 AI Movie Data Extractor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Structured extraction using Pydantic + LangChain</p>', unsafe_allow_html=True)

# SCHEMA
class MovieInfo(BaseModel):
    title: str
    release_year: Optional[int] = None
    genres: Optional[List[str]] = None
    director: Optional[str] = None
    cast: Optional[List[str]] = None
    rating: Optional[float] = None
    summary: Optional[str] = None

# Initialize model
@st.cache_resource
def get_model():
    return ChatMistralAI(
        model="open-mistral-7b",
        temperature=0.3,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )

# Create parser and prompt
@st.cache_resource
def get_parser_and_prompt():
    parser = PydanticOutputParser(pydantic_object=MovieInfo)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a movie information extractor. You will be given a text description of a movie, and your task is to extract the following information from it:

{format_instructions}""",
        ),
        ("human", "{paragraph}"),
    ])

    return parser, prompt

# Sidebar with examples
with st.sidebar:
    st.header("📖 Example Movies")

    examples = {
        "Inception": """Inception (2010) is a science fiction action film directed by Christopher Nolan, starring Leonardo DiCaprio, Joseph Gordon-Levitt, and Ellen Page. It follows Dom Cobb, a skilled thief who steals corporate secrets by infiltrating people's subconscious through dream-sharing technology. When offered a chance to have his criminal past erased, Cobb must perform the nearly impossible task of "inception" — planting an idea into the mind of a powerful CEO rather than stealing one. He assembles a team of specialists and descends through layers of dreams, while his guilt over his late wife's death threatens to unravel the entire mission.""",

        "The Matrix": """The Matrix (1999) is a groundbreaking science fiction action film directed by the Wachowskis, starring Keanu Reeves, Laurence Fishburne, and Carrie-Anne Moss. The film follows Neo, a computer hacker who discovers that humanity is trapped in a simulated reality created by sentient machines. Neo joins a group of rebels led by Morpheus to fight against the machines and free humanity. The film is known for its revolutionary visual effects, philosophical themes, and iconic action sequences.""",

        "Parasite": """Parasite (2019) is a South Korean black comedy thriller directed by Bong Joon-ho, starring Song Kang-ho, Lee Sun-kyun, and Cho Yeo-jeong. The film follows the poor Kim family as they scheme to become employed by the wealthy Park family by infiltrating their household through deception. As the Kims become entangled in the Parks' lives, a shocking discovery leads to a violent and tragic confrontation between the two families. The film won the Palme d'Or at Cannes and became the first non-English language film to win Best Picture at the Academy Awards.""",
    }

    for movie_name, movie_text in examples.items():
        st.markdown(f"**{movie_name}**")
        if st.button(f"Use {movie_name}", key=f"btn_{movie_name}"):
            st.session_state.input_text = movie_text

    st.divider()
    st.markdown("### ⚙️ Tech Stack")
    st.info("🤖 **Model:** Mistral AI (open-mistral-7b)")
    st.info("🔗 **Framework:** LangChain + Pydantic")
    st.info("🎨 **UI:** Streamlit")

# Main content area
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📝 Input Movie Description")

    # Initialize session state
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""

    movie_text = st.text_area(
        "Paste or type a movie description:",
        value=st.session_state.input_text,
        height=350,
        placeholder="Example: The Shawshank Redemption (1994) is an American drama directed by Frank Darabont...",
        key="text_input"
    )

    extract_button = st.button("🔍 Extract Structured Data", use_container_width=True)

with col2:
    st.subheader("📊 Extracted Information")

    if extract_button:
        if not movie_text or movie_text.strip() == "":
            st.warning("⚠️ Please enter a movie description first!")
        else:
            with st.spinner("🤖 AI is analyzing and structuring the data..."):
                try:
                    # Get components
                    model = get_model()
                    parser, prompt = get_parser_and_prompt()

                    # Create chain
                    chain = prompt | model | parser

                    # Invoke
                    result = chain.invoke({
                        "paragraph": movie_text,
                        "format_instructions": parser.get_format_instructions()
                    })

                    # Display structured result
                    st.markdown('<div class="info-card">', unsafe_allow_html=True)

                    # Title
                    st.markdown(f'<div class="field-label">🎬 TITLE</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="field-value" style="font-size: 1.5rem; font-weight: 700;">{result.title}</div>', unsafe_allow_html=True)

                    # Release Year
                    if result.release_year:
                        st.markdown(f'<div class="field-label">📅 RELEASE YEAR</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="field-value">{result.release_year}</div>', unsafe_allow_html=True)

                    # Director
                    if result.director:
                        st.markdown(f'<div class="field-label">🎥 DIRECTOR</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="field-value">{result.director}</div>', unsafe_allow_html=True)

                    # Genres
                    if result.genres:
                        st.markdown(f'<div class="field-label">🎭 GENRES</div>', unsafe_allow_html=True)
                        genres_html = "".join([f'<span class="genre-tag">{genre}</span>' for genre in result.genres])
                        st.markdown(f'<div style="margin-bottom: 1rem;">{genres_html}</div>', unsafe_allow_html=True)

                    # Cast
                    if result.cast:
                        st.markdown(f'<div class="field-label">👥 CAST</div>', unsafe_allow_html=True)
                        cast_html = "".join([f'<span class="cast-member">{member}</span>' for member in result.cast])
                        st.markdown(f'<div style="margin-bottom: 1rem;">{cast_html}</div>', unsafe_allow_html=True)

                    # Rating
                    if result.rating:
                        st.markdown(f'<div class="field-label">⭐ RATING</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="field-value">{result.rating}/10</div>', unsafe_allow_html=True)

                    # Summary
                    if result.summary:
                        st.markdown(f'<div class="field-label">📖 SUMMARY</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="field-value">{result.summary}</div>', unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                    # Success message
                    st.success("✅ Data extracted and structured successfully!")

                    # Show raw JSON
                    with st.expander("📄 View Raw JSON"):
                        st.json(result.model_dump())

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Make sure your MISTRAL_API_KEY is set correctly in the .env file")
    else:
        st.info("👈 Enter a movie description and click 'Extract Structured Data' to get started!")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Powered by <strong>Mistral AI</strong> & <strong>LangChain</strong> with <strong>Pydantic</strong> validation | Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
