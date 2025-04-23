import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from portfolio import Portfolio
from utils import clean_text

def create_streamlit_app(llm, portfolio, clean_text):
    # Hide Streamlit UI elements
    hide_streamlit_style = """
        <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.title("📧 Cold Mail Generator")
    url_input = st.text_input("Enter a URL:", value="https://jobs.mahindracareers.com/job/Mumbai-Head-Service-Strategy-MUMB/1245261200/")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            with st.spinner("🔄 Fetching job details and generating email..."):
                loader = WebBaseLoader([url_input])
                data = clean_text(loader.load().pop().page_content)
                portfolio.load_portfolio()
                jobs = llm.extract_jobs(data)

                if jobs:
                    job = jobs[0]  # ✅ Take only the first job to prevent looping
                    skills = job.get('skills', [])
                    links = portfolio.query_links(skills)
                    email = llm.write_mail(job, links)
                    st.success("✅ Email generated successfully!")

                    # ✅ Scrollable Markdown Email Output
                    st.markdown(f"""
                        <div style="max-height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: black; border-radius: 5px;">
                            <pre style="white-space: pre-wrap; color: white;">{email}</pre>
                        </div>
                    """, unsafe_allow_html=True)

                    # ✅ Copy-to-Clipboard Button
                    st.code(email, language='markdown')

                else:
                    st.error("❌ No job details found. Please try another URL.")

        except Exception as e:
            st.error(f"❌ An Error Occurred: {e}")

if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio, clean_text)
