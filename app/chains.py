import os
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="deepseek-r1-distill-llama-70b")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills`, and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Shubham, a freelance web developer specializing in PHP, WordPress, and custom website development. 
            With over three years of experience, you have successfully delivered 25+ projects, including e-commerce platforms, 
            business websites, and discussion forums.

            Your expertise includes:
            - WordPress development using Elementor, WooCommerce, and custom themes.
            - Full-stack PHP development with MySQL, Laravel, and CodeIgniter.
            - SEO optimization for high-ranking, fast-loading websites.
            - Frontend development using JavaScript, Bootstrap, and Tailwind CSS.
            - Custom API development and integration.
            
            Your job is to write a cold email to the client regarding their website project, showcasing your experience 
            in fulfilling their requirements. Use the most relevant ones from the following links to showcase your portfolio: {link_list}
            
            **IMPORTANT RULES:**
            - Do NOT include "<think>" or any internal thoughts.
            - Do NOT use markdown formatting (no asterisks `**` for bold text).
            - Do NOT include "mailto:" in the email address.
            - Use the following signature format at the end:
            
            Best regards,  
            Shubham Choudhary  
            shubhamchoudharyshubh@gmail.com  
            LinkedIn: https://www.linkedin.com/in/shubham-choudhary-shubh/  
            WhatsApp: https://wa.me/919302418061  
            Portfolio: https://shubhamchoudharyshubh.in  

            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        
        # Post-process email to remove unwanted formatting
        email_text = res.content

        # Remove "<think> ... </think>" sections
        email_text = re.sub(r'<think>.*?</think>', '', email_text, flags=re.DOTALL).strip()

        # Remove markdown-style bold (**text**)
        email_text = re.sub(r'\*\*(.*?)\*\*', r'\1', email_text)

        return email_text

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))
