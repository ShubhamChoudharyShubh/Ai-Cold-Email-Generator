# 🚀 Cold Email Generator for Software Services Companies

## 📌 Overview
The **Cold Email Generator** is an AI-driven tool designed to **automate and personalize cold email outreach** for software service companies. By leveraging job postings from company career pages, this system crafts **highly relevant** and **engaging cold emails** using advanced **LLM (Large Language Model)**, **Chroma DB (Vector Database)**, and a **user-friendly Streamlit UI**. 

This project aims to **streamline lead generation** and **increase client engagement** for companies like **TCS, Infosys, and other IT service providers**.

## 🎯 Key Features
✅ **AI-Powered Email Generation** – Utilizes Llama-based LLM for dynamic, personalized email writing.  
✅ **Automated Job Scraping** – Extracts job postings from company career pages using **Selenium Web Scraper**.  
✅ **Chroma DB for Smart Storage** – Efficiently stores and retrieves job descriptions for better AI response.  
✅ **Intuitive Streamlit UI** – Simplifies user interaction with easy-to-use features.  
✅ **Frontend Landing Page** – Built with **HTML, CSS, Bootstrap, and Tailwind CSS** to highlight features and use cases.  
✅ **Cloud-Based AI Inference** – Powered by **GROQ API** for seamless and scalable AI processing.  
✅ **Scalability & Efficiency** – Designed for adaptability with potential expansion into **multiple industries and automation levels**.  

## 🏗️ System Architecture
![Architecture](imgs/architecture.png)

## 🛠️ Technology Stack
| Component   | Technology Used |
|------------|----------------|
| **Frontend** | HTML, CSS, Bootstrap, Tailwind CSS |
| **Backend** | Python, Streamlit, LangChain |
| **Database** | Chroma DB, MySQL |
| **Web Scraping** | Selenium |
| **AI Model** | GROQ API (DeepSeek-R1-Distill-Llama-70B) |

## 🔧 Installation & Setup Guide

### 1️⃣ Configure API Key
- **Sign up** for an API key from [GROQ Console](https://console.groq.com/keys).
- Inside `app/.env`, update `GROQ_API_KEY` with your **generated API key**.

### 2️⃣ Install Dependencies
Run the following command to install all required Python dependencies:
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit Application
```bash
streamlit run app/main.py
```

## 🚀 Future Enhancements
🔹 **Multi-Industry Support** – Expand to support cold email generation across different sectors.  
🔹 **Email Tracking & Analytics** – Provide insights into open rates, replies, and engagement.  
🔹 **Automated Email Scheduling** – Allow users to schedule and send bulk cold emails at optimal times.  
🔹 **Integrations with CRM Tools** – Seamlessly connect with HubSpot, Salesforce, or Zoho CRM.  

## 📜 License
This project is open-source and available under the **[MIT License](LICENSE)**.

---

### 🔗 Connect with Me
**Shubham Choudhary**  
🌐 GitHub: [ShubhamChoudharyShubh](https://github.com/ShubhamChoudharyShubh)  
💼 LinkedIn: [Shubham Choudhary](https://www.linkedin.com/in/shubham-choudhary-shubh)  
✉️ Email: shubhamchoudharyshubh@gmail.com
