# 🚀 DataPilot AI

**DataPilot AI** is an AI-powered Slack assistant designed for Data Engineers, SQL developers, students, and analysts.

Instead of switching between documentation, Stack Overflow, and AI chatbots, DataPilot lets you work directly inside Slack to understand, clean, and generate SQL using natural language.

---

## ✨ Features

### 🧠 SQL Explanation

Understand complex SQL queries in plain English.

Example

Input

@DataPilot explain

SELECT * FROM employees WHERE salary > 50000;

Output

• Explains what the query does
• Describes filters
• Explains returned results
• Easy for beginners to understand

---

### 🧹 SQL Cleaner

Automatically formats messy SQL into clean, readable SQL.

Example

Input

@DataPilot clean

select id,name from employee where salary>10000;

Output

```sql
SELECT
    id,
    name
FROM employee
WHERE salary > 10000;
```

---

### ⚡ SQL Generator

Generate SQL from natural language.

Example

Input

@DataPilot generate

Show the top 5 highest paid employees.

Output

```sql
SELECT
    employee_name,
    salary
FROM employees
ORDER BY salary DESC
LIMIT 5;
```

---

## 🛠️ Tech Stack

- Python 3.12
- Slack Bolt
- Google Gemini API
- Google AI Studio
- python-dotenv
- Modular AI Provider Architecture
- Object-Oriented Programming
- Dependency Injection

---

## 🏗️ Project Architecture

```
Slack Workspace
        │
        ▼
     app.py
        │
        ▼
 Command Dispatcher
        │
 ┌──────┼─────────┐
 ▼      ▼         ▼
Explain Clean   Generate
 │        │         │
 ▼        ▼         ▼
Analyzer Cleaner Generator
        │
        ▼
 AI Provider Layer
        │
        ▼
 Gemini Provider
        │
        ▼
 Gemini API
```

---

## 📁 Project Structure

```
DataPilot/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
│
├── services/
│   ├── analyzer.py
│   ├── cleaner.py
│   ├── generator.py
│   └── ai/
│       ├── base.py
│       ├── gemini_provider.py
│       ├── provider_factory.py
│       └── __init__.py
│
└── prompts/
```

---

## 🚀 Getting Started

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DataPilot.git
```

Move into the project

```bash
cd DataPilot
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
SLACK_BOT_TOKEN=your_bot_token
SLACK_APP_TOKEN=your_app_token

AI_PROVIDER=gemini

GOOGLE_API_KEY=your_google_api_key

GEMINI_MODEL=gemini-2.5-flash
```

Run the bot

```bash
python app.py
```

---

## 🎯 Current Commands

| Command | Description |
|----------|-------------|
| `@DataPilot explain` | Explain SQL queries |
| `@DataPilot clean` | Format SQL queries |
| `@DataPilot generate` | Generate SQL from natural language |

---

## 🚧 Upcoming Features

- SQL Query Optimizer
- SQL Insights & Analysis
- Execution Plan Suggestions
- CSV Dataset Analysis
- Database Schema Awareness
- Interactive SQL Assistant

---

## 💡 Why DataPilot?

Data engineers and SQL developers spend significant time:

- Understanding legacy SQL
- Cleaning poorly formatted queries
- Writing repetitive SQL
- Switching between documentation and AI tools

DataPilot brings these capabilities directly into Slack, allowing teams to collaborate without leaving their workspace.

---

## 👨‍💻 Author

Developed by **Mamba**

Built for AI and Data Engineering Hackathons 🚀