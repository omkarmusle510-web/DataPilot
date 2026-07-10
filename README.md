# 🚀 DataPilot AI

<p align="center">
  <img src="assets/logo.png" alt="DataPilot AI" width="180"/>
</p>

<p align="center">
An AI-powered Slack Assistant for SQL Intelligence and Dataset Intelligence.<br>
Analyze SQL, clean datasets, generate insights, and boost data engineering productivity directly inside Slack.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Slack](https://img.shields.io/badge/Slack-Bolt-4A154B?logo=slack)
![SQLite](https://img.shields.io/badge/Database-SQLite-blue)
![Gemini](https://img.shields.io/badge/Google-Gemini-orange)
![Groq](https://img.shields.io/badge/Groq-Supported-green)
![License](https://img.shields.io/badge/License-MIT-success)

</p>

---

# 📌 Overview

Modern data engineers constantly switch between SQL editors, documentation, AI tools, spreadsheets, and collaboration platforms.

**DataPilot AI** brings all of these capabilities into **Slack**, allowing developers to:

- Explain SQL
- Generate SQL
- Optimize SQL
- Validate SQL
- Clean SQL
- Profile datasets
- Automatically clean CSV files
- Generate AI-powered dataset reports
- Download cleaned datasets
- Track command history

without ever leaving Slack.

---

# ✨ Key Features

---

## 🧠 SQL Intelligence

### 🔍 SQL Explain

Explain complex SQL queries in simple English.

Example

```sql
SELECT name
FROM employees
WHERE salary > 50000;
```

Returns

- Query purpose
- Table relationships
- Filters explained
- Beginner-friendly explanation

---

### 🧹 SQL Clean

Automatically formats SQL using professional formatting standards.

Example

Before

```sql
select * from employees where salary>50000
```

After

```sql
SELECT *
FROM employees
WHERE salary > 50000;
```

---

### ⚡ SQL Generate

Generate SQL from natural language.

Example

```
@DataPilot generate Show top 10 customers by revenue
```

Returns

```sql
SELECT customer_name,
SUM(revenue)
FROM sales
GROUP BY customer_name
ORDER BY SUM(revenue) DESC
LIMIT 10;
```

---

### 🚀 SQL Optimize

Analyze SQL performance.

Returns

- Performance Score
- Bottlenecks
- Optimized Query
- Expected Improvements
- Best Practices

---

### ✅ SQL Validate

Detects

- Syntax problems
- Bad SQL practices
- Potential issues
- Query recommendations

---

# 📊 Dataset Intelligence

Simply upload a CSV file to Slack.

DataPilot automatically:

✅ Downloads the dataset

✅ Profiles the dataset

✅ Detects quality issues

✅ Generates AI insights

✅ Cleans the dataset

✅ Uploads the cleaned CSV back to Slack

---

## 📈 AI Dataset Report

Automatically generates:

- Dataset Overview
- Key Insights
- Data Quality Analysis
- Cleaning Recommendations
- Suggested Visualizations
- Suggested SQL Analysis

---

## 🧹 Automatic Dataset Cleaning

The cleaning engine automatically performs:

- Duplicate removal
- Empty row removal
- Empty column removal
- Column name standardization
- Missing value handling
- Numeric conversion
- Date conversion
- Whitespace trimming
- Constant column detection
- Outlier detection
- Data type normalization

---

## 📥 Clean Dataset Download

After cleaning,

DataPilot uploads a cleaned CSV directly into the Slack thread for download.

No manual work required.

---

# 🗂 Command History

Every command execution is automatically stored.

History includes

- User ID
- Command
- Input
- Output
- AI Provider
- AI Model
- Execution Time
- Success / Failure
- Timestamp

SQLite is used as the local history database.

---

# 🤖 Multiple AI Providers

DataPilot follows a Provider Pattern.

Currently supported:

- Google Gemini
- Groq

Switching providers only requires updating

```
AI_PROVIDER=gemini
```

or

```
AI_PROVIDER=groq
```

No business logic changes are required.

---

# 🏗 System Architecture

```
                        Slack Workspace
                              │
                              ▼
                     Slack Socket Mode
                              │
                              ▼
                           app.py
                              │
          ┌───────────────────┼────────────────────┐
          │                   │                    │
          ▼                   ▼                    ▼
     SQL Services      Dataset Services      History System
          │                   │                    │
          │                   │                    ▼
          │            File Handler          SQLite Database
          │                   │
          ▼                   ▼
   SQL Analyzer        Dataset Profiler
   SQL Cleaner         Dataset Cleaner
   SQL Generator
   SQL Optimizer
   SQL Validator
          │
          ▼
     AI Provider Factory
          │
     ┌────┴────┐
     ▼         ▼
  Gemini      Groq
```

---

# 🧩 Design Principles

The project follows modern software engineering principles.

- SOLID Principles
- Provider Pattern
- Factory Pattern
- Dependency Injection
- Modular Architecture
- Separation of Concerns
- Clean Code Practices

Every service is independent and easily extendable.

---

# 🛠 Tech Stack

## Backend

- Python

## AI

- Google Gemini
- Groq

## Slack

- Slack Bolt SDK
- Slack Socket Mode

## Data Processing

- Pandas

## Database

- SQLite

## Configuration

- python-dotenv

---

# 📂 Project Structure

```
DataPilot/
│
├── app.py
│
├── services/
│   ├── analyzer.py
│   ├── cleaner.py
│   ├── generator.py
│   ├── optimizer.py
│   ├── validator.py
│   ├── dataset_profiler.py
│   ├── dataset_cleaner.py
│   ├── file_handler.py
│   │
│   └── ai/
│       ├── base.py
│       ├── provider_factory.py
│       ├── gemini_provider.py
│       └── groq_provider.py
│
├── database/
│
├── prompts/
│
├── assets/
│
└── data/
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/omkarmusle510-web/DataPilot.git
```

Move into the project

```bash
cd DataPilot
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env`

```env
SLACK_BOT_TOKEN=xoxb-...

SLACK_APP_TOKEN=xapp-...

GOOGLE_API_KEY=your_google_api_key

GROQ_API_KEY=your_groq_api_key

AI_PROVIDER=groq

LOG_LEVEL=INFO
```

Run

```bash
python app.py
```

---

# 💬 Slack Commands

### Explain SQL

```
@DataPilot explain SELECT * FROM employees;
```

---

### Clean SQL

```
@DataPilot clean SELECT * FROM employees WHERE salary>50000;
```

---

### Generate SQL

```
@DataPilot generate Show total sales by month
```

---

### Optimize SQL

```
@DataPilot optimize SELECT * FROM orders WHERE YEAR(order_date)=2024;
```

---

### Validate SQL

```
@DataPilot validate SELECT * FROM employees;
```

---

### Dataset Intelligence

Simply upload a CSV file into Slack.

DataPilot automatically performs the entire workflow.

No command required.

---

# 📸 Demo

## SQL Assistant

*(Add screenshot here)*

```
assets/sql_demo.png
```

---

## Dataset Intelligence

*(Add screenshot here)*

```
assets/dataset_demo.png
```Future Enhancements

- Excel (.xlsx) support
- Interactive dashboards
- Automatic visualization generation
- OpenAI support
- Claude support
- OpenRouter integration
- Multi-file dataset comparison
- Cloud deployment
- Real-time collaboration analytics

---

# 👨‍💻 Author

**Omkar Musale**

Data Science Student

Mumbai University

---

# ⭐ Acknowledgements

Built using

- Python
- Slack Bolt SDK
- Google Gemini
- Groq
- Pandas
- SQLite

---
⭐ If you found this project interesting, consider giving it a star!
