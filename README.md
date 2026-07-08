# 🚀 DataPilot AI

DataPilot AI is an intelligent Slack assistant that helps developers understand, improve, and generate SQL using Google's Gemini AI.

Instead of switching between documentation, ChatGPT, Stack Overflow, and SQL editors, users can simply mention the bot inside Slack and receive AI-powered assistance instantly.

---

# ✨ Features

## 🧠 SQL Explain

Explain complex SQL queries in simple English.

Example

```sql
SELECT name
FROM employees
WHERE salary > 50000;
```

Output

- Explains what the query does
- Describes filters
- Explains returned data
- Beginner friendly

---

## 🧹 SQL Clean

Automatically formats SQL using professional conventions.

Converts

```sql
select * from users where age>20
```

into

```sql
SELECT *
FROM users
WHERE age > 20;
```

---

## ⚡ SQL Generate

Generate SQL from natural language.

Example

```
generate show top 10 customers by revenue
```

Produces a complete ANSI SQL query.

---

## 🚀 SQL Optimize

Analyzes SQL performance and suggests improvements.

Example

```sql
SELECT *
FROM orders
WHERE YEAR(order_date)=2024;
```

Returns

- Performance Rating
- Bottlenecks
- Improved Query
- Expected Benefits
- Final Recommendation

---

# 🏗 Architecture

```
app.py
│
├── services
│   ├── analyzer.py
│   ├── cleaner.py
│   ├── generator.py
│   ├── optimizer.py
│   │
│   └── ai
│       ├── base.py
│       ├── gemini_provider.py
│       ├── provider_factory.py
│       └── __init__.py
│
├── prompts
├── database
├── bot
└── assets
```

The project follows:

- Dependency Injection
- Provider Pattern
- Factory Pattern
- SOLID Principles
- Modular Architecture

Changing AI providers only requires adding a provider inside:

```
services/ai/
```

without modifying business logic.

---

# 🛠 Tech Stack

- Python
- Slack Bolt SDK
- Google Gemini API
- Google AI Studio
- python-dotenv

---

# Installation

Clone the repository

```bash
git clone https://github.com/omkarmusle510-web/DataPilot.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env`

```env
GOOGLE_API_KEY=your_api_key

SLACK_BOT_TOKEN=xoxb-...

SLACK_APP_TOKEN=xapp-...

AI_PROVIDER=gemini

GEMINI_MODEL=gemini-2.5-flash

LOG_LEVEL=INFO
```

Run

```bash
python app.py
```

---

# Slack Commands

### Explain SQL

```
@DataPilot explain SELECT * FROM employees;
```

---

### Clean SQL

```
@DataPilot clean select * from employees where salary>50000
```

---

### Generate SQL

```
@DataPilot generate show all customers who purchased in January
```

---

### Optimize SQL

<<<<<<< HEAD
Developed by **OMKAR MUSALE**
=======
```
@DataPilot optimize SELECT * FROM orders WHERE YEAR(order_date)=2024;
```
>>>>>>> 9c3ca32 (Add complete SQL AI toolkit with explain, clean, generate, optimize and validate features)

---

# Example

### Input

```
@DataPilot optimize
SELECT * FROM orders
WHERE YEAR(order_date)=2024;
```

### Output

✅ Performance Rating

✅ Optimization Opportunities

✅ Improved SQL

✅ Expected Benefits

✅ Final Recommendation

---

# Future Roadmap

- SQL Validator
- Query Cost Estimation
- Execution Plan Analysis
- Database Schema Understanding
- Multi-provider AI Support
- CSV Dataset Cleaning
- Data Profiling
- Data Quality Reports

---

# Author

**Omkar Musale**

Data Science Student

Built with ❤️ using Python, Slack Bolt, and Google Gemini.