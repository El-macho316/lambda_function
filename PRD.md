# 📄 Product Requirements Document (PRD): Fundamental Analysis on Request

## 1. Project Goals and Vision

### 🎯 Vision
Build an intelligent, conversational tool that enables investment consultants and clients to obtain **on-demand fundamental analysis** of publicly traded companies. Users should be able to inquire about a stock's valuation, financial health, and key metrics — all via a simple chat interface — to make faster and smarter investment decisions.

### ✅ Goals
- Deliver real-time and accurate fundamental insights using up-to-date financial data.
- Support natural language queries like "Is Tesla undervalued?" or "Give me a fundamental analysis of Apple."
- Design the module to be **modular and easily integrated** into a broader chatbot ecosystem.
- Introduce a **universal scoring system (0–100)** for comparing companies.

## 2. Core Features and Functionality

### 💬 Natural Language Understanding (NLU)
- Extract company names or tickers and intent from user messages.

### 🏢 Company Lookup
- Resolve company names to stock tickers using a reference mapping or financial API.

### 📈 Financial Data Retrieval
- Retrieve real-time financial metrics (P/E, EPS, ROE, Debt/Equity, EV/EBITDA, etc.) using **Yahoo Finance** and/or **Financial Modeling Prep API**.

### 🧠 Valuation Analysis
- Apply rule-based heuristics to assess company value.
- Classify company as **undervalued**, **fairly valued**, or **overvalued**.

### 📊 Scoring Engine
- Compute a **fundamental health score (0–100)** based on:
  - Profitability (e.g., ROE, Net Margin)
  - Valuation (e.g., P/E, EV/EBITDA)
  - Leverage (e.g., Debt/Equity)
  - Growth (e.g., EPS growth)
- Normalize and weight each metric for **industry-agnostic** comparability.

### 🧾 Response Generation
- Return well-formatted summaries with scoring, classification, and rationale.

### ❌ Error Handling
- Handle unrecognized tickers, ambiguous inputs, and missing data gracefully.

## 3. AWS Architecture Overview

| Purpose                              | AWS Service                  |
|--------------------------------------|------------------------------|
| LLM orchestration (agent)            | Amazon Bedrock (Agents)      |
| Business logic & scoring engine      | AWS Lambda                   |
| Data storage / caching               | DynamoDB                     |
| External API integration             | (e.g., Yahoo, FMP APIs)      |

## 4. Technical Requirements and Constraints

### 🛠 Languages/Frameworks
- AWS Lambda (Node.js or Python)
- Bedrock Agents
- Optional: Scikit-learn or NumPy for internal scoring logic

### 🧱 Infrastructure
- Serverless and event-driven
- IAM-secured resources
- API Gateway (for optional external access)

### ⚠ Constraints
- Sub-3s response time for 95% of queries
- Cache data for 24h max
- Support both ticker and company name input

## 5. User Stories and Acceptance Criteria

### 📌 User Stories

#### 1. Request Full Analysis
As a user, I want a fundamental breakdown of a company so that I can assess its investment appeal.

#### 2. Metric-Specific Query
As a user, I want to ask about individual financial metrics so I can judge specific factors like valuation or profitability.

#### 3. Valuation Verdict
As a user, I want to know whether a company is undervalued so I can make better buy/sell decisions.

#### 4. Universal Scoring
As a user, I want to see a score (0–100) for a company's fundamentals so I can compare it easily to others.

#### 5. Input Error Recovery
As a user, if I input an invalid or partial company name, I want the chatbot to guide me toward a valid query.

### ✅ Acceptance Criteria

- Accepts ticker or company name
- Returns metrics (P/E, EPS, ROE, EV/EBITDA, Market Cap)
- Includes:
  - Valuation label (e.g., undervalued)
  - **Fundamental score (0–100)**
- Provides rationale in plain English
- Handles errors with clear suggestions

## 6. Lambda Tool: `FundamentalAnalysisTool`

### 🎯 Purpose
Retrieve key metrics, compute a universal score, and classify valuation status.

### 🧾 Input
```json
{
  "ticker": "AAPL"
}
```

### ⚙️ Processing Steps

1. **Fetch Metrics**
   - P/E ratio, EPS, ROE, EV/EBITDA, Debt/Equity, Market Cap
   - Sector average benchmarks (if available)

2. **Score Fundamentals (0–100)**
   - Weight metrics across 4 categories:
     - Profitability (25%)
     - Valuation (25%)
     - Leverage (25%)
     - Growth/Consistency (25%)
   - Normalize and score relative to ideal thresholds or sector norms

3. **Run Valuation Heuristics**
   - P/E < 15–20 → undervalued
   - EV/EBITDA < 10 → value
   - ROE < 10% → weak profitability

4. **Generate Output**

### 📤 Output Example
```json
{
  "companyName": "Apple Inc.",
  "ticker": "AAPL",
  "peRatio": 18.2,
  "roe": 28.0,
  "evToEbitda": 15.3,
  "score": 83,
  "valuation": "Fairly valued",
  "rationale": "Apple's P/E (~18) and EV/EBITDA (~15) are close to sector averages, and its high ROE (28%) supports a balanced valuation.",
  "timestamp": "2025-06-16T07:27:03.968520Z"
}
```

## 7. Error Handling Examples

### ❌ Invalid Ticker
```json
{
  "error": "TickerNotFound",
  "message": "Financial data for ticker 'XYZ' could not be retrieved."
}
```

### ⚠ Partial Data
```json
{
  "valuation": "Insufficient data",
  "rationale": "P/E ratio was not available. Analysis was based only on ROE and EV/EBITDA."
}
```

## 8. Sample Conversation Flows

**User:** "Is Tesla undervalued?"  
**Agent:**  
> "Tesla (TSLA) appears undervalued. Its P/E is 10 (below typical), and EV/EBITDA is 8. ROE is modest, but multiples support a value signal. Score: 78/100."

**User:** "Give me Apple's score."  
**Agent:**  
> "Apple (AAPL) has a fundamental score of 83/100. Its strong ROE (28%) and stable valuation metrics suggest balanced performance."

## 9. Unified DynamoDB Schema: `FundamentalAnalysisData`

| Field               | Type        | Description                                                                 |
|---------------------|-------------|-----------------------------------------------------------------------------|
| `ticker`            | String (PK) | Unique stock ticker                                                         |
| `company_name`      | String      | Full company name                                                           |
| `sector`            | String      | Sector classification (e.g., "Technology")                                 |
| `industry`          | String      | Industry classification (e.g., "Consumer Electronics")                      |
| `metrics`           | JSON        | Object of financial metrics (P/E, ROE, EPS, etc.)                           |
| `score`             | Number      | Final 0–100 fundamental score                                               |
| `valuation`         | String      | "Undervalued", "Fairly valued", etc.                                       |
| `thresholds`        | JSON        | Metric-specific thresholds (min, max, weight per metric)                   |
| `last_updated`      | Timestamp   | Last time this data was refreshed                                          |

## 10. Performance and Security

### ⚡ Performance
- Daily data cache update

### 🔒 Security
- IAM role restriction on Lambda/API access
- Sanitize input before prompting or embedding
- Rate limiting via API Gateway or usage plan
- Data encryption at rest (DynamoDB, CloudWatch Logs)