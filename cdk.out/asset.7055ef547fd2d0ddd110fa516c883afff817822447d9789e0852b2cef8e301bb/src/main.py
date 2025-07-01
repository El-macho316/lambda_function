import json
import datetime
import boto3
import yfinance as yf
from decimal import Decimal
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Toggle this if you don't want caching
USE_DYNAMODB_CACHE = True
DYNAMO_TABLE_NAME = "FundamentalAnalysisData"

def lambda_handler(event, context):
    agent = event['agent']
    parameters = event.get('parameters', [])
    function = event['function']
    actionGroup = event['actionGroup']

    def get_time():
        return (datetime.datetime.now() + datetime.timedelta(hours=7)).strftime('%Y-%m-%d %H:%M:%S')



    def fetch_financial_data(ticker):
        """Fetch financial metrics using yfinance with enhanced error handling"""
        try:
            logger.info(f"Fetching data for ticker: {ticker}")
            
            # Create ticker object
            stock = yf.Ticker(ticker)
            
            # Try to get basic info first
            try:
                info = stock.info
                logger.info(f"Successfully retrieved info for {ticker}")
            except Exception as e:
                logger.error(f"Failed to get info for {ticker}: {e}")
                return None
            
            # Check if we got valid data
            if not info or len(info) == 0:
                logger.error(f"Empty info object for {ticker}")
                return None
                
            # Check for basic required fields
            company_name = info.get("longName") or info.get("shortName")
            if not company_name:
                logger.error(f"No company name found for {ticker}")
                return None
                
            logger.info(f"Found company: {company_name}")

            # Extract financial data with fallbacks
            financial_data = {
                "companyName": company_name,
                "ticker": ticker.upper(),
                "peRatio": info.get("trailingPE") or info.get("forwardPE"),
                "roe": (info.get("returnOnEquity") * 100) if info.get("returnOnEquity") else None,
                "evToEbitda": info.get("enterpriseToEbitda") or info.get("enterpriseToEbitda"),
                "eps": info.get("trailingEps") or info.get("forwardEps"),
                "deRatio": info.get("debtToEquity"),
                "marketCap": info.get("marketCap"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown")
            }
            
            return financial_data
            
        except Exception as e:
            logger.error(f"Exception in fetch_financial_data for {ticker}: {str(e)}")
            return None

    def compute_score_and_valuation(data):
        """Compute the score and valuation label"""
        weights = {
            "peRatio": 0.25,
            "roe": 0.25,
            "evToEbitda": 0.25,
            "eps": 0.25
        }

        score = 0
        valid_metrics = 0
        score_breakdown = {}

        # PE Ratio scoring
        if data["peRatio"] and data["peRatio"] > 0:
            if data["peRatio"] < 15:
                pe_score = 100
            elif data["peRatio"] < 20:
                pe_score = 80
            else:
                pe_score = 50
            score += pe_score * weights["peRatio"]
            score_breakdown["peRatio"] = pe_score
            valid_metrics += 1

        # ROE scoring
        if data["roe"] and data["roe"] > 0:
            if data["roe"] > 20:
                roe_score = 100
            elif data["roe"] > 10:
                roe_score = 80
            else:
                roe_score = 50
            score += roe_score * weights["roe"]
            score_breakdown["roe"] = roe_score
            valid_metrics += 1

        # EV/EBITDA scoring
        if data["evToEbitda"] and data["evToEbitda"] > 0:
            if data["evToEbitda"] < 10:
                ev_score = 100
            elif data["evToEbitda"] < 15:
                ev_score = 70
            else:
                ev_score = 40
            score += ev_score * weights["evToEbitda"]
            score_breakdown["evToEbitda"] = ev_score
            valid_metrics += 1

        # EPS scoring
        if data["eps"] and data["eps"] > 0:
            if data["eps"] > 5:
                eps_score = 100
            elif data["eps"] > 1:
                eps_score = 70
            else:
                eps_score = 40
            score += eps_score * weights["eps"]
            score_breakdown["eps"] = eps_score
            valid_metrics += 1

        # Ensure we have enough valid metrics
        if valid_metrics == 0:
            return 0, "Unable to evaluate", "Insufficient financial data available for analysis", score_breakdown

        final_score = round(score, 2)
        
        if final_score >= 80:
            valuation = "Undervalued"
        elif final_score >= 60:
            valuation = "Fairly valued"
        else:
            valuation = "Overvalued"

        rationale = f"{data['companyName']} has a score of {final_score} based on {valid_metrics} metrics. "
        rationale += f"P/E: {data['peRatio']:.2f}" if data['peRatio'] else "P/E: N/A"
        rationale += f", ROE: {data['roe']:.2f}%" if data['roe'] else ", ROE: N/A"
        rationale += f", EV/EBITDA: {data['evToEbitda']:.2f}" if data['evToEbitda'] else ", EV/EBITDA: N/A"
        rationale += f", EPS: {data['eps']:.2f}" if data['eps'] else ", EPS: N/A"
        rationale += f". The stock is considered {valuation.lower()}."

        return final_score, valuation, rationale, score_breakdown

    def get_ticker_analysis(ticker):
        """Get comprehensive ticker analysis"""
        try:
            # Fetch financial data
            data = fetch_financial_data(ticker)
            
            if not data:
                return f"Unable to retrieve financial data for ticker '{ticker}'. Please verify the ticker symbol is correct."
            
            if not data.get("companyName"):
                return f"Ticker '{ticker}' does not appear to be a valid stock symbol."
            
            # Compute score and valuation
            score, valuation, rationale, score_breakdown = compute_score_and_valuation(data)
            
            # Format the response
            analysis = f"""
Stock Analysis for {data['companyName']} ({ticker.upper()}):

Financial Metrics:
- P/E Ratio: {data['peRatio']:.2f if data['peRatio'] else 'N/A'}
- ROE: {data['roe']:.2f}% if data['roe'] else 'N/A'
- EV/EBITDA: {data['evToEbitda']:.2f if data['evToEbitda'] else 'N/A'}
- EPS: ${data['eps']:.2f if data['eps'] else 'N/A'}
- Debt-to-Equity: {data['deRatio']:.2f if data['deRatio'] else 'N/A'}
- Market Cap: ${data['marketCap']:,} if data['marketCap'] else 'N/A'
- Sector: {data['sector']}
- Industry: {data['industry']}

Investment Analysis:
- Overall Score: {score}/100
- Valuation: {valuation}
- Analysis: {rationale}
            """.strip()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in get_ticker_analysis: {e}")
            return f"Error analyzing ticker '{ticker}': {str(e)}"

    # Parse parameters
    param_dict = {param['name'].lower(): param['value'] for param in parameters}

    # Route to appropriate function
    if function == "get_time":
        result = get_time()
        result_text = f"The current time is {result}"

    elif function == "get_ticker_analysis":
        ticker = param_dict.get('ticker')
        if ticker:
            result_text = get_ticker_analysis(ticker)
        else:
            result_text = "Error: Please provide a ticker symbol for analysis."

    else:
        result_text = "Error: Unknown function"

    # Prepare response
    responseBody = {
        "TEXT": {
            "body": result_text
        }
    }

    action_response = {
        'actionGroup': actionGroup,
        'function': function,
        'functionResponse': {
            'responseBody': responseBody
        }
    }

    dummy_function_response = {
        'response': action_response,
        'messageVersion': event['messageVersion']
    }

    print("Response : {}".format(dummy_function_response))
    return dummy_function_response