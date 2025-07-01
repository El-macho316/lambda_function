import json
import boto3
import yfinance as yf
from datetime import datetime
from decimal import Decimal
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Toggle this if you don't want caching
USE_DYNAMODB_CACHE = True
DYNAMO_TABLE_NAME = "FundamentalAnalysisData"

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
        
        # Log what we found
        logger.info(f"Financial data extracted: {financial_data}")
        
        return financial_data
        
    except Exception as e:
        logger.error(f"Exception in fetch_financial_data for {ticker}: {str(e)}")
        return None

def test_yfinance_connection():
    """Test if yfinance is working and can connect to Yahoo Finance"""
    try:
        # Try a simple test with a well-known ticker
        test_stock = yf.Ticker("AAPL")
        test_info = test_stock.info
        
        if test_info and test_info.get("longName"):
            logger.info("yfinance connection test successful")
            return True
        else:
            logger.error("yfinance connection test failed - no data returned")
            return False
            
    except Exception as e:
        logger.error(f"yfinance connection test failed with exception: {e}")
        return False

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

def cache_to_dynamo(ticker, output):
    """Cache results to DynamoDB"""
    try:
        dynamo = boto3.resource("dynamodb", region_name="ap-southeast-1")
        table = dynamo.Table(DYNAMO_TABLE_NAME)

        # Convert floats to Decimals for DynamoDB
        def convert_float_to_decimal(obj):
            if isinstance(obj, float):
                return Decimal(str(obj))
            elif isinstance(obj, dict):
                return {k: convert_float_to_decimal(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_float_to_decimal(v) for v in obj]
            return obj

        item = {
            "ticker": ticker.upper(),
            "company_name": output["companyName"],
            "sector": output.get("sector", "Unknown"),
            "industry": output.get("industry", "Unknown"),
            "metrics": convert_float_to_decimal({
                k: v for k, v in output.items() 
                if k not in ["companyName", "ticker", "score", "valuation", "rationale", "timestamp", "scoreBreakdown"]
            }),
            "score": Decimal(str(output["score"])),
            "valuation": output["valuation"],
            "score_breakdown": convert_float_to_decimal(output.get("scoreBreakdown", {})),
            "last_updated": datetime.utcnow().isoformat()
        }

        table.put_item(Item=item)
        logger.info(f"Successfully cached data for {ticker} to DynamoDB")
        
    except Exception as e:
        logger.error(f"Failed to cache to DynamoDB: {e}")

def lambda_handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")

    try:
        agent = event['agent']
        parameters = event.get('parameters', [])
        function = event['function']
        actionGroup = event['actionGroup']
        messageVersion = event.get('messageVersion', "1.0")

        param_dict = {param['name'].lower(): param['value'] for param in parameters if param.get('value')}

        if function == "get_fundamentals":
            ticker = param_dict.get("ticker")

            if not ticker:
                response_text = "Error: Please provide a ticker symbol to analyze."
            else:
                # Test yfinance
                if not test_yfinance_connection():
                    response_text = "Error: Could not connect to financial data service."

                else:
                    data = fetch_financial_data(ticker)
                    if not data:
                        response_text = f"Error: No data found for ticker '{ticker}'."
                    else:
                        score, valuation, rationale, score_breakdown = compute_score_and_valuation(data)
                        response = {
                            **data,
                            "score": score,
                            "valuation": valuation,
                            "rationale": rationale,
                            "scoreBreakdown": score_breakdown,
                            "timestamp": datetime.utcnow().isoformat()
                        }

                        if USE_DYNAMODB_CACHE:
                            cache_to_dynamo(ticker, response)

                        response_text = rationale  # you can also json.dumps(response, indent=2)

            responseBody = {
                "TEXT": {
                    "body": response_text
                }
            }

        else:
            responseBody = {
                "TEXT": {
                    "body": f"Error: Unknown function '{function}'."
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
            'messageVersion': messageVersion
        }

        logger.info(f"Returning response: {json.dumps(dummy_function_response)}")
        return dummy_function_response

    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": "InternalError",
                "message": "Something went wrong."
            })
        }

if __name__ == "__main__":
    test_event = {"ticker": "AAPL"}
    result = lambda_handler(test_event, None)
    # Pretty print the result
    if isinstance(result, dict) and "body" in result:
        try:
            body = json.loads(result["body"])
            print(json.dumps(body, indent=2))
        except Exception:
            print(result)
    else:
        print(result)