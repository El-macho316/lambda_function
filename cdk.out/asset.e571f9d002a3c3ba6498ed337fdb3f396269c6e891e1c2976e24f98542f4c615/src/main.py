import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
import time
import logging
import yfinance as yf
from decimal import Decimal

# Assuming 'logger.py' is available as in the original script.
# If not, a basic logger can be configured here.
try:
    from logger import get_logger
except ImportError:
    # Fallback basic logger if logger.py is not available
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

class FinancialDataService:
    """
    Service for retrieving, processing, and scoring financial data for a given stock ticker.
    It fetches data, computes a valuation score, and provides a rationale.
    """
    
    def __init__(self, use_caching=True):
        """
        Initializes the service.
        
        Args:
            use_caching (bool): Flag to enable/disable DynamoDB caching.
        """
        self.logger = get_logger("FinancialDataService")
        self.use_caching = use_caching
        self.dynamo_table_name = "FundamentalAnalysisData"

    def get_financial_data(self, ticker: str) -> Dict[str, Any]:
        """
        Main method to retrieve, analyze, and score financial data for a ticker.

        Args:
            ticker: The stock ticker symbol.

        Returns:
            A dictionary containing the analysis, score, and metadata.
        """
        try:
            ticker = ticker.upper()
            self.logger.info(f"Processing request for ticker: {ticker}")
            
            # 1. Validate inputs
            if not ticker:
                return self._error_response("Ticker symbol is required")

            # 2. Fetch data from yfinance
            raw_data = self._fetch_from_yfinance(ticker)
            if raw_data is None:
                return self._error_response(f"Could not retrieve valid financial data for ticker '{ticker}'. Please verify the symbol.")

            # 3. Compute score and valuation
            score, valuation, rationale, score_breakdown = self._compute_score_and_valuation(raw_data)

            # 4. Prepare successful response
            result = {
                'success': True,
                'ticker': ticker,
                'timestamp': datetime.now().isoformat(),
                'data': {
                    **raw_data,
                    "score": score,
                    "valuation": valuation,
                    "rationale": rationale,
                    "scoreBreakdown": score_breakdown,
                }
            }
            
            # 5. Cache the result to DynamoDB if enabled
            if self.use_caching and result['success']:
                self._cache_to_dynamo(ticker, result['data'])

            self.logger.info(f"Successfully processed data for {ticker}")
            return result
            
        except Exception as e:
            self.logger.error(f"Financial data retrieval and analysis failed for {ticker}", error=e)
            return self._error_response(f"An internal error occurred during data retrieval: {str(e)}")

    def _fetch_from_yfinance(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetches financial metrics using yfinance with enhanced error handling.
        """
        self.logger.info(f"Fetching data from yfinance for: {ticker}")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or not info.get("longName"):
                self.logger.warning(f"No valid data returned for ticker: {ticker}")
                return None
                
            financial_data = {
                "companyName": info.get("longName") or info.get("shortName"),
                "peRatio": info.get("trailingPE") or info.get("forwardPE"),
                "roe": (info.get("returnOnEquity") * 100) if info.get("returnOnEquity") else None,
                "evToEbitda": info.get("enterpriseToEbitda"),
                "eps": info.get("trailingEps") or info.get("forwardEps"),
                "deRatio": info.get("debtToEquity"),
                "marketCap": info.get("marketCap"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "retrieved_at": datetime.now().isoformat()
            }
            self.logger.info(f"Successfully extracted data for {ticker}")
            return financial_data
        except Exception as e:
            self.logger.error(f"Exception in _fetch_from_yfinance for {ticker}: {e}")
            return None

    def _compute_score_and_valuation(self, data: Dict[str, Any]) -> tuple:
        """Computes a valuation score and generates a rationale based on key metrics."""
        weights = {"peRatio": 0.25, "roe": 0.25, "evToEbitda": 0.25, "eps": 0.25}
        score, valid_metrics = 0, 0
        score_breakdown = {}

        # Scoring logic for each metric
        if data.get("peRatio") and data["peRatio"] > 0:
            pe_score = 100 if data["peRatio"] < 15 else (80 if data["peRatio"] < 20 else 50)
            score += pe_score * weights["peRatio"]
            score_breakdown["peRatio"] = pe_score
            valid_metrics += 1
        
        if data.get("roe") and data["roe"] > 0:
            roe_score = 100 if data["roe"] > 20 else (80 if data["roe"] > 10 else 50)
            score += roe_score * weights["roe"]
            score_breakdown["roe"] = roe_score
            valid_metrics += 1

        if data.get("evToEbitda") and data["evToEbitda"] > 0:
            ev_score = 100 if data["evToEbitda"] < 10 else (70 if data["evToEbitda"] < 15 else 40)
            score += ev_score * weights["evToEbitda"]
            score_breakdown["evToEbitda"] = ev_score
            valid_metrics += 1
            
        if data.get("eps") and data["eps"] > 0:
            eps_score = 100 if data["eps"] > 5 else (70 if data["eps"] > 1 else 40)
            score += eps_score * weights["eps"]
            score_breakdown["eps"] = eps_score
            valid_metrics += 1

        if valid_metrics == 0:
            return 0, "Unable to evaluate", "Insufficient financial data for analysis.", {}

        final_score = round(score, 2)
        valuation = "Undervalued" if final_score >= 80 else ("Fairly valued" if final_score >= 60 else "Overvalued")
        
        rationale = (
            f"{data['companyName']} has a score of {final_score} based on {valid_metrics} metrics. "
            f"P/E: {data.get('peRatio', 'N/A'):.2f}, ROE: {data.get('roe', 'N/A'):.2f}%, "
            f"EV/EBITDA: {data.get('evToEbitda', 'N/A'):.2f}, EPS: {data.get('eps', 'N/A'):.2f}. "
            f"The stock is considered {valuation.lower()}."
        )
        return final_score, valuation, rationale, score_breakdown

    def _cache_to_dynamo(self, ticker: str, data: Dict[str, Any]):
        """Caches the analysis result to DynamoDB."""
        try:
            # Conditional import of boto3
            import boto3
            dynamo = boto3.resource("dynamodb")
            table = dynamo.Table(self.dynamo_table_name)
            
            # Helper to convert floats to Decimals for DynamoDB
            def convert_floats_to_decimals(obj):
                if isinstance(obj, float):
                    return Decimal(str(obj))
                if isinstance(obj, dict):
                    return {k: convert_floats_to_decimals(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_floats_to_decimals(i) for i in obj]
                return obj

            item_to_cache = {
                "ticker": ticker.upper(),
                "last_updated": datetime.utcnow().isoformat(),
                **convert_floats_to_decimals(data)
            }
            
            table.put_item(Item=item_to_cache)
            self.logger.info(f"Successfully cached data for {ticker} to DynamoDB.")
        except ImportError:
            self.logger.warning("boto3 not found. Skipping DynamoDB caching.")
        except Exception as e:
            self.logger.error(f"Failed to cache to DynamoDB for {ticker}: {e}")

    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Generates a standardized error response dictionary."""
        return {
            'success': False,
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        }

def lambda_handler(event, context):
    """
    AWS Lambda handler for financial data requests, optimized for Amazon Bedrock Agents.
    This handler now only requires a 'ticker' and performs a full financial analysis.
    """
    logger = get_logger("FinancialDataHandler")
    logger.info(f"Lambda received event: {json.dumps(event)}")

    # Bedrock Agent metadata
    actionGroup = event.get('actionGroup', 'FinancialAnalysisActionGroup')
    function = event.get('function', 'getFinancialAnalysis') # Function name updated for clarity
    messageVersion = event.get('messageVersion', '1.0')

    try:
        ticker = ''
        # Extract 'ticker' from Bedrock Agent's 'parameters' list
        if 'parameters' in event and isinstance(event.get('parameters'), list):
            for param in event['parameters']:
                if param.get('name') == 'ticker' and param.get('value'):
                    ticker = str(param['value']).upper()
                    break
        
        # Fallback for direct Lambda tests
        if not ticker and event.get('ticker'):
            ticker = str(event['ticker']).upper()

        logger.info(f"Extracted Ticker: '{ticker}' for processing.")

        # If ticker is still missing, return a structured error
        if not ticker:
            service = FinancialDataService()
            error_details = service._error_response("Missing required parameter: ticker")
            response_body = {"TEXT": {"body": json.dumps(error_details)}}
            return {
                'messageVersion': messageVersion,
                'response': {
                    'actionGroup': actionGroup,
                    'function': function,
                    'functionResponse': {'responseBody': response_body}
                }
            }
        
        # Initialize service and process the request
        service = FinancialDataService()
        result = service.get_financial_data(ticker)
        
        # Prepare the final response for Bedrock Agent
        response_body = {"TEXT": {"body": json.dumps(result, default=str)}}
        
        final_response = {
            'messageVersion': messageVersion,
            'response': {
                'actionGroup': actionGroup,
                'function': function,
                'functionResponse': {'responseBody': response_body}
            }
        }
        
        logger.info(f"Request for {ticker} successful. Returning response.")
        logger.debug(f"Final Lambda response: {json.dumps(final_response)}")
        return final_response
        
    except Exception as e:
        logger.error(f"Lambda handler failed unexpectedly.", error=e)
        service = FinancialDataService()
        error_details = service._error_response(f"Internal server error: {str(e)}")
        response_body = {"TEXT": {"body": json.dumps(error_details)}}
        return {
            'messageVersion': messageVersion,
            'response': {
                'actionGroup': actionGroup,
                'function': function,
                'functionResponse': {'responseBody': response_body}
            }
        }

# For local testing
if __name__ == "__main__":
    # Mock yfinance for local testing to avoid actual network calls
    class MockYFinanceTicker:
        def __init__(self, ticker):
            self._ticker = ticker.upper()
            self.info = self._get_mock_data()

        def _get_mock_data(self):
            mock_db = {
                "AAPL": {
                    'longName': 'Apple Inc.', 'trailingPE': 28.5, 'returnOnEquity': 1.6, 
                    'enterpriseToEbitda': 22.0, 'trailingEps': 6.1, 'debtToEquity': 150.0,
                    'marketCap': 2800000000000, 'sector': 'Technology', 'industry': 'Consumer Electronics'
                },
                "MSFT": {
                    'longName': 'Microsoft Corp', 'trailingPE': 35.0, 'returnOnEquity': 0.40, 
                    'enterpriseToEbitda': 25.0, 'trailingEps': 11.5, 'debtToEquity': 45.0,
                    'marketCap': 3100000000000, 'sector': 'Technology', 'industry': 'Software - Infrastructure'
                },
                "INVALID": {}
            }
            return mock_db.get(self._ticker, {})

    # Monkey-patch yfinance.Ticker to use our mock class
    yf.Ticker = MockYFinanceTicker

    print("="*60)
    print("Financial Data Service - Local Demonstration")
    print("="*60)
    
    test_events = [
        # Test Case 1: Bedrock Agent style event for a valid ticker (AAPL)
        {
            "messageVersion": "1.0",
            "actionGroup": "FinancialAnalysisActionGroup",
            "function": "getFinancialAnalysis",
            "parameters": [{"name": "ticker", "value": "AAPL"}],
            "sessionId": "test-session-1"
        },
        # Test Case 2: Direct Lambda console style event (MSFT)
        {"ticker": "MSFT"},
        # Test Case 3: Bedrock event with an invalid ticker
        {
            "messageVersion": "1.0",
            "actionGroup": "FinancialAnalysisActionGroup",
            "function": "getFinancialAnalysis",
            "parameters": [{"name": "ticker", "value": "INVALID"}],
            "sessionId": "test-session-2"
        },
        # Test Case 4: Bedrock event with missing ticker
        {
            "messageVersion": "1.0",
            "actionGroup": "FinancialAnalysisActionGroup",
            "function": "getFinancialAnalysis",
            "parameters": [], # No ticker parameter
            "sessionId": "test-session-3"
        }
    ]
    
    for i, event in enumerate(test_events, 1):
        print(f"\n--- Test Case {i} ---\nInput Event: {json.dumps(event)}")
        result = lambda_handler(event, None)
        
        # Pretty print the final JSON body for readability
        print("\nOutput Response Body:")
        try:
            body_str = result['response']['functionResponse']['responseBody']['TEXT']['body']
            parsed_body = json.loads(body_str)
            print(json.dumps(parsed_body, indent=2))
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Could not parse response body. Raw response: {json.dumps(result)}")

    print("\n" + "=" * 60)
    print("Local Demonstration Complete!")
