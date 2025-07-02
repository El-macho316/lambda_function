import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import time
import logging
import yfinance as yf
from decimal import Decimal

# ========================================
# CONFIGURATION CONSTANTS
# ========================================

# Scoring thresholds for financial metrics
class ScoringThresholds:
    """Constants for financial metric scoring thresholds."""
    
    # P/E Ratio thresholds
    PE_EXCELLENT = 15
    PE_GOOD = 20
    
    # Return on Equity thresholds (%)
    ROE_EXCELLENT = 20
    ROE_GOOD = 10
    
    # EV/EBITDA thresholds
    EV_EBITDA_EXCELLENT = 10
    EV_EBITDA_GOOD = 15
    
    # Earnings Per Share thresholds
    EPS_EXCELLENT = 5
    EPS_GOOD = 1

# Scoring weights for different metrics
class ScoringWeights:
    """Weights for different financial metrics in overall score calculation."""
    PE_RATIO = 0.25
    ROE = 0.25
    EV_EBITDA = 0.25
    EPS = 0.25

# Score values
class ScoreValues:
    """Score values for different performance levels."""
    EXCELLENT = 100
    GOOD = 80
    FAIR = 70
    POOR = 50
    VERY_POOR = 40

# Valuation categories
class ValuationCategories:
    """Stock valuation categories based on overall score."""
    UNDERVALUED_THRESHOLD = 80
    FAIRLY_VALUED_THRESHOLD = 60
    
    UNDERVALUED = "Undervalued"
    FAIRLY_VALUED = "Fairly valued"
    OVERVALUED = "Overvalued"

# ========================================
# UTILITY FUNCTIONS
# ========================================

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a logger with consistent formatting.
    
    Args:
        name: Name for the logger
        
    Returns:
        Configured logger instance
    """
    try:
        from logger import get_logger
        return get_logger(name)
    except ImportError:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

def convert_floats_to_decimals(obj: Any) -> Any:
    """
    Recursively converts float values to Decimal for DynamoDB compatibility.
    
    Args:
        obj: Object that may contain float values
        
    Returns:
        Object with floats converted to Decimals
    """
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {key: convert_floats_to_decimals(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimals(item) for item in obj]
    return obj

def format_market_cap(market_cap: Optional[float]) -> str:
    """Format market cap in human-readable format (B for billions, T for trillions)."""
    if market_cap is None:
        return "N/A"
    
    if market_cap >= 1_000_000_000_000:  # Trillions
        return f"${market_cap / 1_000_000_000_000:.2f}T"
    elif market_cap >= 1_000_000_000:  # Billions
        return f"${market_cap / 1_000_000_000:.2f}B"
    elif market_cap >= 1_000_000:  # Millions
        return f"${market_cap / 1_000_000:.2f}M"
    else:
        return f"${market_cap:,.0f}"

def get_performance_indicator(score: int) -> str:
    """Get visual performance indicator based on score."""
    if score >= 90:
        return "🟢 Excellent"
    elif score >= 80:
        return "🔵 Very Good"
    elif score >= 70:
        return "🟡 Good"
    elif score >= 60:
        return "🟠 Fair"
    elif score >= 50:
        return "🔴 Poor"
    else:
        return "⚫ Very Poor"

def get_valuation_indicator(valuation: str) -> str:
    """Get visual indicator for valuation category."""
    if valuation == "Undervalued":
        return "💚 Undervalued (Potential Buy)"
    elif valuation == "Fairly valued":
        return "💙 Fairly Valued (Hold/Monitor)"
    else:
        return "❤️ Overvalued (Consider Carefully)"

def get_metric_explanation(metric: str) -> str:
    """Get simple explanation for financial metrics."""
    explanations = {
        "peRatio": "Price-to-Earnings: How much investors pay per dollar of earnings (lower is generally better)",
        "roe": "Return on Equity: How efficiently the company uses shareholder money (higher is better)",
        "evToEbitda": "Enterprise Value to EBITDA: Company valuation relative to earnings (lower is better)",
        "eps": "Earnings Per Share: Company's profit per share (higher is better)"
    }
    return explanations.get(metric, "")

# ========================================
# FINANCIAL SCORING ENGINE
# ========================================

class FinancialMetricsScorer:
    """
    Handles the scoring logic for individual financial metrics.
    Separated for better maintainability and testing.
    """
    
    @staticmethod
    def score_pe_ratio(pe_ratio: float) -> int:
        """Score P/E ratio based on value ranges."""
        if pe_ratio < ScoringThresholds.PE_EXCELLENT:
            return ScoreValues.EXCELLENT
        elif pe_ratio < ScoringThresholds.PE_GOOD:
            return ScoreValues.GOOD
        else:
            return ScoreValues.POOR
    
    @staticmethod
    def score_roe(roe: float) -> int:
        """Score Return on Equity based on percentage ranges."""
        if roe > ScoringThresholds.ROE_EXCELLENT:
            return ScoreValues.EXCELLENT
        elif roe > ScoringThresholds.ROE_GOOD:
            return ScoreValues.GOOD
        else:
            return ScoreValues.POOR
    
    @staticmethod
    def score_ev_ebitda(ev_ebitda: float) -> int:
        """Score EV/EBITDA ratio based on value ranges."""
        if ev_ebitda < ScoringThresholds.EV_EBITDA_EXCELLENT:
            return ScoreValues.EXCELLENT
        elif ev_ebitda < ScoringThresholds.EV_EBITDA_GOOD:
            return ScoreValues.FAIR
        else:
            return ScoreValues.VERY_POOR
    
    @staticmethod
    def score_eps(eps: float) -> int:
        """Score Earnings Per Share based on value ranges."""
        if eps > ScoringThresholds.EPS_EXCELLENT:
            return ScoreValues.EXCELLENT
        elif eps > ScoringThresholds.EPS_GOOD:
            return ScoreValues.FAIR
        else:
            return ScoreValues.VERY_POOR

# ========================================
# MAIN SERVICE CLASS
# ========================================

class FinancialDataService:
    """
    Service for retrieving, processing, and scoring financial data for stock tickers.
    
    This service provides comprehensive financial analysis including:
    - Data retrieval from Yahoo Finance
    - Financial metrics scoring
    - Valuation assessment
    - Caching to DynamoDB (optional)
    """
    
    def __init__(self, use_caching: bool = True):
        """
        Initialize the financial data service.
        
        Args:
            use_caching: Whether to cache results to DynamoDB
        """
        self.logger = setup_logger("FinancialDataService")
        self.use_caching = use_caching
        self.dynamo_table_name = "FundamentalAnalysisData"
        self.metrics_scorer = FinancialMetricsScorer()

    def analyze_stock(self, ticker: str) -> Dict[str, Any]:
        """
        Complete financial analysis for a given stock ticker.
        
        This is the main entry point that orchestrates the entire analysis process.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            
        Returns:
            Complete analysis results including score, valuation, and rationale
        """
        try:
            ticker = ticker.upper().strip()
            self.logger.info(f"Starting financial analysis for ticker: {ticker}")
            
            # Validate input
            if not ticker:
                return self._create_error_response("Ticker symbol is required")

            # Fetch raw financial data
            financial_data = self._fetch_financial_data(ticker)
            if financial_data is None:
                return self._create_error_response(
                    f"Unable to retrieve financial data for '{ticker}'. "
                    f"Please verify the ticker symbol is correct."
                )

            # Perform comprehensive analysis
            analysis_results = self._perform_financial_analysis(financial_data)

            # Prepare successful response
            complete_results = {
                'success': True,
                'ticker': ticker,
                'timestamp': datetime.now().isoformat(),
                'data': {
                    **financial_data,
                    **analysis_results
                }
            }
            
            # Cache results if enabled
            if self.use_caching and complete_results['success']:
                self._save_to_cache(ticker, complete_results['data'])

            self.logger.info(f"Financial analysis completed successfully for {ticker}")
            return complete_results
            
        except Exception as error:
            self.logger.error(f"Analysis failed for {ticker}: {str(error)}")
            return self._create_error_response(
                f"Internal error during analysis: {str(error)}"
            )

    def _fetch_financial_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve financial data from Yahoo Finance with robust error handling.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary of financial metrics or None if data unavailable
        """
        self.logger.info(f"Fetching financial data from Yahoo Finance: {ticker}")
        
        try:
            stock = yf.Ticker(ticker)
            stock_info = stock.info
            
            # Validate that we received meaningful data
            if not stock_info or not stock_info.get("longName"):
                self.logger.warning(f"No valid company data found for: {ticker}")
                return None
                
            # Extract and organize financial metrics
            financial_metrics = {
                "companyName": stock_info.get("longName") or stock_info.get("shortName"),
                "peRatio": stock_info.get("trailingPE") or stock_info.get("forwardPE"),
                "roe": self._convert_roe_to_percentage(stock_info.get("returnOnEquity")),
                "evToEbitda": stock_info.get("enterpriseToEbitda"),
                "eps": stock_info.get("trailingEps") or stock_info.get("forwardEps"),
                "debtToEquity": stock_info.get("debtToEquity"),
                "marketCap": stock_info.get("marketCap"),
                "sector": stock_info.get("sector", "Unknown"),
                "industry": stock_info.get("industry", "Unknown"),
                "dataRetrievedAt": datetime.now().isoformat()
            }
            
            self.logger.info(f"Successfully retrieved financial data for {ticker}")
            return financial_metrics
            
        except Exception as error:
            self.logger.error(f"Error fetching data for {ticker}: {str(error)}")
            return None

    def _convert_roe_to_percentage(self, roe_decimal: Optional[float]) -> Optional[float]:
        """Convert ROE from decimal to percentage format for consistency."""
        if roe_decimal is not None:
            return roe_decimal * 100
        return None

    def _perform_financial_analysis(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze financial data and generate comprehensive scoring and valuation.
        
        Args:
            financial_data: Raw financial metrics
            
        Returns:
            Analysis results including score, valuation, and detailed breakdown
        """
        overall_score, score_breakdown, valid_metrics_count = self._calculate_overall_score(financial_data)
        
        if valid_metrics_count == 0:
            return {
                "score": 0,
                "valuation": "Unable to evaluate",
                "rationale": "Insufficient financial data available for meaningful analysis.",
                "scoreBreakdown": {},
                "metricsAnalyzed": 0,
                "userFriendlyReport": "❌ Unable to generate analysis report due to insufficient data."
            }

        valuation_category = self._determine_valuation_category(overall_score)
        analysis_rationale = self._generate_analysis_rationale(
            financial_data, overall_score, valuation_category, valid_metrics_count
        )
        
        # Create comprehensive analysis results
        analysis_results = {
            "score": round(overall_score, 2),
            "valuation": valuation_category,
            "rationale": analysis_rationale,
            "scoreBreakdown": score_breakdown,
            "metricsAnalyzed": valid_metrics_count
        }
        
        # Generate user-friendly report
        user_friendly_report = self._create_user_friendly_report(financial_data, analysis_results)
        analysis_results["userFriendlyReport"] = user_friendly_report
        
        return analysis_results

    def _calculate_overall_score(self, data: Dict[str, Any]) -> Tuple[float, Dict[str, int], int]:
        """
        Calculate weighted overall score based on available financial metrics.
        
        Args:
            data: Financial metrics data
            
        Returns:
            Tuple of (overall_score, score_breakdown, valid_metrics_count)
        """
        total_weighted_score = 0
        score_breakdown = {}
        valid_metrics_count = 0

        # Score P/E Ratio
        if self._is_valid_metric(data.get("peRatio")):
            pe_score = self.metrics_scorer.score_pe_ratio(data["peRatio"])
            total_weighted_score += pe_score * ScoringWeights.PE_RATIO
            score_breakdown["peRatio"] = pe_score
            valid_metrics_count += 1
        
        # Score Return on Equity
        if self._is_valid_metric(data.get("roe")):
            roe_score = self.metrics_scorer.score_roe(data["roe"])
            total_weighted_score += roe_score * ScoringWeights.ROE
            score_breakdown["roe"] = roe_score
            valid_metrics_count += 1

        # Score EV/EBITDA
        if self._is_valid_metric(data.get("evToEbitda")):
            ev_score = self.metrics_scorer.score_ev_ebitda(data["evToEbitda"])
            total_weighted_score += ev_score * ScoringWeights.EV_EBITDA
            score_breakdown["evToEbitda"] = ev_score
            valid_metrics_count += 1
            
        # Score Earnings Per Share
        if self._is_valid_metric(data.get("eps")):
            eps_score = self.metrics_scorer.score_eps(data["eps"])
            total_weighted_score += eps_score * ScoringWeights.EPS
            score_breakdown["eps"] = eps_score
            valid_metrics_count += 1

        return total_weighted_score, score_breakdown, valid_metrics_count

    def _is_valid_metric(self, value: Any) -> bool:
        """Check if a metric value is valid for scoring."""
        return value is not None and value > 0

    def _determine_valuation_category(self, score: float) -> str:
        """
        Determine valuation category based on overall score.
        
        Args:
            score: Overall financial score
            
        Returns:
            Valuation category string
        """
        if score >= ValuationCategories.UNDERVALUED_THRESHOLD:
            return ValuationCategories.UNDERVALUED
        elif score >= ValuationCategories.FAIRLY_VALUED_THRESHOLD:
            return ValuationCategories.FAIRLY_VALUED
        else:
            return ValuationCategories.OVERVALUED

    def _generate_analysis_rationale(
        self, 
        data: Dict[str, Any], 
        score: float, 
        valuation: str, 
        metrics_count: int
    ) -> str:
        """
        Generate human-readable analysis rationale.
        
        Args:
            data: Financial data
            score: Overall score
            valuation: Valuation category
            metrics_count: Number of metrics analyzed
            
        Returns:
            Detailed rationale string
        """
        company_name = data.get('companyName', 'Unknown Company')
        
        # Format metric values with proper handling of None values
        pe_display = f"{data.get('peRatio', 0):.2f}" if data.get('peRatio') else 'N/A'
        roe_display = f"{data.get('roe', 0):.2f}%" if data.get('roe') else 'N/A'
        ev_display = f"{data.get('evToEbitda', 0):.2f}" if data.get('evToEbitda') else 'N/A'
        eps_display = f"{data.get('eps', 0):.2f}" if data.get('eps') else 'N/A'
        
        rationale = (
            f"Financial Analysis for {company_name}: "
            f"Overall score of {score:.1f}/100 based on {metrics_count} key metrics. "
            f"Key metrics - P/E Ratio: {pe_display}, ROE: {roe_display}, "
            f"EV/EBITDA: {ev_display}, EPS: ${eps_display}. "
            f"Based on this analysis, the stock appears to be {valuation.lower()}."
        )
        
        return rationale

    def _create_user_friendly_report(self, financial_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> str:
        """
        Create a concise, user-friendly analysis report showing only essential information.
        
        Args:
            financial_data: Raw financial data
            analysis_results: Analysis results including scores and valuation
            
        Returns:
            Formatted report string
        """
        company_name = financial_data.get('companyName', 'Unknown Company')
        overall_score = analysis_results.get('score', 0)
        valuation = analysis_results.get('valuation', 'Unknown')
        score_breakdown = analysis_results.get('scoreBreakdown', {})
        
        # Concise header
        report_lines = [
            f"📊 {company_name} ({financial_data.get('sector', 'Unknown')})",
            f"💰 Market Cap: {format_market_cap(financial_data.get('marketCap'))}",
            ""
        ]
        
        # Overall assessment - simplified
        report_lines.extend([
            f"🎯 Overall Score: {overall_score:.1f}/100 {get_performance_indicator(int(overall_score))}",
            f"💡 {get_valuation_indicator(valuation)}",
            ""
        ])
        
        # Key metrics - condensed format
        report_lines.append("📊 Key Metrics:")
        
        metrics_info = [
            ("peRatio", "P/E Ratio", financial_data.get('peRatio'), "x"),
            ("roe", "ROE", financial_data.get('roe'), "%"),
            ("evToEbitda", "EV/EBITDA", financial_data.get('evToEbitda'), "x"),
            ("eps", "EPS", financial_data.get('eps'), "$")
        ]
        
        for metric_key, metric_name, value, unit in metrics_info:
            if value is not None and metric_key in score_breakdown:
                score = score_breakdown[metric_key]
                if unit == "$":
                    value_str = f"${value:.2f}"
                elif unit == "%":
                    value_str = f"{value:.1f}%"
                else:
                    value_str = f"{value:.1f}{unit}"
                
                report_lines.append(f"  {metric_name}: {value_str} {get_performance_indicator(score)}")
        
        # Simple investment guidance
        report_lines.append("")
        if valuation == "Undervalued":
            report_lines.append("💚 Recommendation: Consider for investment")
        elif valuation == "Fairly valued":
            report_lines.append("💙 Recommendation: Hold or monitor")
        else:
            report_lines.append("❤️ Recommendation: Proceed with caution")
        
        # Brief disclaimer
        report_lines.extend([
            "",
            "⚠️ For informational purposes only. Not investment advice."
        ])
        
        return "\n".join(report_lines)

    def _save_to_cache(self, ticker: str, analysis_data: Dict[str, Any]) -> None:
        """
        Save analysis results to DynamoDB cache.
        
        Args:
            ticker: Stock ticker symbol
            analysis_data: Complete analysis data to cache
        """
        try:
            import boto3
            
            dynamo_resource = boto3.resource("dynamodb")
            table = dynamo_resource.Table(self.dynamo_table_name)
            
            # Prepare data for DynamoDB (convert floats to Decimals)
            cache_item = {
                "ticker": ticker.upper(),
                "lastUpdated": datetime.utcnow().isoformat(),
                **convert_floats_to_decimals(analysis_data)
            }
            
            table.put_item(Item=cache_item)
            self.logger.info(f"Successfully cached analysis results for {ticker}")
            
        except ImportError:
            self.logger.warning("boto3 not available - skipping DynamoDB caching")
        except Exception as error:
            self.logger.error(f"Failed to cache data for {ticker}: {str(error)}")

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create standardized error response.
        
        Args:
            error_message: Description of the error
            
        Returns:
            Standardized error response dictionary
        """
        return {
            'success': False,
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        }

# ========================================
# AWS LAMBDA HANDLER
# ========================================

def lambda_handler(event, context):
    """
    AWS Lambda handler optimized for Amazon Bedrock Agents.
    
    This handler processes financial analysis requests and returns
    comprehensive stock analysis including scoring and valuation.
    
    Args:
        event: Lambda event containing request parameters
        context: Lambda context (unused)
        
    Returns:
        Bedrock Agent compatible response with analysis results
    """
    logger = setup_logger("FinancialAnalysisHandler")
    logger.info(f"Processing financial analysis request: {json.dumps(event)}")

    # Extract Bedrock Agent metadata
    action_group = event.get('actionGroup', 'FinancialAnalysisActionGroup')
    function_name = event.get('function', 'getFinancialAnalysis')
    message_version = event.get('messageVersion', '1.0')

    try:
        # Extract ticker symbol from event
        ticker_symbol = extract_ticker_from_event(event, logger)
        
        if not ticker_symbol:
            error_response = create_bedrock_error_response(
                "Missing required parameter: ticker symbol",
                action_group, function_name, message_version
            )
            return error_response
        
        # Perform financial analysis
        financial_service = FinancialDataService()
        analysis_results = financial_service.analyze_stock(ticker_symbol)
        
        # Create Bedrock-compatible response
        bedrock_response = create_bedrock_success_response(
            analysis_results, action_group, function_name, message_version
        )
        
        logger.info(f"Analysis completed successfully for {ticker_symbol}")
        return bedrock_response
        
    except Exception as error:
        logger.error(f"Lambda handler failed: {str(error)}")
        error_response = create_bedrock_error_response(
            f"Internal server error: {str(error)}",
            action_group, function_name, message_version
        )
        return error_response

def extract_ticker_from_event(event: Dict[str, Any], logger: logging.Logger) -> str:
    """
    Extract ticker symbol from various event formats.
    
    Args:
        event: Lambda event
        logger: Logger instance
        
    Returns:
        Ticker symbol or empty string if not found
    """
    ticker = ''
    
    # Try Bedrock Agent parameters format
    if 'parameters' in event and isinstance(event.get('parameters'), list):
        for param in event['parameters']:
            if param.get('name') == 'ticker' and param.get('value'):
                ticker = str(param['value']).upper().strip()
                break
    
    # Fallback to direct ticker parameter
    if not ticker and event.get('ticker'):
        ticker = str(event['ticker']).upper().strip()

    logger.info(f"Extracted ticker symbol: '{ticker}'")
    return ticker

def create_bedrock_success_response(
    analysis_results: Dict[str, Any],
    action_group: str,
    function_name: str,
    message_version: str
) -> Dict[str, Any]:
    """Create Bedrock Agent compatible success response with user-friendly formatting."""
    
    # Prepare user-friendly response
    if analysis_results.get('success') and analysis_results.get('data', {}).get('userFriendlyReport'):
        # Show the user-friendly report prominently
        formatted_response = {
            "analysisReport": analysis_results['data']['userFriendlyReport'],
            "technicalData": analysis_results  # Keep full data for systems that need it
        }
    else:
        # Fallback to original format if no user-friendly report available
        formatted_response = analysis_results
    
    response_body = {"TEXT": {"body": json.dumps(formatted_response, default=str)}}
    
    return {
        'messageVersion': message_version,
        'response': {
            'actionGroup': action_group,
            'function': function_name,
            'functionResponse': {'responseBody': response_body}
        }
    }

def create_bedrock_error_response(
    error_message: str,
    action_group: str,
    function_name: str,
    message_version: str
) -> Dict[str, Any]:
    """Create Bedrock Agent compatible error response with user-friendly formatting."""
    service = FinancialDataService()
    error_details = service._create_error_response(error_message)
    
    # Create user-friendly error message
    user_friendly_error = {
        "analysisReport": f"❌ ANALYSIS ERROR\n{'='*30}\n\n🚨 {error_message}\n\nPlease check the ticker symbol and try again.",
        "technicalData": error_details
    }
    
    response_body = {"TEXT": {"body": json.dumps(user_friendly_error)}}
    
    return {
        'messageVersion': message_version,
        'response': {
            'actionGroup': action_group,
            'function': function_name,
            'functionResponse': {'responseBody': response_body}
        }
    }

# ========================================
# LOCAL TESTING AND DEMONSTRATION
# ========================================

def create_mock_financial_data():
    """Create mock financial data for local testing."""
    return {
        "AAPL": {
            'longName': 'Apple Inc.',
            'trailingPE': 28.5,
            'returnOnEquity': 1.6,  # Will be converted to 160%
            'enterpriseToEbitda': 22.0,
            'trailingEps': 6.1,
            'debtToEquity': 150.0,
            'marketCap': 2800000000000,
            'sector': 'Technology',
            'industry': 'Consumer Electronics'
        },
        "NVDA": {
            'longName': 'NVIDIA Corporation',
            'trailingPE': 51.10,
            'returnOnEquity': 1.1546,  # Will be converted to 115.46%
            'enterpriseToEbitda': 43.19,
            'trailingEps': 3.00,
            'debtToEquity': 12.267,
            'marketCap': 3852996182016,
            'sector': 'Technology',
            'industry': 'Semiconductors'
        },
        "MSFT": {
            'longName': 'Microsoft Corporation',
            'trailingPE': 35.0,
            'returnOnEquity': 0.40,  # Will be converted to 40%
            'enterpriseToEbitda': 25.0,
            'trailingEps': 11.5,
            'debtToEquity': 45.0,
            'marketCap': 3100000000000,
            'sector': 'Technology',
            'industry': 'Software - Infrastructure'
        },
        "INVALID": {}  # Simulates invalid ticker
    }

class MockYFinanceTicker:
    """Mock yfinance.Ticker for local testing without network calls."""
    
    def __init__(self, ticker: str):
        self._ticker = ticker.upper()
        self.info = create_mock_financial_data().get(self._ticker, {})

def run_local_demonstration():
    """Run comprehensive local testing demonstration."""
    # Replace yfinance with mock for testing
    yf.Ticker = MockYFinanceTicker

    print("=" * 60)
    print("FINANCIAL ANALYSIS - DEMONSTRATION")
    print("=" * 60)
    
    test_scenarios = [
        {
            "description": "NVIDIA Corporation (NVDA) Analysis",
            "event": {
                "messageVersion": "1.0",
                "actionGroup": "FinancialAnalysisActionGroup",
                "function": "getFinancialAnalysis",
                "parameters": [{"name": "ticker", "value": "NVDA"}],
                "sessionId": "demo-session-1"
            }
        },
        {
            "description": "Invalid ticker example",
            "event": {
                "messageVersion": "1.0",
                "actionGroup": "FinancialAnalysisActionGroup",
                "function": "getFinancialAnalysis",
                "parameters": [{"name": "ticker", "value": "INVALID"}],
                "sessionId": "demo-session-2"
            }
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['description']}")
        print("-" * 40)
        
        # Execute test
        result = lambda_handler(scenario['event'], None)
        
        # Display results
        try:
            response_body = result['response']['functionResponse']['responseBody']['TEXT']['body']
            parsed_results = json.loads(response_body)
            
            # Check if we have the new user-friendly format
            if 'analysisReport' in parsed_results:
                print(parsed_results['analysisReport'])
            else:
                # Fallback to original format
                print(json.dumps(parsed_results, indent=2))
                
        except (KeyError, json.JSONDecodeError) as error:
            print(f"Error parsing response: {error}")
            print(f"Raw response: {json.dumps(result, indent=2)}")

    print(f"\n{'=' * 60}")
    print("🎉 DEMONSTRATION COMPLETED!")
    print("✨ Now showing concise, user-friendly analysis reports")
    print(f"{'=' * 60}")

# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    run_local_demonstration()
