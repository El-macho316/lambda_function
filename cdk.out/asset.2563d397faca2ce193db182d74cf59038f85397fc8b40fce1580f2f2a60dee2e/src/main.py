import pandas as pd
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
import os
from datetime import datetime
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class StockVectorDatabase:
    """
    A comprehensive system for managing stock data in Pinecone vector database.
    Handles data preprocessing, vector creation, uploading, and similarity searches.
    """
    
    def __init__(self, api_key: str, index_name: str = "stock-analysis", 
                 environment: str = "us-east-1"):
        """
        Initialize the Pinecone client and setup.
        
        Args:
            api_key: Pinecone API key
            index_name: Name for the Pinecone index
            environment: Pinecone environment/region
        """
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.environment = environment
        self.index = None
        self.logger = self._setup_logger()
        
        # Key financial metrics for vector representation
        self.key_metrics = [
            'OPEN_PRC', 'CLOSE_PRC', 'LOW_PRC', 'HIGH_PRC', 'DVD_YIELD',
            'BOOK_VALUE', 'PB_RATIO', 'PE_RATIO', 'Q_VOLUME', 'MKT_CAP',
            'TURNOVER', 'Baro_Index', 'Fundamental_Score', 'Technic_Score',
            'Quant_Score', 'Total_score', 'Rank', 'SET_CLOSE', 'SET50_CLOSE',
            'SECTOR_YIELD', 'SECTOR_MKT_PE', 'SECTOR_MKT_CAP', 'SECTOR_MKT_PBV'
        ]
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the vector database operations."""
        logger = logging.getLogger('StockVectorDB')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def create_index(self, dimension: int = 23, metric: str = "cosine") -> bool:
        """
        Create a new Pinecone index for stock data.
        
        Args:
            dimension: Vector dimension (number of metrics)
            metric: Distance metric for similarity search
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if index already exists
            if self.index_name in self.pc.list_indexes().names():
                self.logger.info(f"Index '{self.index_name}' already exists")
                self.index = self.pc.Index(self.index_name)
                return True
            
            # Create new index
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud='aws',
                    region=self.environment
                )
            )
            
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            
            self.index = self.pc.Index(self.index_name)
            self.logger.info(f"Successfully created index '{self.index_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create index: {str(e)}")
            return False
    
    def load_stock_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load and preprocess stock data from CSV.
        
        Args:
            csv_path: Path to the stock database CSV file
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            # Load the CSV
            df = pd.read_csv(csv_path)
            self.logger.info(f"Loaded {len(df)} records from {csv_path}")
            
            # Clean column names (remove spaces)
            df.columns = df.columns.str.strip()
            
            # Convert date column
            df['D_TRADE'] = pd.to_datetime(df['D_TRADE'])
            
            # Clean security names
            df['N_SECURITY'] = df['N_SECURITY'].str.strip()
            
            # Handle missing values in key metrics
            for metric in self.key_metrics:
                if metric in df.columns:
                    df[metric] = pd.to_numeric(df[metric], errors='coerce').fillna(0)
            
            self.logger.info("Data preprocessing completed")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load stock data: {str(e)}")
            raise
    
    def create_stock_vectors(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert stock data to vectors for Pinecone upload.
        
        Args:
            df: Preprocessed stock DataFrame
            
        Returns:
            List of vector dictionaries ready for Pinecone
        """
        vectors = []
        
        for idx, row in df.iterrows():
            try:
                # Create vector from key metrics
                vector_values = []
                for metric in self.key_metrics:
                    if metric in df.columns:
                        value = float(row[metric]) if pd.notna(row[metric]) else 0.0
                        vector_values.append(value)
                    else:
                        vector_values.append(0.0)
                
                # Normalize the vector
                vector_array = np.array(vector_values)
                if np.linalg.norm(vector_array) > 0:
                    vector_array = vector_array / np.linalg.norm(vector_array)
                
                # Create vector ID
                vector_id = f"{row['N_SECURITY']}_{row['D_TRADE'].strftime('%Y%m%d')}"
                
                # Create metadata with additional financial data
                metadata = {
                    'id': vector_id,
                    'security': str(row['N_SECURITY']),
                    'trade_date': str(row['D_TRADE']),
                    'close_price': float(row.get('CLOSE_PRC', 0)),
                    'open_price': float(row.get('OPEN_PRC', 0)),
                    'high_price': float(row.get('HIGH_PRC', 0)),
                    'low_price': float(row.get('LOW_PRC', 0)),
                    'market_cap': float(row.get('MKT_CAP', 0)),
                    'pe_ratio': float(row.get('PE_RATIO', 0)),
                    'pb_ratio': float(row.get('PB_RATIO', 0)),
                    'book_value': float(row.get('BOOK_VALUE', 0)),
                    'dividend_yield': float(row.get('DVD_YIELD', 0)),
                    'total_score': float(row.get('Total_score', 0)),
                    'fundamental_score': float(row.get('Fundamental_Score', 0)),
                    'technical_score': float(row.get('Technic_Score', 0)),
                    'quant_score': float(row.get('Quant_Score', 0)),
                    'rank': int(row.get('Rank', 0)) if pd.notna(row.get('Rank')) else 0,
                    'sector': str(row.get('I_SECTOR', 'Unknown')),
                    'industry': str(row.get('I_INDUSTRY', 'Unknown')),
                    'market': str(row.get('I_MARKET', 'Unknown')),
                    'volume': float(row.get('Q_VOLUME', 0)),
                    'turnover': float(row.get('TURNOVER', 0))
                }
                
                # Create formatted document and add to metadata
                document = self.prepare_document(metadata)
                metadata['document'] = document
                
                vectors.append({
                    'id': vector_id,
                    'values': vector_array.tolist(),
                    'metadata': metadata
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to process row {idx}: {str(e)}")
                continue
        
        self.logger.info(f"Created {len(vectors)} vectors")
        return vectors
    
    def upload_vectors(self, vectors: List[Dict[str, Any]], batch_size: int = 100) -> bool:
        """
        Upload vectors to Pinecone in batches.
        
        Args:
            vectors: List of vector dictionaries
            batch_size: Number of vectors to upload per batch
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.index:
                self.logger.error("Index not initialized. Call create_index() first.")
                return False
            
            # Upload in batches
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                self.logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
                time.sleep(0.1)  # Small delay to avoid rate limits
            
            self.logger.info(f"Successfully uploaded {len(vectors)} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upload vectors: {str(e)}")
            return False
    
    def find_similar_stocks(self, target_security: str, target_date: str = None, 
                          top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Find stocks similar to a target stock based on financial metrics.
        
        Args:
            target_security: Security name to find similar stocks for
            target_date: Specific date (YYYY-MM-DD format), uses latest if None
            top_k: Number of similar stocks to return
            
        Returns:
            List of similar stocks with similarity scores
        """
        try:
            if not self.index:
                self.logger.error("Index not initialized")
                return []
            
            # Query for the target stock
            if target_date:
                query_id = f"{target_security}_{datetime.strptime(target_date, '%Y-%m-%d').strftime('%Y%m%d')}"
            else:
                # Find the latest record for this security
                query_response = self.index.query(
                    vector=[0] * len(self.key_metrics),  # Dummy vector for filter search
                    filter={"security": {"$eq": target_security}},
                    top_k=1,
                    include_metadata=True
                )
                if not query_response.matches:
                    self.logger.warning(f"No data found for security: {target_security}")
                    return []
                query_id = query_response.matches[0].id
            
            # Get the target vector
            fetch_response = self.index.fetch(ids=[query_id])
            if query_id not in fetch_response.vectors:
                self.logger.warning(f"Vector not found for ID: {query_id}")
                return []
            
            target_vector = fetch_response.vectors[query_id].values
            
            # Find similar stocks (excluding the target itself)
            similar_response = self.index.query(
                vector=target_vector,
                top_k=top_k + 5,  # Get extra to filter out same security
                include_metadata=True,
                filter={"security": {"$ne": target_security}}
            )
            
            # Format results
            similar_stocks = []
            for match in similar_response.matches[:top_k]:
                similar_stocks.append({
                    'security': match.metadata['security'],
                    'similarity_score': float(match.score),
                    'close_price': match.metadata['close_price'],
                    'market_cap': match.metadata['market_cap'],
                    'pe_ratio': match.metadata['pe_ratio'],
                    'total_score': match.metadata['total_score'],
                    'rank': match.metadata['rank'],
                    'sector': match.metadata['sector'],
                    'trade_date': match.metadata['trade_date'],
                    'document': match.metadata.get('document', '')
                })
            
            return similar_stocks
            
        except Exception as e:
            self.logger.error(f"Failed to find similar stocks: {str(e)}")
            return []
    
    def search_by_criteria(self, criteria: Dict[str, Any], top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Search stocks based on specific criteria.
        
        Args:
            criteria: Dictionary with search criteria (e.g., {'sector': 'Technology', 'min_score': 80})
            top_k: Number of results to return
            
        Returns:
            List of stocks matching criteria
        """
        try:
            if not self.index:
                self.logger.error("Index not initialized")
                return []
            
            # Build filter
            filter_dict = {}
            for key, value in criteria.items():
                if key.startswith('min_'):
                    metric = key.replace('min_', '')
                    filter_dict[metric] = {"$gte": value}
                elif key.startswith('max_'):
                    metric = key.replace('max_', '')
                    filter_dict[metric] = {"$lte": value}
                else:
                    filter_dict[key] = {"$eq": value}
            
            # Query with filter
            response = self.index.query(
                vector=[0] * len(self.key_metrics),  # Dummy vector for filter-only search
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results
            results = []
            for match in response.matches:
                results.append({
                    'security': match.metadata['security'],
                    'close_price': match.metadata['close_price'],
                    'market_cap': match.metadata['market_cap'],
                    'pe_ratio': match.metadata['pe_ratio'],
                    'total_score': match.metadata['total_score'],
                    'rank': match.metadata['rank'],
                    'sector': match.metadata['sector'],
                    'trade_date': match.metadata['trade_date'],
                    'document': match.metadata.get('document', '')
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search by criteria: {str(e)}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        try:
            if not self.index:
                return {"error": "Index not initialized"}
            
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'namespaces': stats.namespaces
            }
        except Exception as e:
            return {"error": str(e)}
    
    def prepare_document(self, record: Dict[str, Any]) -> str:
        """
        Create a formatted text document for a stock record with valuation analysis.
        
        Args:
            record: Dictionary containing stock data
            
        Returns:
            Formatted text document string with valuation analysis
        """
        try:
            # Format market cap with commas
            market_cap = record.get('market_cap', 0)
            if isinstance(market_cap, (int, float)):
                market_cap_str = f"{market_cap:,.0f}"
            else:
                market_cap_str = str(market_cap)
            
            # Perform valuation analysis
            valuation_analysis = self._analyze_valuation(record)
            
            # Create formatted document
            document = f"""
ID: {record.get('id', 'N/A')}
Company: {record.get('security', 'N/A')}
Trade Date: {record.get('trade_date', 'N/A')}
Close Price: {record.get('close_price', 'N/A')}
Market: {record.get('market', 'N/A')}
Market Cap: {market_cap_str}
P/B Ratio: {record.get('pb_ratio', 'N/A')}
P/E Ratio: {record.get('pe_ratio', 'N/A')}
Rank: {record.get('rank', 'N/A')}
Sector: {record.get('sector', 'N/A')}
Industry: {record.get('industry', 'N/A')}
Total Score: {record.get('total_score', 'N/A')}

VALUATION ANALYSIS:
{valuation_analysis['analysis']}

INVESTMENT RECOMMENDATION:
{valuation_analysis['recommendation']}

Summary:
{record.get('security', 'N/A')} traded at {record.get('close_price', 'N/A')} on {record.get('trade_date', 'N/A')} with a P/E of {record.get('pe_ratio', 'N/A')} and P/B of {record.get('pb_ratio', 'N/A')}. Market cap was approximately {market_cap_str}. Sector {record.get('sector', 'N/A')}, Industry {record.get('industry', 'N/A')}. {valuation_analysis['summary']}
"""
            return document.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to prepare document: {str(e)}")
            return f"Error preparing document for {record.get('security', 'Unknown')}"
    
    def _analyze_valuation(self, record: Dict[str, Any]) -> Dict[str, str]:
        """
        Analyze stock valuation using available metrics and logic from main.py.
        
        Args:
            record: Stock record data
            
        Returns:
            Dictionary with valuation analysis, recommendation, and summary
        """
        try:
            # Get key metrics
            pe_ratio = record.get('pe_ratio', 0)
            pb_ratio = record.get('pb_ratio', 0)
            total_score = record.get('total_score', 0)
            rank = record.get('rank', 0)
            dividend_yield = record.get('dividend_yield', 0)
            
            # Convert to numeric values
            pe_ratio = float(pe_ratio) if pe_ratio and pe_ratio != 'N/A' else 0
            pb_ratio = float(pb_ratio) if pb_ratio and pb_ratio != 'N/A' else 0
            total_score = float(total_score) if total_score and total_score != 'N/A' else 0
            rank = int(rank) if rank and rank != 'N/A' else 0
            dividend_yield = float(dividend_yield) if dividend_yield and dividend_yield != 'N/A' else 0
            
            # Calculate additional metrics if possible
            calculated_metrics = self._calculate_additional_metrics(record)
            
            # Valuation analysis based on available metrics
            valuation_indicators = []
            
            # P/E Ratio analysis (from main.py logic)
            if pe_ratio > 0:
                if pe_ratio < 15:
                    valuation_indicators.append("🟢 Excellent P/E Ratio (< 15) - Strong value")
                elif pe_ratio < 20:
                    valuation_indicators.append("🟡 Good P/E Ratio (15-20) - Reasonable value")
                else:
                    valuation_indicators.append("🔴 High P/E Ratio (> 20) - May be overvalued")
            else:
                valuation_indicators.append("⚪ P/E Ratio not available")
            
            # P/B Ratio analysis 
            if pb_ratio > 0:
                if pb_ratio < 1.0:
                    valuation_indicators.append("🟢 Low P/B Ratio (< 1.0) - Trading below book value")
                elif pb_ratio < 3.0:
                    valuation_indicators.append("🟡 Moderate P/B Ratio (1.0-3.0) - Fair valuation")
                else:
                    valuation_indicators.append("🔴 High P/B Ratio (> 3.0) - May be overvalued")
            else:
                valuation_indicators.append("⚪ P/B Ratio not available")
            
            # Dividend Yield analysis
            if dividend_yield > 0:
                if dividend_yield >= 4.0:
                    valuation_indicators.append("🟢 High Dividend Yield (≥ 4%) - Good income potential")
                elif dividend_yield >= 2.0:
                    valuation_indicators.append("🟡 Moderate Dividend Yield (2-4%) - Decent income")
                else:
                    valuation_indicators.append("🔶 Low Dividend Yield (< 2%) - Growth focused")
            else:
                valuation_indicators.append("⚪ No dividend yield")
            
            # Total Score analysis (your existing scoring system)
            if total_score >= 80:
                valuation_indicators.append("🟢 High Total Score (≥ 80) - Strong fundamentals")
                overall_valuation = "UNDERVALUED"
            elif total_score >= 60:
                valuation_indicators.append("🟡 Good Total Score (60-79) - Decent fundamentals")
                overall_valuation = "FAIRLY VALUED"
            elif total_score > 0:
                valuation_indicators.append("🔴 Low Total Score (< 60) - Weak fundamentals")
                overall_valuation = "OVERVALUED"
            else:
                overall_valuation = "UNABLE TO EVALUATE"
            
            # Rank analysis
            if rank > 0:
                if rank <= 100:
                    valuation_indicators.append(f"🟢 Top Rank (#{rank}) - Among best performers")
                elif rank <= 500:
                    valuation_indicators.append(f"🟡 Good Rank (#{rank}) - Above average")
                else:
                    valuation_indicators.append(f"🔴 Lower Rank (#{rank}) - Below average")
            
            # Add calculated metrics if available
            if calculated_metrics['eps'] > 0:
                valuation_indicators.append(f"💰 EPS: ${calculated_metrics['eps']:.2f}")
            
            if calculated_metrics['roe'] > 0:
                valuation_indicators.append(f"📈 ROE: {calculated_metrics['roe']:.1f}%")
            
            # Generate analysis text
            analysis_text = "Based on available financial metrics:\n" + "\n".join(f"• {indicator}" for indicator in valuation_indicators)
            
            # Generate recommendation
            if overall_valuation == "UNDERVALUED":
                recommendation = "💚 BUY - Stock appears undervalued with strong fundamentals"
            elif overall_valuation == "FAIRLY VALUED":
                recommendation = "💙 HOLD - Stock appears fairly valued, monitor for opportunities"
            elif overall_valuation == "OVERVALUED":
                recommendation = "❤️ CAUTION - Stock may be overvalued, consider carefully"
            else:
                recommendation = "⚪ INSUFFICIENT DATA - Unable to make recommendation"
            
            # Generate summary  
            summary = f"Analysis indicates this stock is {overall_valuation.lower()}."
            
            return {
                'analysis': analysis_text,
                'recommendation': recommendation,
                'summary': summary,
                'valuation': overall_valuation
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze valuation: {str(e)}")
            return {
                'analysis': "Unable to perform valuation analysis due to data issues.",
                'recommendation': "⚪ INSUFFICIENT DATA - Unable to make recommendation",
                'summary': "Valuation analysis could not be completed.",
                'valuation': "UNABLE TO EVALUATE"
            }
    
    def _calculate_additional_metrics(self, record: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate additional financial metrics from available data.
        
        Args:
            record: Stock record data
            
        Returns:
            Dictionary with calculated metrics (EPS, ROE, etc.)
        """
        try:
            # Get available data
            close_price = record.get('close_price', 0)
            pe_ratio = record.get('pe_ratio', 0)
            pb_ratio = record.get('pb_ratio', 0)
            book_value = record.get('book_value', 0)
            market_cap = record.get('market_cap', 0)
            
            # Convert to numeric
            close_price = float(close_price) if close_price else 0
            pe_ratio = float(pe_ratio) if pe_ratio else 0
            pb_ratio = float(pb_ratio) if pb_ratio else 0
            book_value = float(book_value) if book_value else 0
            market_cap = float(market_cap) if market_cap else 0
            
            calculated = {
                'eps': 0,
                'roe': 0,
                'ev_ebitda': 0
            }
            
            # Calculate EPS from P/E ratio and price
            if pe_ratio > 0 and close_price > 0:
                calculated['eps'] = close_price / pe_ratio
            
            # Calculate ROE from P/B ratio and market metrics
            if pb_ratio > 0 and book_value > 0:
                # ROE estimation: This is a simplified calculation
                # Real ROE = Net Income / Shareholder Equity
                # We can approximate if we have more data
                pass  # Would need more data for accurate ROE calculation
            
            return calculated
            
        except Exception as e:
            self.logger.error(f"Failed to calculate additional metrics: {str(e)}")
            return {'eps': 0, 'roe': 0, 'ev_ebitda': 0}
    
    def get_stock_document(self, target_security: str, target_date: str = None) -> str:
        """
        Get the formatted document for a specific stock.
        
        Args:
            target_security: Security name
            target_date: Specific date (YYYY-MM-DD format), uses latest if None
            
        Returns:
            Formatted document string or empty string if not found
        """
        try:
            if not self.index:
                self.logger.error("Index not initialized")
                return ""
            
            # Query for the target stock
            if target_date:
                query_id = f"{target_security}_{datetime.strptime(target_date, '%Y-%m-%d').strftime('%Y%m%d')}"
                fetch_response = self.index.fetch(ids=[query_id])
                if query_id in fetch_response.vectors:
                    return fetch_response.vectors[query_id].metadata.get('document', '')
            else:
                # Find the latest record for this security
                query_response = self.index.query(
                    filter={"security": {"$eq": target_security}},
                    top_k=1,
                    include_metadata=True
                )
                if query_response.matches:
                    return query_response.matches[0].metadata.get('document', '')
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Failed to get document for {target_security}: {str(e)}")
            return ""
    
    def delete_index(self) -> bool:
        """Delete the Pinecone index."""
        try:
            self.pc.delete_index(self.index_name)
            self.logger.info(f"Deleted index: {self.index_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete index: {str(e)}")
            return False


# Example usage and helper functions
def create_sample_stock_data() -> pd.DataFrame:
    """
    Create sample stock data for testing Pinecone functionality.
    
    Returns:
        DataFrame with sample stock data
    """
    sample_data = {
        'N_SECURITY': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX', 'CRM', 'UBER'],
        'D_TRADE': ['2024-01-15'] * 10,
        'OPEN_PRC': [185.0, 2650.0, 380.0, 240.0, 155.0, 350.0, 550.0, 480.0, 260.0, 65.0],
        'CLOSE_PRC': [187.5, 2680.0, 385.0, 245.0, 158.0, 355.0, 560.0, 485.0, 265.0, 67.0],
        'LOW_PRC': [183.0, 2640.0, 378.0, 238.0, 152.0, 348.0, 545.0, 475.0, 258.0, 63.0],
        'HIGH_PRC': [189.0, 2690.0, 388.0, 248.0, 160.0, 358.0, 565.0, 488.0, 268.0, 69.0],
        'DVD_YIELD': [0.45, 0.0, 0.68, 0.0, 0.0, 0.0, 0.16, 0.0, 0.0, 0.0],
        'BOOK_VALUE': [3.85, 320.0, 15.2, 28.5, 84.0, 46.0, 26.0, 45.0, 6.8, 8.5],
        'PB_RATIO': [48.7, 8.4, 25.3, 8.6, 1.9, 7.7, 21.5, 10.8, 39.0, 7.9],
        'PE_RATIO': [29.2, 22.5, 32.1, 45.8, 42.3, 24.6, 65.2, 35.4, 48.5, 28.7],
        'Q_VOLUME': [45678900, 28934500, 32145600, 89234700, 41256300, 19876500, 52341800, 15432100, 8765400, 25643200],
        'MKT_CAP': [2875000000000, 1650000000000, 2825000000000, 780000000000, 1520000000000, 875000000000, 1380000000000, 215000000000, 175000000000, 145000000000],
        'TURNOVER': [8567432100, 7734521000, 12365478900, 21876543200, 6543210987, 7054321098, 29321456780, 7456321098, 2345678901, 1718293456],
        'Baro_Index': [75.5, 82.3, 78.9, 65.2, 80.1, 73.6, 88.7, 69.4, 71.8, 66.3],
        'Fundamental_Score': [85.2, 78.6, 82.4, 72.3, 79.8, 75.1, 88.9, 73.2, 76.5, 68.7],
        'Technic_Score': [76.8, 84.2, 79.1, 88.5, 72.6, 81.3, 92.4, 75.8, 79.2, 74.1],
        'Quant_Score': [82.1, 80.5, 85.3, 79.7, 83.2, 78.9, 91.6, 77.4, 81.8, 72.6],
        'Total_score': [81.4, 81.1, 82.3, 80.2, 78.5, 78.4, 90.9, 75.5, 79.2, 71.8],
        'Rank': [15, 18, 12, 22, 28, 29, 3, 45, 35, 62],
        'SET_CLOSE': [1654.8] * 10,
        'SET50_CLOSE': [1123.4] * 10,
        'SECTOR_YIELD': [2.1, 1.8, 2.3, 1.5, 1.9, 2.0, 1.7, 2.2, 2.4, 1.6],
        'SECTOR_MKT_PE': [28.5, 25.3, 29.1, 42.7, 38.9, 26.8, 58.2, 32.1, 45.3, 31.2],
        'SECTOR_MKT_CAP': [3200000000000, 2100000000000, 3100000000000, 950000000000, 1800000000000, 1200000000000, 1600000000000, 280000000000, 220000000000, 180000000000],
        'SECTOR_MKT_PBV': [7.2, 4.8, 6.9, 5.1, 3.2, 5.8, 8.4, 6.1, 7.8, 4.6],
        'I_SECTOR': ['Technology', 'Technology', 'Technology', 'Automotive', 'E-commerce', 'Technology', 'Technology', 'Entertainment', 'Technology', 'Transportation'],
        'I_INDUSTRY': ['Consumer Electronics', 'Internet Services', 'Software', 'Electric Vehicles', 'E-commerce', 'Social Media', 'Semiconductors', 'Streaming', 'Cloud Software', 'Ride Sharing'],
        'I_MARKET': ['NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NYSE', 'NYSE']
    }
    
    return pd.DataFrame(sample_data)

def setup_stock_database(api_key: str, csv_path: str = None) -> StockVectorDatabase:
    """
    Complete setup of stock vector database using sample data instead of CSV.
    
    Args:
        api_key: Pinecone API key
        csv_path: Ignored - using sample data instead
        
    Returns:
        Configured StockVectorDatabase instance
    """
    # Initialize the database
    stock_db = StockVectorDatabase(api_key)
    
    # Create index
    if not stock_db.create_index():
        raise Exception("Failed to create Pinecone index")
    
    # Create sample data instead of loading from CSV
    print("Creating sample stock data for testing...")
    df = create_sample_stock_data()
    print(f"Generated {len(df)} sample stock records")
    
    # Process the sample data
    vectors = stock_db.create_stock_vectors(df)
    
    # Upload to Pinecone
    if not stock_db.upload_vectors(vectors):
        raise Exception("Failed to upload vectors to Pinecone")
    
    return stock_db


def demo_stock_search(stock_db: StockVectorDatabase):
    """Demonstrate various search capabilities."""
    print("=== Stock Vector Database Demo ===\n")
    
    # Get index stats
    stats = stock_db.get_index_stats()
    print("Index Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Find similar stocks example
    print("Finding stocks similar to 'PTT'...")
    similar = stock_db.find_similar_stocks("PTT", top_k=5)
    for stock in similar:
        print(f"  {stock['security']}: Score {stock['similarity_score']:.3f}, "
              f"Price {stock['close_price']}, Total Score {stock['total_score']}")
    print()
    
    # Demo formatted document
    if similar:
        print("=== Sample Formatted Document ===")
        first_stock = similar[0]
        if 'document' in first_stock and first_stock['document']:
            print(first_stock['document'])
        else:
            # Fallback to get_stock_document method
            document = stock_db.get_stock_document(first_stock['security'])
            if document:
                print(document)
            else:
                print("No document found")
        print()
    
    # Search by criteria example
    print("Searching for high-performing stocks in Technology sector...")
    criteria = {
        'sector': 'Technology',
        'min_total_score': 80,
        'max_pe_ratio': 25
    }
    results = stock_db.search_by_criteria(criteria, top_k=5)
    for stock in results:
        print(f"  {stock['security']}: Score {stock['total_score']}, "
              f"PE {stock['pe_ratio']}, Rank {stock['rank']}")


if __name__ == "__main__":
    # Pinecone API key
    API_KEY = "pcsk_VuT6p_3DZQT6bCBKfPNr63ToDSFqTVRMRfoaBvLaUg9vUps3UEzH6b7Uc4HfuTEm3BZWf"
    
    if not API_KEY:
        print("Please set your PINECONE_API_KEY environment variable")
        print("You can get your API key from: https://app.pinecone.io/")
    else:
        try:
            # Initialize the database connection (no data upload needed)
            print("Connecting to existing Pinecone database...")
            stock_db = StockVectorDatabase(API_KEY)
            
            # Connect to existing index
            if not stock_db.create_index():
                raise Exception("Failed to connect to Pinecone index")
            
            print("Successfully connected to Pinecone!")
            
            # Test the search functions with your existing data
            demo_stock_search(stock_db)
            
        except Exception as e:
            print(f"Error: {str(e)}")
