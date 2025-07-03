import json
import os
from src.main import StockVectorDatabase

def lambda_handler(event, context):
    """
    AWS Lambda handler for stock analysis using Pinecone.
    
    Supported commands:
    1. find_similar_stocks
    2. search_by_criteria  
    3. get_stock_document
    4. get_index_stats
    """
    
    try:
        # Initialize Pinecone connection
        api_key = os.environ.get('PINECONE_API_KEY', 'pcsk_VuT6p_3DZQT6bCBKfPNr63ToDSFqTVRMRfoaBvLaUg9vUps3UEzH6b7Uc4HfuTEm3BZWf')
        stock_db = StockVectorDatabase(api_key)
        
        # Connect to existing index
        if not stock_db.create_index():
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Failed to connect to Pinecone'})
            }
        
        # Parse the command from event
        command = event.get('command', 'get_index_stats')
        
        # Route commands
        if command == 'find_similar_stocks':
            return handle_find_similar_stocks(stock_db, event)
            
        elif command == 'search_by_criteria':
            return handle_search_by_criteria(stock_db, event)
            
        elif command == 'get_stock_document':
            return handle_get_stock_document(stock_db, event)
            
        elif command == 'get_index_stats':
            return handle_get_index_stats(stock_db)
            
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Invalid command',
                    'supported_commands': [
                        'find_similar_stocks',
                        'search_by_criteria', 
                        'get_stock_document',
                        'get_index_stats'
                    ]
                })
            }
            
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def handle_find_similar_stocks(stock_db, event):
    """Find stocks similar to target stock."""
    target_security = event.get('target_security', 'PTT')
    target_date = event.get('target_date')  # Optional: YYYY-MM-DD
    top_k = event.get('top_k', 5)
    
    results = stock_db.find_similar_stocks(target_security, target_date, top_k)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'command': 'find_similar_stocks',
            'target_security': target_security,
            'similar_stocks': results
        })
    }

def handle_search_by_criteria(stock_db, event):
    """Search stocks by specific criteria."""
    criteria = event.get('criteria', {})
    top_k = event.get('top_k', 10)
    
    # Example criteria:
    # {
    #   "sector": "Technology",
    #   "min_total_score": 80,
    #   "max_pe_ratio": 25
    # }
    
    results = stock_db.search_by_criteria(criteria, top_k)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'command': 'search_by_criteria',
            'criteria': criteria,
            'results': results
        })
    }

def handle_get_stock_document(stock_db, event):
    """Get detailed document for a specific stock."""
    target_security = event.get('target_security', 'PTT')
    target_date = event.get('target_date')  # Optional
    
    document = stock_db.get_stock_document(target_security, target_date)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'command': 'get_stock_document',
            'target_security': target_security,
            'document': document
        })
    }

def handle_get_index_stats(stock_db):
    """Get Pinecone index statistics."""
    stats = stock_db.get_index_stats()
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'command': 'get_index_stats',
            'stats': stats
        })
    } 