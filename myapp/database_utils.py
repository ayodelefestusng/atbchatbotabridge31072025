"""
Database utilities for using QuerySQLDatabaseTool to generate DataFrame results
"""

import pandas as pd
import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from typing import Optional, Union, Dict, Any

class DatabaseQueryTool:
    """
    A wrapper class for QuerySQLDatabaseTool that provides DataFrame output
    """
    
    def __init__(self, db_uri: str = "sqlite:///db.sqlite3"):
        """
        Initialize the database query tool
        
        Args:
            db_uri (str): Database URI (default: SQLite database)
        """
        self.db_uri = db_uri
        self.db = SQLDatabase.from_uri(db_uri)
        self.query_tool = QuerySQLDataBaseTool(db=self.db)
    
    def query_to_dataframe(self, sql_query: str) -> Optional[pd.DataFrame]:
        """
        Execute SQL query and return result as DataFrame
        
        Args:
            sql_query (str): SQL query to execute
            
        Returns:
            pandas.DataFrame or None: Query result as DataFrame, None if error
        """
        try:
            # Execute query using QuerySQLDatabaseTool
            result = self.query_tool.run(sql_query)
            
            # Try to convert result to DataFrame
            df = self._parse_result_to_dataframe(result)
            
            if df is None:
                # Fallback to direct pandas query
                df = self._direct_pandas_query(sql_query)
            
            return df
            
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
    
    def _parse_result_to_dataframe(self, result: str) -> Optional[pd.DataFrame]:
        """
        Parse the string result from QuerySQLDatabaseTool to DataFrame
        
        Args:
            result (str): Raw result string from QuerySQLDatabaseTool
            
        Returns:
            pandas.DataFrame or None: Parsed DataFrame
        """
        if not isinstance(result, str):
            return None
        
        # Method 1: Try CSV parsing
        try:
            from io import StringIO
            df = pd.read_csv(StringIO(result))
            return df
        except:
            pass
        
        # Method 2: Parse tabular format (pipe-separated)
        try:
            lines = result.strip().split('\n')
            if len(lines) > 1:
                # Extract headers
                headers = [h.strip() for h in lines[0].split('|') if h.strip()]
                
                # Extract data rows
                data = []
                for line in lines[1:]:
                    if line.strip() and '|' in line:
                        row = [cell.strip() for cell in line.split('|') if cell.strip()]
                        if len(row) == len(headers):
                            data.append(row)
                
                if data:
                    df = pd.DataFrame(data, columns=headers)
                    return df
        except:
            pass
        
        return None
    
    def _direct_pandas_query(self, sql_query: str) -> Optional[pd.DataFrame]:
        """
        Execute query directly using pandas (fallback method)
        
        Args:
            sql_query (str): SQL query to execute
            
        Returns:
            pandas.DataFrame or None: Query result
        """
        try:
            if self.db_uri.startswith("sqlite:///"):
                db_path = self.db_uri.replace("sqlite:///", "")
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query(sql_query, conn)
                conn.close()
                return df
        except Exception as e:
            print(f"Direct pandas query failed: {e}")
        
        return None
    
    def get_table_info(self) -> Dict[str, Any]:
        """
        Get information about database tables
        
        Returns:
            dict: Table information
        """
        try:
            return self.db.get_table_info()
        except Exception as e:
            print(f"Error getting table info: {e}")
            return {}
    
    def get_table_names(self) -> list:
        """
        Get list of table names in the database
        
        Returns:
            list: Table names
        """
        try:
            return self.db.get_table_names()
        except Exception as e:
            print(f"Error getting table names: {e}")
            return []

# Convenience function for quick queries
def quick_query(sql_query: str, db_uri: str = "sqlite:///db.sqlite3") -> Optional[pd.DataFrame]:
    """
    Quick function to execute a SQL query and get DataFrame result
    
    Args:
        sql_query (str): SQL query to execute
        db_uri (str): Database URI
        
    Returns:
        pandas.DataFrame or None: Query result
    """
    tool = DatabaseQueryTool(db_uri)
    return tool.query_to_dataframe(sql_query)

# Example usage functions
def example_queries():
    """Run example queries to demonstrate usage"""
    
    # Initialize the tool
    db_tool = DatabaseQueryTool()
    
    print("=== Database Tables ===")
    tables = db_tool.get_table_names()
    print(f"Available tables: {tables}")
    
    print("\n=== Table Information ===")
    table_info = db_tool.get_table_info()
    for table_name, info in table_info.items():
        print(f"\nTable: {table_name}")
        print(f"Columns: {info}")
    
    # Example queries
    print("\n=== Example Queries ===")
    
    # Query 1: Get all users
    print("\n1. Getting all users:")
    df_users = db_tool.query_to_dataframe("SELECT * FROM myapp_user LIMIT 5")
    if df_users is not None:
        print(df_users)
        print(f"Shape: {df_users.shape}")
    else:
        print("No data returned")
    
    # Query 2: Count users
    print("\n2. Counting users:")
    df_count = db_tool.query_to_dataframe("SELECT COUNT(*) as user_count FROM myapp_user")
    if df_count is not None:
        print(df_count)
    else:
        print("No data returned")
    
    # Query 3: User statistics
    print("\n3. User statistics:")
    df_stats = db_tool.query_to_dataframe("""
        SELECT 
            COUNT(*) as total_users,
            COUNT(CASE WHEN is_active = 1 THEN 1 END) as active_users,
            COUNT(CASE WHEN is_staff = 1 THEN 1 END) as staff_users
        FROM myapp_user
    """)
    if df_stats is not None:
        print(df_stats)
    else:
        print("No data returned")

if __name__ == "__main__":
    # Run examples
    example_queries()
    
    # Show how to use the convenience function
    print("\n=== Using Quick Query Function ===")
    df = quick_query("SELECT COUNT(*) as count FROM myapp_user")
    if df is not None:
        print(df) 