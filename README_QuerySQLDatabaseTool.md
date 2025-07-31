# Using QuerySQLDatabaseTool to Generate DataFrame Results

This guide shows you how to use `QuerySQLDatabaseTool` from LangChain to execute SQL queries and get the results as pandas DataFrames.

## Files Created

1. **`myapp/query_database_example.py`** - Comprehensive example with sample database setup
2. **`myapp/simple_query_example.py`** - Simple, focused example
3. **`myapp/database_utils.py`** - Reusable utility class and functions
4. **`myapp/views_with_dataframe.py`** - Django integration examples

## Quick Start

### Method 1: Simple Function (Recommended for beginners)

```python
from myapp.database_utils import quick_query

# Execute a simple query
df = quick_query("SELECT * FROM myapp_user LIMIT 5")
if df is not None:
    print(df)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
```

### Method 2: Using the DatabaseQueryTool Class

```python
from myapp.database_utils import DatabaseQueryTool

# Initialize the tool
db_tool = DatabaseQueryTool()

# Execute queries
df_users = db_tool.query_to_dataframe("SELECT * FROM myapp_user")
df_count = db_tool.query_to_dataframe("SELECT COUNT(*) as user_count FROM myapp_user")

# Get database information
tables = db_tool.get_table_names()
table_info = db_tool.get_table_info()
```

### Method 3: Direct Implementation

```python
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

def execute_sql_to_dataframe(sql_query, db_uri="sqlite:///db.sqlite3"):
    # Initialize database connection
    db = SQLDatabase.from_uri(db_uri)
    
    # Create the QuerySQLDatabaseTool
    execute_query_tool = QuerySQLDataBaseTool(db=db)
    
    # Execute the query
    result = execute_query_tool.run(sql_query)
    
    # Convert result to DataFrame (multiple fallback methods)
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(result))
        return df
    except:
        # Fallback to direct pandas query
        if db_uri.startswith("sqlite:///"):
            db_path = db_uri.replace("sqlite:///", "")
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            return df
    
    return None

# Usage
df = execute_sql_to_dataframe("SELECT * FROM myapp_user")
```

## Django Integration

### API Endpoint for SQL Queries

```python
# In your Django views
@csrf_exempt
@require_http_methods(["POST"])
def execute_sql_query(request):
    data = json.loads(request.body)
    sql_query = data.get('query')
    
    df = db_tool.query_to_dataframe(sql_query)
    
    if df is not None:
        result = {
            'success': True,
            'data': df.to_dict('records'),
            'columns': list(df.columns),
            'shape': df.shape
        }
    else:
        result = {'success': False, 'error': 'Query failed'}
    
    return JsonResponse(result)
```

### Dashboard View Example

```python
def dashboard_view(request):
    queries = {
        'user_count': "SELECT COUNT(*) as total_users FROM myapp_user",
        'active_users': "SELECT COUNT(*) as active_users FROM myapp_user WHERE is_active = 1",
        'recent_users': "SELECT username, date_joined FROM myapp_user ORDER BY date_joined DESC LIMIT 10"
    }
    
    dashboard_data = {}
    for key, query in queries.items():
        df = db_tool.query_to_dataframe(query)
        if df is not None:
            dashboard_data[key] = df.to_dict('records')
    
    return render(request, 'myapp/dashboard.html', {'data': dashboard_data})
```

## Example Queries

### Basic Queries
```python
# Get all users
df = quick_query("SELECT * FROM myapp_user")

# Count users
df = quick_query("SELECT COUNT(*) as user_count FROM myapp_user")

# Get recent users
df = quick_query("SELECT username, date_joined FROM myapp_user ORDER BY date_joined DESC LIMIT 10")
```

### Complex Queries
```python
# User statistics
df = quick_query("""
    SELECT 
        COUNT(*) as total_users,
        COUNT(CASE WHEN is_active = 1 THEN 1 END) as active_users,
        COUNT(CASE WHEN is_staff = 1 THEN 1 END) as staff_users
    FROM myapp_user
""")

# Search users
df = quick_query("""
    SELECT username, email, date_joined, is_active
    FROM myapp_user 
    WHERE username LIKE '%admin%' 
       OR email LIKE '%@example.com%'
    ORDER BY date_joined DESC
""")
```

## DataFrame Operations

Once you have a DataFrame, you can perform all standard pandas operations:

```python
df = quick_query("SELECT * FROM myapp_user")

if df is not None:
    # Basic operations
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types: {df.dtypes}")
    
    # Filtering
    active_users = df[df['is_active'] == True]
    staff_users = df[df['is_staff'] == True]
    
    # Grouping and aggregation
    if 'date_joined' in df.columns:
        df['date_joined'] = pd.to_datetime(df['date_joined'])
        users_by_month = df.groupby(df['date_joined'].dt.to_period('M')).size()
    
    # Export to CSV
    df.to_csv('user_export.csv', index=False)
    
    # Convert to JSON
    json_data = df.to_dict('records')
```

## Error Handling

The utility functions include error handling:

```python
df = quick_query("SELECT * FROM non_existent_table")
if df is None:
    print("Query failed or returned no data")

# The functions will print error messages automatically
```

## Database Support

The code supports:
- **SQLite** (default): `sqlite:///db.sqlite3`
- **PostgreSQL**: `postgresql://user:password@host:port/database`
- **MySQL**: `mysql://user:password@host:port/database`

## Installation Requirements

Make sure you have these packages installed:

```bash
pip install pandas langchain langchain-community sqlalchemy
```

For PostgreSQL support:
```bash
pip install psycopg2-binary
```

For MySQL support:
```bash
pip install pymysql
```

## Running the Examples

1. **Simple Example:**
   ```bash
   python myapp/simple_query_example.py
   ```

2. **Comprehensive Example:**
   ```bash
   python myapp/query_database_example.py
   ```

3. **Utility Functions:**
   ```bash
   python myapp/database_utils.py
   ```

## Key Features

- ✅ **Multiple parsing methods** for different result formats
- ✅ **Fallback to direct pandas queries** for reliability
- ✅ **Error handling** with informative messages
- ✅ **Django integration** examples
- ✅ **Utility functions** for common operations
- ✅ **Type hints** for better code completion
- ✅ **Comprehensive examples** for different use cases

## Troubleshooting

### Common Issues

1. **"No module named 'langchain'"**
   - Install: `pip install langchain langchain-community`

2. **"Database connection failed"**
   - Check your database URI
   - Ensure the database file exists (for SQLite)
   - Verify database credentials (for PostgreSQL/MySQL)

3. **"Query returned no data"**
   - Check if the table exists
   - Verify the SQL syntax
   - Ensure the table has data

4. **"DataFrame parsing failed"**
   - The code includes multiple fallback methods
   - Check the raw result format
   - Use the direct pandas query method as fallback

### Debug Mode

Enable debug output by modifying the utility functions:

```python
# In database_utils.py, add print statements
print(f"Raw result: {result}")
print(f"Result type: {type(result)}")
```

This will help you understand what format the QuerySQLDatabaseTool returns and how it's being parsed. 