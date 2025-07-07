import sqlite3
import pandas as pd
from abc import ABC, abstractmethod
from .decorator import print_separator
from typing import Dict, Any, List, Optional, Union
from contextlib import contextmanager


class Database(ABC):
    """Abstract base class for database operations using Query Builder Pattern."""
    
    def __init__(self):
        self.connection = None
    
    # Table operations
    @abstractmethod
    def create_table(self, table_name: str, schema: Dict[str, str]) -> None:
        """Create a new table with given schema."""
        pass
    
    @abstractmethod
    def drop_table(self, table_name: str) -> None:
        """Drop an existing table."""
        pass
    
    @abstractmethod
    def list_tables(self) -> List[str]:
        """List all tables in the database."""
        pass
    
    # Schema operations
    @abstractmethod
    def add_column(self, table_name: str, column_name: str, data_type: str) -> None:
        """Add a new column to existing table."""
        pass
    
    # CRUD operations
    @abstractmethod
    def insert(self, table_name: str, data: Dict[str, Any]) -> None:
        """Insert a single record."""
        pass
    
    @abstractmethod
    def insert_many(self, table_name: str, data: List[Dict[str, Any]]) -> None:
        """Insert multiple records."""
        pass
    
    @abstractmethod
    def select(self, table_name: str, columns: Optional[List[str]] = None, 
               where: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """Select records with optional filtering."""
        pass
    
    @abstractmethod
    def update(self, table_name: str, data: Dict[str, Any], where: Dict[str, Any]) -> int:
        """Update records matching where clause. Returns number of affected rows."""
        pass
    
    @abstractmethod
    def delete(self, table_name: str, where: Dict[str, Any]) -> int:
        """Delete records matching where clause. Returns number of affected rows."""
        pass
    
    # Raw SQL operations
    @abstractmethod
    def execute(self, sql: str, params: Optional[List[Any]] = None) -> None:
        """Execute raw SQL command."""
        pass
    
    @abstractmethod
    def query(self, sql: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """Execute raw SQL query and return DataFrame."""
        pass
    
    # Connection management
    @abstractmethod
    def close(self) -> None:
        """Close database connection."""
        pass


class SQLiteDB(Database):
    @print_separator()
    def __init__(self, db_path: str):
        """Initialize SQLite database connection."""
        super().__init__()
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self.tables = self.list_tables()
        print(f"Connected to SQLite database with {len(self.tables)} tables")
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
    
    def _build_where_clause(self, where: Dict[str, Any]) -> tuple:
        """Build WHERE clause with parameters."""
        if not where:
            return "", ()
        
        conditions = []
        params = []
        for key, value in where.items():
            conditions.append(f"{key} = ?")
            params.append(value)
        
        return f" WHERE {' AND '.join(conditions)}", tuple(params)
    
    def _build_columns(self, columns: Optional[List[str]]) -> str:
        """Build column list for SELECT."""
        return ", ".join(columns) if columns else "*"
    
    # Table operations
    def create_table(self, table_name: str, schema: Dict[str, str]) -> None:
        """Create a new table with given schema."""
        columns = ", ".join(f"{col} {dtype}" for col, dtype in schema.items())
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
        self.cursor.execute(sql)
        self.connection.commit()
        self.tables = self.list_tables()  # Refresh table list
    
    def drop_table(self, table_name: str) -> None:
        """Drop an existing table."""
        sql = f"DROP TABLE IF EXISTS {table_name}"
        self.cursor.execute(sql)
        self.connection.commit()
        self.tables = self.list_tables()  # Refresh table list
    
    def list_tables(self) -> List[str]:
        """List all tables in the database."""
        sql = "SELECT name FROM sqlite_master WHERE type='table'"
        self.cursor.execute(sql)
        return [row[0] for row in self.cursor.fetchall()]
    
    # Schema operations
    def add_column(self, table_name: str, column_name: str, data_type: str) -> None:
        """Add a new column to existing table."""
        sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {data_type}"
        self.cursor.execute(sql)
        self.connection.commit()
    
    # CRUD operations
    def insert(self, table_name: str, data: Dict[str, Any]) -> None:
        """Insert a single record."""
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        self.cursor.execute(sql, tuple(data.values()))
        self.connection.commit()
    
    def insert_many(self, table_name: str, data: List[Dict[str, Any]]) -> None:
        """Insert multiple records."""
        if not data:
            return
        
        columns = ", ".join(data[0].keys())
        placeholders = ", ".join("?" * len(data[0]))
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        
        values = [tuple(row.values()) for row in data]
        self.cursor.executemany(sql, values)
        self.connection.commit()
    
    def select(self, table_name: str, columns: Optional[List[str]] = None, 
               where: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """Select records with optional filtering."""
        col_str = self._build_columns(columns)
        where_clause, params = self._build_where_clause(where)
        
        sql = f"SELECT {col_str} FROM {table_name}{where_clause}"
        if limit:
            sql += f" LIMIT {limit}"
        
        return pd.read_sql_query(sql, self.connection, params=params)
    
    def update(self, table_name: str, data: Dict[str, Any], where: Dict[str, Any]) -> int:
        """Update records matching where clause. Returns number of affected rows."""
        set_clause = ", ".join(f"{col} = ?" for col in data.keys())
        where_clause, where_params = self._build_where_clause(where)
        
        sql = f"UPDATE {table_name} SET {set_clause}{where_clause}"
        params = tuple(data.values()) + where_params
        
        self.cursor.execute(sql, params)
        self.connection.commit()
        return self.cursor.rowcount
    
    def delete(self, table_name: str, where: Dict[str, Any]) -> int:
        """Delete records matching where clause. Returns number of affected rows."""
        where_clause, params = self._build_where_clause(where)
        sql = f"DELETE FROM {table_name}{where_clause}"
        
        self.cursor.execute(sql, params)
        self.connection.commit()
        return self.cursor.rowcount
    
    # Raw SQL operations
    def execute(self, sql: str, params: Optional[List[Any]] = None) -> None:
        """Execute raw SQL command."""
        self.cursor.execute(sql, params or [])
        self.connection.commit()
    
    def query(self, sql: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """Execute raw SQL query and return DataFrame."""
        return pd.read_sql_query(sql, self.connection, params=params)
    
    # Connection management
    def close(self) -> None:
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()