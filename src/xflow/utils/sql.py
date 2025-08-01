import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .decorator import print_separator

# Registry of supported database types
SUPPORTED_DB_TYPES = {
    "sqlite": "SQLiteDB",
    # Future database types can be added here
    # 'postgresql': 'PostgreSQLDB',
    # 'mysql': 'MySQLDB',
}


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
    def select(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Select records with optional filtering."""
        pass

    @abstractmethod
    def update(
        self, table_name: str, data: Dict[str, Any], where: Dict[str, Any]
    ) -> int:
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

    def select(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Select records with optional filtering."""
        col_str = self._build_columns(columns)
        where_clause, params = self._build_where_clause(where)

        sql = f"SELECT {col_str} FROM {table_name}{where_clause}"
        if limit:
            sql += f" LIMIT {limit}"

        return pd.read_sql_query(sql, self.connection, params=params)

    def update(
        self, table_name: str, data: Dict[str, Any], where: Dict[str, Any]
    ) -> int:
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


def get_supported_db_types() -> List[str]:
    """Get list of supported database types."""
    return list(SUPPORTED_DB_TYPES.keys())


def find_connection(source: Dict[str, Any]) -> str:
    """Find connection string from source dict.

    Priority: explicit keys -> auto-detection from values

    Raises:
        ValueError: If no connection found
    """
    # Try explicit keys first
    explicit_keys = ["connection", "path", "url", "database_path", "db_path"]
    for key in explicit_keys:
        if key in source and source[key]:
            return str(source[key])

    # Fallback: detect from any string value that looks like connection
    for value in source.values():
        if isinstance(value, str) and _looks_like_connection(value):
            return value

    raise ValueError(f"No connection found in source: {source}")


def find_sql(source: Dict[str, Any]) -> str:
    """Find SQL query from source dict.

    Priority: explicit keys -> auto-detection from values

    Raises:
        ValueError: If no SQL query found
    """
    # Try explicit keys first
    explicit_keys = ["sql", "query", "statement", "command"]
    for key in explicit_keys:
        if key in source and source[key]:
            return str(source[key])

    # Fallback: detect from any string value that looks like SQL
    for value in source.values():
        if isinstance(value, str) and _looks_like_sql(value):
            return value

    raise ValueError(f"No SQL query found in source: {source}")


def find_db_type(source: Dict[str, Any], connection: str = None) -> str:
    """Find database type from source dict.

    Priority: explicit keys -> auto-detection from connection -> default

    Args:
        source: Source dictionary
        connection: Optional connection string for auto-detection

    Returns:
        Database type string (never fails, has fallback)

    Raises:
        ValueError: If detected type is not supported
    """
    # Try explicit keys first
    explicit_keys = ["db_type", "database_type", "type", "engine"]
    for key in explicit_keys:
        if key in source and source[key]:
            db_type = str(source[key]).lower()
            if db_type not in SUPPORTED_DB_TYPES:
                raise ValueError(
                    f"Unsupported database type: {db_type}. Supported: {list(SUPPORTED_DB_TYPES.keys())}"
                )
            return db_type

    # Fallback: detect from connection string
    if connection:
        db_type = detect_db_type(connection)
        if db_type not in SUPPORTED_DB_TYPES:
            raise ValueError(
                f"Detected unsupported database type: {db_type}. Supported: {list(SUPPORTED_DB_TYPES.keys())}"
            )
        return db_type

    # Default fallback (always valid)
    return "sqlite"


def _looks_like_connection(value: str) -> bool:
    """Check if string looks like a database connection."""
    value_lower = value.lower()

    # URL schemes (high confidence)
    url_schemes = ["sqlite://", "postgresql://", "postgres://", "mysql://"]
    if any(value_lower.startswith(scheme) for scheme in url_schemes):
        return True

    # File extensions (high confidence)
    file_extensions = [".db", ".sqlite", ".sqlite3", ".mdb"]
    if any(value_lower.endswith(ext) for ext in file_extensions):
        return True

    # Path-like patterns (medium confidence)
    if ("/" in value or "\\" in value) and not _looks_like_sql(value):
        return True

    return False


def _looks_like_sql(value: str) -> bool:
    """Check if string looks like a SQL query."""
    sql_starters = ["select", "insert", "update", "delete", "create", "drop", "alter"]
    value_lower = value.lower().strip()

    # Check if starts with SQL keywords
    return any(value_lower.startswith(keyword) for keyword in sql_starters)


def detect_db_type(connection: str) -> str:
    """Auto-detect database type from connection string.

    Args:
        connection: Connection string (file path, URL, etc.)

    Returns:
        Database type string
    """
    connection_lower = str(connection).lower()

    # URL scheme detection (most reliable)
    if connection_lower.startswith("sqlite://"):
        return "sqlite"
    elif connection_lower.startswith(("postgresql://", "postgres://")):
        return "postgresql"
    elif connection_lower.startswith("mysql://"):
        return "mysql"

    # File extension detection (reliable for file-based DBs)
    elif connection_lower.endswith((".db", ".sqlite", ".sqlite3")):
        return "sqlite"

    # Default fallback
    else:
        return "sqlite"


def create_database_instance(db_type: str, connection: str) -> "Database":
    """Create database instance based on type.

    Args:
        db_type: Database type string
        connection: Connection string

    Returns:
        Database instance
    """
    if db_type == "sqlite":
        return SQLiteDB(connection)
    else:
        # Future database types
        raise NotImplementedError(f"Database type {db_type} not yet implemented")
