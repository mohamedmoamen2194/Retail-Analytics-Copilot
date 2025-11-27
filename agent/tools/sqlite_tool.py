import sqlite3
from typing import Optional, Dict, Any, List, Tuple


class SQLiteTool:
    def __init__(self, db_path: str = "data/northwind.sqlite"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def list_tables(self) -> List[str]:
        self.connect()
        cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row["name"] for row in cursor.fetchall()]

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        self.connect()
        cursor = self.conn.execute(f"PRAGMA table_info('{table_name}');")
        return [dict(row) for row in cursor.fetchall()]

    def execute(self, sql: str) -> Dict[str, Any]:
        self.connect()

        try:
            cursor = self.conn.execute(sql)
            rows = cursor.fetchall()

            rows_as_dict = [dict(row) for row in rows]

            return {
                "success": True,
                "error": None,
                "columns": rows[0].keys() if rows else [],
                "rows": rows_as_dict,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "columns": [],
                "rows": []
            }

    def test_query(self):
        return self.execute("SELECT * FROM Customers LIMIT 3;")
