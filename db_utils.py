##### db_utils.py #####
import sqlite3
import datetime

db_file = "token_usage.db"

def create_table():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            api_name TEXT,
            prompt TEXT,
            response TEXT,
            response_time_ms REAL,
            token_count INTEGER
        )
    """)
    conn.commit()
    conn.close()

def log_data_to_arctic(api_name, prompt, response, response_time, token_count):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO api_logs (timestamp, api_name, prompt, response, response_time_ms, token_count) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, (datetime.datetime.utcnow().isoformat(), api_name, prompt, response, response_time, token_count))
    conn.commit()
    conn.close()
