from typing import Dict, List, Optional
import sqlite3
import json
from datetime import datetime
import pandas as pd

class DatabaseHandler:
    def __init__(self, db_path: str = "trading_data.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.initialize_database()
        
    def initialize_database(self):
        """Create necessary database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                size REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                entry_time TIMESTAMP NOT NULL,
                exit_time TIMESTAMP,
                pnl REAL,
                status TEXT NOT NULL,
                strategy TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        # Create market data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                timestamp TIMESTAMP NOT NULL,
                symbol TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (timestamp, symbol)
            )
        """)
        
        # Create signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                timestamp TIMESTAMP NOT NULL,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                signal REAL NOT NULL,
                strength REAL NOT NULL,
                metadata TEXT,
                PRIMARY KEY (timestamp, symbol, strategy)
            )
        """)
        
        conn.commit()
        conn.close()
        
    def store_trade(self, trade_data: Dict):
        """Store trade information in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata = json.dumps(trade_data.get('metadata', {}))
        
        cursor.execute("""
            INSERT INTO trades (
                id, symbol, side, size, entry_price, exit_price,
                entry_time, exit_time, pnl, status, strategy, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_data['id'],
            trade_data['symbol'],
            trade_data['side'],
            trade_data['size'],
            trade_data['entry_price'],
            trade_data.get('exit_price'),
            trade_data['entry_time'],
            trade_data.get('exit_time'),
            trade_data.get('pnl'),
            trade_data['status'],
            trade_data['strategy'],
            metadata
        ))
        
        conn.commit()
        conn.close()
        
    def update_trade(self, trade_id: str, update_data: Dict):
        """Update existing trade record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        update_fields = []
        update_values = []
        
        for key, value in update_data.items():
            if key in ['exit_price', 'exit_time', 'pnl', 'status']:
                update_fields.append(f"{key} = ?")
                update_values.append(value)
                
        if update_fields:
            query = f"""
                UPDATE trades
                SET {', '.join(update_fields)}
                WHERE id = ?
            """
            update_values.append(trade_id)
            
            cursor.execute(query, update_values)
            conn.commit()
            
        conn.close()
        
    def store_market_data(self, symbol: str, data: pd.DataFrame):
        """Store market data in database."""
        conn = sqlite3.connect(self.db_path)
        
        # Convert DataFrame to database format
        data_to_store = data.reset_index()
        data_to_store['symbol'] = symbol
        
        data_to_store.to_sql(
            'market_data',
            conn,
            if_exists='append',
            index=False
        )
        
        conn.close()
        
    def store_signal(self, signal_data: Dict):
        """Store strategy signal in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata = json.dumps(signal_data.get('metadata', {}))
        
        cursor.execute("""
            INSERT INTO signals (
                timestamp, symbol, strategy, signal,
                strength, metadata
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            signal_data['timestamp'],
            signal_data['symbol'],
            signal_data['strategy'],
            signal_data['signal'],
            signal_data['strength'],
            metadata
        ))
        
        conn.commit()
        conn.close()
        
    def get_trades(self,
                   symbol: Optional[str] = None,
                   status: Optional[str] = None,
                   strategy: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[Dict]:
        """Retrieve trades based on filters."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
            
        if status:
            query += " AND status = ?"
            params.append(status)
            
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
            
        if start_time:
            query += " AND entry_time >= ?"
            params.append(start_time)
            
        if end_time:
            query += " AND entry_time <= ?"
            params.append(end_time)
            
        cursor.execute(query, params)
        columns = [description[0] for description in cursor.description]
        trades = []
        
        for row in cursor.fetchall():
            trade = dict(zip(columns, row))
            if trade['metadata']:
                trade['metadata'] = json.loads(trade['metadata'])
            trades.append(trade)
            
        conn.close()
        return trades
        
    def get_market_data(self,
                       symbol: str,
                       start_time: datetime,
                       end_time: datetime) -> pd.DataFrame:
        """Retrieve market data for analysis."""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT * FROM market_data
            WHERE symbol = ?
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """
        
        df = pd.read_sql_query(
            query,
            conn,
            params=(symbol, start_time, end_time)
        )
        
        conn.close()
        return df
        
    def get_signals(self,
                    symbol: Optional[str] = None,
                    strategy: Optional[str] = None,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None) -> List[Dict]:
        """Retrieve strategy signals based on filters."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM signals WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
            
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
            
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
            
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
            
        cursor.execute(query, params)
        columns = [description[0] for description in cursor.description]
        signals = []
        
        for row in cursor.fetchall():
            signal = dict(zip(columns, row))
            if signal['metadata']:
                signal['metadata'] = json.loads(signal['metadata'])
            signals.append(signal)
            
        conn.close()
        return signals 