
class Database:
    """
    A class used to manage SQLite database operations.
    
    Attributes:
    ----------
    connection : sqlite3.Connection
        An SQLite connection object to the database.
    cursor : sqlite3.Cursor
        A cursor object to execute SQL queries.
    """
    
    def __init__(self, db_path=':memory:'):
        """
        Initializes the Database class with a connection to the specified SQLite database.
        
        Parameters:
        ----------
        db_path : str
            The path to the SQLite database file. Default is in-memory database.
        """
        self.db_path = db_path
        self.connection = None

    def connect(self):
        self.connection = sqlite3.connect(self.db_path)
        return self.connection

    def close(self):
        if self.connection:
            self.connection.close()class Database:
    def __init__(self, db_path=':memory:'):
        self.db_path = db_path
        self.connection = None
        self.cursor = None

    def connect(self):
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        return self.connection

    def initialize_metrics_table(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY,
            uptime INTEGER,
            total_requests INTEGER,
            total_processing_time INTEGER,
            labeled_data_points INTEGER,
            active_agents TEXT
        )
        """)
        # Insert default metrics if the table is empty
        self.cursor.execute("INSERT INTO metrics (uptime, total_requests, total_processing_time, labeled_data_points, active_agents) SELECT 0,0,0,0,'engineer,analyst' WHERE NOT EXISTS (SELECT 1 FROM metrics)")
        self.connection.commit()

    def get_metrics(self):
        self.cursor.execute("SELECT * FROM metrics ORDER BY id DESC LIMIT 1")
        return self.cursor.fetchone()

    def update_metrics(self, **kwargs):
        # Sample update operation, can be extended based on specific requirements
        for key, value in kwargs.items():
            self.cursor.execute(f"UPDATE metrics SET {key} = {key} + ? WHERE id = 1", (value,))
        self.connection.commit()

    def close(self):
        if self.connection:
            self.connection.close()
