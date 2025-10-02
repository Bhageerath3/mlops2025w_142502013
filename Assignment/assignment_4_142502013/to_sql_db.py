import pandas as pd
import sqlite3

# Load 1000 rows from CSV
df = pd.read_csv("online_retail_1000.csv", encoding="ISO-8859-1", nrows=1000)

# Connect SQLite DB
conn = sqlite3.connect("online_retail.db")
cur = conn.cursor()

# Create schema
cur.executescript("""
CREATE TABLE IF NOT EXISTS Customer (
    CustomerID INT PRIMARY KEY,
    Country TEXT
);
CREATE TABLE IF NOT EXISTS Product (
    StockCode TEXT PRIMARY KEY,
    Description TEXT,
    UnitPrice REAL
);
CREATE TABLE IF NOT EXISTS Invoice (
    InvoiceNo TEXT PRIMARY KEY,
    InvoiceDate TEXT,
    CustomerID INT,
    FOREIGN KEY (CustomerID) REFERENCES Customer(CustomerID)
);
CREATE TABLE IF NOT EXISTS InvoiceItem (
    InvoiceNo TEXT,
    StockCode TEXT,
    Quantity INT,
    PRIMARY KEY (InvoiceNo, StockCode),
    FOREIGN KEY (InvoiceNo) REFERENCES Invoice(InvoiceNo),
    FOREIGN KEY (StockCode) REFERENCES Product(StockCode)
);
""")

# Insert customers
for row in df[['CustomerID','Country']].dropna().drop_duplicates().itertuples(index=False):
    cur.execute("INSERT OR IGNORE INTO Customer VALUES (?, ?)", (row.CustomerID, row.Country))

# Insert products
for row in df[['StockCode','Description','UnitPrice']].drop_duplicates().itertuples(index=False):
    cur.execute("INSERT OR IGNORE INTO Product VALUES (?, ?, ?)", (row.StockCode, row.Description, row.UnitPrice))

# Insert invoices
for row in df[['InvoiceNo','InvoiceDate','CustomerID']].drop_duplicates().itertuples(index=False):
    cur.execute("INSERT OR IGNORE INTO Invoice VALUES (?, ?, ?)", (row.InvoiceNo, row.InvoiceDate, row.CustomerID))

# Insert invoice items
for row in df[['InvoiceNo','StockCode','Quantity']].itertuples(index=False):
    cur.execute("INSERT OR REPLACE INTO InvoiceItem VALUES (?, ?, ?)", (row.InvoiceNo, row.StockCode, row.Quantity))

conn.commit()
conn.close()

print("Inserted 1000 records into normalized SQLite DB (online_retail.db)")
