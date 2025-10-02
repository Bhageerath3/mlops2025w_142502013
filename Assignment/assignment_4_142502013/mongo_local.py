from pymongo import MongoClient, errors
import pandas as pd
import sqlite3
import time
from pprint import pprint


def get_connection(uri):
    try:
        client = MongoClient(uri, maxPoolSize=50, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        print("MongoDB local connected\n")
        return client
    except errors.ServerSelectionTimeoutError as err:
        print("MongoDB local connection failed:\n", err)
        return None


def load_from_sqlite():
    conn = sqlite3.connect("online_retail.db")
    query = """
    SELECT i.InvoiceNo, i.InvoiceDate, c.CustomerID, c.Country,
           p.StockCode, p.Description, ii.Quantity, p.UnitPrice
    FROM Invoice i
    JOIN Customer c ON i.CustomerID = c.CustomerID
    JOIN InvoiceItem ii ON i.InvoiceNo = ii.InvoiceNo
    JOIN Product p ON ii.StockCode = p.StockCode
    LIMIT 1000;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df = df.astype(object).where(pd.notnull(df), None) 
    return df


def timed_operation(name, func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    print(f"{name} | Time taken: {end - start:.8f} sec")
    return result, end - start

def insert_transaction_centric(db, df):
    invoices = []
    for inv, group in df.groupby("InvoiceNo"):
        invoice_doc = {
            "InvoiceNo": str(inv),
            "InvoiceDate": str(group.iloc[0]["InvoiceDate"]),
            "CustomerID": int(group.iloc[0]["CustomerID"]),
            "Country": str(group.iloc[0]["Country"]),
            "Items": group[["StockCode", "Description", "Quantity", "UnitPrice"]].to_dict("records")
        }
        invoices.append(invoice_doc)
    db.transactions.insert_many(invoices)
    print("Inserted Transaction-Centric documents")


def insert_customer_centric(db, df):
    customers = []
    for cust, group in df.groupby("CustomerID"):
        cust_doc = {
            "CustomerID": int(cust),
            "Country": str(group.iloc[0]["Country"]),
            "Invoices": []
        }
        for inv, inv_group in group.groupby("InvoiceNo"):
            invoice_doc = {
                "InvoiceNo": str(inv),
                "InvoiceDate": str(inv_group.iloc[0]["InvoiceDate"]),
                "Items": inv_group[["StockCode", "Description", "Quantity", "UnitPrice"]].to_dict("records")
            }
            cust_doc["Invoices"].append(invoice_doc)
        customers.append(cust_doc)
    db.customers.insert_many(customers)
    print("Inserted Customer-Centric documents")


def transaction_crud(db):
    print("\nTransaction-Centric CRUD:")
    print()
    new_invoice = {
        "InvoiceNo": "152648",
        "InvoiceDate": "2025-10-02",
        "CustomerID": 54816,
        "Country": "India",
        "Items": [{"StockCode": "ABC123", "Description": "Test Product", "Quantity": 2, "UnitPrice": 11}]
    }
    insert_result, _ = timed_operation("Created new invoice", db.transactions.insert_one, new_invoice)

    print()

    read_invoice, _ = timed_operation("Read invoice", db.transactions.find_one, {"_id": insert_result.inserted_id})
    pprint(read_invoice)

    print()

    _, _ = timed_operation("Added new item to invoice",
                           db.transactions.update_one,
                           {"_id": insert_result.inserted_id},
                           {"$push": {"Items": {"StockCode": "NEW123",
                                                "Description": "New Test Product",
                                                "Quantity": 3,
                                                "UnitPrice": 12}}})

    print()

    _, _ = timed_operation("Deleted test invoice", db.transactions.delete_one, {"_id": insert_result.inserted_id})

def customer_crud(db):
    print("\nCustomer-Centric CRUD:")
    print()
    new_customer = {
        "CustomerID": 99999,
        "Country": "India",
        "Invoices": [
            {"InvoiceNo": "999999", "InvoiceDate": "2025-10-02",
             "Items": [{"StockCode": "XYZ456", "Description": "Another Product", "Quantity": 1, "UnitPrice": 25}]}
        ]
    }
    cust_insert_result, _ = timed_operation("Created new customer with invoice", db.customers.insert_one, new_customer)

    print()
    
    read_customer, _ = timed_operation("Read customer", db.customers.find_one, {"_id": cust_insert_result.inserted_id})
    pprint(read_customer["Invoices"][0])

    
    print()
    new_invoice = {
        "InvoiceNo": "888888", "InvoiceDate": "2025-10-03",
        "Items": [{"StockCode": "LMN789", "Description": "Extra Product", "Quantity": 3, "UnitPrice": 15}]
    }
    _, _ = timed_operation("Added new invoice to existing customer",
                           db.customers.update_one,
                           {"_id": cust_insert_result.inserted_id},
                           {"$push": {"Invoices": new_invoice}})

    print()
    
    _, _ = timed_operation("Deleted test customer", db.customers.delete_one, {"_id": cust_insert_result.inserted_id})

    print()


if __name__ == "__main__":
    
    uri = "mongodb://localhost:27017"
    client = get_connection(uri)
    if client:
        db = client["online_retail"]
        df = load_from_sqlite()


        db.transactions.drop()
        db.customers.drop()

        insert_transaction_centric(db, df)
        insert_customer_centric(db, df)

        print("Transaction docs:", db.transactions.count_documents({}))
        print("Customer docs:", db.customers.count_documents({}))


        transaction_crud(db)
        customer_crud(db)

        client.close()
