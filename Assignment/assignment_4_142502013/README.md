The script "load_data.py" takes data from original excel file and creates a small online_retail_1000.csv which only contains 1000 records.

The file "to_sql_db.py" takes data from online_retail_1000.csv and makes a local sqlite database.

The script "mongo_local.py" connects to local mongodb server and adds documents to it from the sqlite database, it also performs CRUD operations on this data and gives time taken for each CRUD operation.

The script "mongo_atlas.py" connects to mongodb atlas server and adds documents to it from the sqlite database, it also performs CRUD operations on this data and gives time taken for each CRUD operation.

