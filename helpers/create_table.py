import os
import psycopg2
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

load_dotenv()
# get the postgres db connection parameters from environment variable
name = os.getenv("DATABASE_NAME")
user = os.getenv("DATABASE_USERNAME")
password = os.getenv("DATABASE_PASSWORD")
host = os.getenv("DATABASE_HOST")
port = os.getenv("DATABASE_PORT")

# connect an instance of postgres DB
con = psycopg2.connect(
    database=name, user=user, password=password, host=host, port=port
)
print("Database opened successfully")

cur = con.cursor()

# create a table "Taskdata" in db "task" in our postgres
cur.execute('''CREATE TABLE News
      (id SERIAL PRIMARY KEY,
      input_text VARCHAR(2500) NOT NULL,
      text_length VARCHAR(250) NOT NULL,
      genuine BOOLEAN,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      updated_at TIMESTAMPTZ DEFAULT NOW());''')
print("Table created successfully")
con.commit()
con.close()  # commit and close connection
