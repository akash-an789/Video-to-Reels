import psycopg2
import os


# Database connection
def create_connection():
    conn = psycopg2.connect(
            host="localhost",
            database="Userdata",
            user="postgres",
            password="1905",
            port=5432
        )
    return conn


def create_tables():
    try:
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(50),
                email VARCHAR(100) UNIQUE,
                password VARCHAR(100),
                phone_no VARCHAR(15),
                dob DATE,
                profession VARCHAR(100),
                gender VARCHAR(10),
                profile_picture BYTEA
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS clips (
                user_id INT NOT NULL,
                file_name VARCHAR(255) NOT NULL,
                no_of_clips INT NOT NULL,
                duration NUMERIC(10, 2) NOT NULL,  -- Stores duration with 2 decimal places
                PRIMARY KEY (user_id, file_name)
            );
        """
        )

        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(e)
        return e
