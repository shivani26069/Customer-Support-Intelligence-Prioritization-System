import sqlite3

conn = sqlite3.connect("support_cases.db")
cursor = conn.cursor()

cursor.execute("ALTER TABLE cases ADD COLUMN message_id TEXT")
cursor.execute("ALTER TABLE cases ADD COLUMN sender TEXT")
cursor.execute("ALTER TABLE cases ADD COLUMN received_at TEXT")

conn.commit()
conn.close()

print("Database updated successfully!")