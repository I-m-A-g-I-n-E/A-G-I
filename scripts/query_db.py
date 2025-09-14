import sqlite3

conn = sqlite3.connect('outputs/metrics.db')
c = conn.cursor()
c.execute("SELECT * FROM runs")
rows = c.fetchall()
for row in rows:
    print(row)
conn.close()
