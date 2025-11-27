DATA_DIR="data"
mkdir -p "${DATA_DIR}"

curl -L -o "${DATA_DIR}/northwind.sqlite" \
     "https://raw.githubusercontent.com/microsoft/sql-server-samples/main/samples/databases/northwind-pubs/northwind.db"
