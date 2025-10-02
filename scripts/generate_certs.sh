#!/bin/sh
# QFLARE Certificate Generation Script

set -e

CERT_DIR="/certs"
CA_KEY="$CERT_DIR/ca-key.pem"
CA_CERT="$CERT_DIR/ca-cert.pem"
SERVER_KEY="$CERT_DIR/server-key.pem"
SERVER_CERT="$CERT_DIR/server-cert.pem"

echo "Generating QFLARE certificates..."

# Create certificates directory
mkdir -p "$CERT_DIR"

# Generate CA private key
echo "Generating CA private key..."
openssl genrsa -out "$CA_KEY" ${CA_KEY_SIZE:-4096}

# Generate CA certificate
echo "Generating CA certificate..."
openssl req -new -x509 -days ${CERT_VALIDITY_DAYS:-365} -key "$CA_KEY" -out "$CA_CERT" \
    -subj "/C=US/ST=CA/L=San Francisco/O=QFLARE/CN=QFLARE CA"

# Generate server private key
echo "Generating server private key..."
openssl genrsa -out "$SERVER_KEY" ${SERVER_KEY_SIZE:-2048}

# Generate server certificate signing request
echo "Generating server certificate..."
openssl req -new -key "$SERVER_KEY" -out "$CERT_DIR/server.csr" \
    -subj "/C=US/ST=CA/L=San Francisco/O=QFLARE/CN=qflare-server"

# Generate server certificate signed by CA
openssl x509 -req -days ${CERT_VALIDITY_DAYS:-365} -in "$CERT_DIR/server.csr" \
    -CA "$CA_CERT" -CAkey "$CA_KEY" -CAcreateserial -out "$SERVER_CERT"

# Clean up CSR
rm "$CERT_DIR/server.csr"

# Set appropriate permissions
chmod 600 "$CA_KEY" "$SERVER_KEY"
chmod 644 "$CA_CERT" "$SERVER_CERT"

echo "Certificates generated successfully in $CERT_DIR"
echo "CA Certificate: $CA_CERT"
echo "Server Certificate: $SERVER_CERT"
echo "Server Private Key: $SERVER_KEY"