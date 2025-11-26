#!/bin/bash

# Check if SUB_DOMAIN is provided
if [ -z "$1" ]; then
    echo "Error: Sub-domain not provided. Please run the script with a sub-domain, e.g., 'bash script.sh my-app'"
    exit 1
fi

# --- Configuration ---
SUB_DOMAIN=uxx999 # Replace with your actual sub-domain
YOUR_DOMAIN="$SUB_DOMAIN.dev.openconsultinguk.com" # Replace with your actual domain
YOUR_EMAIL="uxx999@OpenConsultingUK.com" # Replace with your email for urgent notices
CERT_STORAGE_PATH="ssl/$YOUR_DOMAIN" # Replace with a local path to store certificates
# ---------------------


# Create necessary directory if it doesn't exist
mkdir -p "$CERT_STORAGE_PATH"
# Create necessary directories
mkdir -p "$CERT_STORAGE_PATH/conf"
mkdir -p "ssl/www"
mkdir -p "ssl/log"

echo "Attempting to obtain certificate for $YOUR_DOMAIN using standalone method..."
echo "Ensure port 80 on your server is free and accessible from the internet."

docker run \
  --rm \
  --user "$(id -u):$(id -g)" \
  -p 80:80 \
  -v "$PWD/$CERT_STORAGE_PATH:/etc/letsencrypt" \
  -v "$PWD/ssl/www:/var/www/certbot" \
  -v "$PWD/ssl/log:/var/log/letsencrypt" \
  certbot/certbot \
  certonly \
  --standalone \
  --agree-tos \
  --non-interactive \
  -d "$YOUR_DOMAIN" \
  --email "$YOUR_EMAIL" \
  --config-dir /etc/letsencrypt \
  --work-dir /var/www/certbot \
  --logs-dir /var/log/letsencrypt

# Check the exit status of the docker command
if [ $? -eq 0 ]; then
  echo "Certificate obtained successfully!"
  echo "Certificates stored in: $CERT_STORAGE_PATH/live/$YOUR_DOMAIN/"
else
  echo "Failed to obtain certificate."
fi


# chmod +x get_ssl_certificate.sh
# ./get_ssl_certificate.sh