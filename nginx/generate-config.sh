#!/bin/bash
HOSTNAME=$(hostname -f)
sed "s/YOUR_HOSTNAME/${HOSTNAME}/g" template.web.conf > web.conf

# chmod +x generate-config.sh
# ./generate-config.sh