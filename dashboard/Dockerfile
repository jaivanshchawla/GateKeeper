FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY dashboard/package*.json ./

# Install dependencies
RUN npm install

# Copy app source
COPY dashboard/ .

# Expose port
EXPOSE 3000

# Start the app
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]
