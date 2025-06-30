# Dependencies stage - separate stage just for installing dependencies
FROM node:16-slim as deps
WORKDIR /app
COPY package.json .npmrc ./

# Install dependencies with optimized flags
RUN npm install --legacy-peer-deps --force --no-audit --no-fund --loglevel=error && \
    # Fix specific dependency versions if needed
    npm install ajv@8.12.0 ajv-keywords@5.1.0 && \
    npm cache clean --force

# Build stage - separate from dependencies to optimize layer caching
FROM node:16-slim as build
WORKDIR /app

# Copy dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules
COPY package.json ./

# Copy only necessary files for building
COPY public/ ./public/
COPY src/ ./src/
COPY nginx.conf ./nginx.conf
COPY tsconfig.json ./

# Build the app with memory optimization
# Disable source maps and use conservative memory limits
RUN npm run build

# Production stage - using nginx as the web server
FROM nginx:alpine

# Copy built files and nginx configuration
COPY --from=build /app/build /usr/share/nginx/html
COPY --from=build /app/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
