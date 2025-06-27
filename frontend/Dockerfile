# Build stage
FROM node:16-slim as build

# Set working directory
WORKDIR /app

# Copy package.json and .npmrc
COPY package.json .npmrc ./

# Install dependencies with optimized flags
RUN npm install --legacy-peer-deps --force --no-audit --no-fund --loglevel=error && \
    # Fix specific dependency versions if needed
    npm install ajv@8.12.0 ajv-keywords@5.1.0 && \
    npm cache clean --force

# Copy only necessary files for building
COPY public/ ./public/
COPY src/ ./src/
COPY tsconfig.json ./

# Build the app with increased memory limit and production optimization
RUN NODE_ENV=production NODE_OPTIONS=--max_old_space_size=4096 npm run build && \
    rm -rf node_modules

# Production stage - using Alpine for smaller image size
FROM nginx:alpine

# Copy built files from build stage to nginx
COPY --from=build /app/build /usr/share/nginx/html

# Copy custom nginx config for React routing and API proxying
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Remove default nginx website to reduce image size
RUN rm -rf /usr/share/nginx/html/50x.html

# Cache configuration for better performance
RUN echo 'fastcgi_cache_path /var/cache/nginx levels=1:2 keys_zone=REACTAPP:10m inactive=60m;' > /etc/nginx/conf.d/cache.conf

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
