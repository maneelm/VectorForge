# syntax=docker/dockerfile:1
FROM node:24-bookworm-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY index.html tsconfig.json vite.config.ts ./
COPY src ./src
ENV VITE_API_BASE=/api/v1
RUN npm run build

FROM nginxinc/nginx-unprivileged:stable-alpine AS runtime
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/templates/default.conf.template
COPY docker-entrypoint.d/05-access-policy.envsh /docker-entrypoint.d/05-access-policy.envsh
USER root
RUN chmod 0555 /docker-entrypoint.d/05-access-policy.envsh
USER 101:101
ENV VECTORFORGE_API_PORT=8000 VECTORFORGE_ACCESS_MODE=local VECTORFORGE_BIND_ADDRESS=127.0.0.1 \
    NGINX_ENVSUBST_FILTER="^VECTORFORGE_(API_PORT|AUTH_REALM)$"
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=4 \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1
