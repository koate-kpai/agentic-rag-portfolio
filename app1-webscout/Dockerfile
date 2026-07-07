# Dockerfile for app1-webscout
#
# ARCHITECTURAL DECISION: Multi-stage build without libgomp1
# ------------------------------------------------------------
# scikit-learn relies on libgomp1 (GNU OpenMP runtime) for parallel
# computation.  app1-webscout does NOT import scikit-learn — its only
# domain-specific dependency is duckduckgo-search.  The libgomp1 install
# was erroneously carried over from app2-documind's Dockerfile via copy-paste.
#
# Removing it:
#   - Saves ~30 MB from the final image (reducing pull time and attack surface).
#   - Eliminates an unnecessary apt-get layer (faster builds).
#   - Removes a misleading comment about "missing system lib for scikit-learn".

# Stage 1: Base — shared environment variables
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Stage 2: Dependencies — install Python packages in isolation
FROM base AS dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 3: Final image — application code only, no OS-level packages needed
FROM base AS final

COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY src/ ./src/

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]