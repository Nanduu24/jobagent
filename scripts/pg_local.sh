#!/usr/bin/env bash
# Local PostgreSQL 16 for JobAgent WITHOUT Docker (Homebrew keg).
#
# Docker is the documented path (docker-compose.yml), but on machines without a
# container runtime this script stands up the same Postgres 16 database that the
# app + Alembic expect. It is the setup that this repo was actually verified on.
#
# Usage:
#   scripts/pg_local.sh start    # initdb (first run), start server, create db
#   scripts/pg_local.sh stop     # stop server
#   scripts/pg_local.sh status   # show server status + version
#   scripts/pg_local.sh psql     # open a psql shell on the jobagent db
#
# Matches the default DATABASE_URL host/port/creds so no .env change is needed:
#   postgresql+asyncpg://jobagent:jobagent@localhost:5433/jobagent
set -euo pipefail

PGBIN="${PGBIN:-/opt/homebrew/opt/postgresql@16/bin}"
PGDATA="${PGDATA:-$HOME/.jobagent/pg16data}"
PGPORT="${PGPORT:-5433}"
# Unix socket dir must be short (<103 bytes) — keep it out of the repo path.
SOCKDIR="${SOCKDIR:-/tmp/jobagent-pg}"
export LC_ALL="${LC_ALL:-en_US.UTF-8}"

mkdir -p "$SOCKDIR"

case "${1:-}" in
  start)
    if [ ! -d "$PGDATA" ]; then
      echo "initdb -> $PGDATA"
      "$PGBIN/initdb" -D "$PGDATA" -U postgres --auth=trust >/dev/null
      { echo "port = $PGPORT"; echo "unix_socket_directories = '$SOCKDIR'"; } >> "$PGDATA/postgresql.conf"
    fi
    "$PGBIN/pg_ctl" -D "$PGDATA" -w start
    # Create role + db if missing (idempotent).
    "$PGBIN/psql" -h localhost -p "$PGPORT" -U postgres -tc \
      "SELECT 1 FROM pg_roles WHERE rolname='jobagent'" | grep -q 1 || \
      "$PGBIN/psql" -h localhost -p "$PGPORT" -U postgres -c \
      "CREATE ROLE jobagent LOGIN PASSWORD 'jobagent';"
    "$PGBIN/psql" -h localhost -p "$PGPORT" -U postgres -tc \
      "SELECT 1 FROM pg_database WHERE datname='jobagent'" | grep -q 1 || \
      "$PGBIN/psql" -h localhost -p "$PGPORT" -U postgres -c \
      "CREATE DATABASE jobagent OWNER jobagent;"
    echo "ready: postgresql+asyncpg://jobagent:jobagent@localhost:$PGPORT/jobagent"
    ;;
  stop)
    "$PGBIN/pg_ctl" -D "$PGDATA" -w stop
    ;;
  status)
    "$PGBIN/pg_ctl" -D "$PGDATA" status || true
    "$PGBIN/psql" -h localhost -p "$PGPORT" -U jobagent -d jobagent -tAc "select version();" 2>/dev/null || true
    ;;
  psql)
    exec "$PGBIN/psql" -h localhost -p "$PGPORT" -U jobagent -d jobagent
    ;;
  *)
    echo "usage: $0 {start|stop|status|psql}" >&2
    exit 1
    ;;
esac
