#!/usr/bin/env python3
"""Ensure Open WebUI's SQLite DB has our local backends registered.

Open WebUI reads OPENAI_API_BASE_URLS only on first run (seed_defaults skips
keys that already exist in the DB). After that, the DB takes precedence, so
editing the UI or this script is the only way to change connections.

This script is idempotent: it upserts the connection rows every time the webui
starts, so the backend list is declarative — defined here + the Makefile, not
hand-clicked in the UI.

Usage:
    python scripts/webui-config.py <db_path> <url;key;url;key...>
    python scripts/webui-config.py coach-history/webui.db \
        "http://localhost:8081/v1;local;http://localhost:8082/v1;local"
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path


def upsert(db: sqlite3.Connection, key: str, value: object) -> None:
	"""Insert or replace a config row."""
	import time
	db.execute(
		"INSERT INTO config (key, value, updated_at) VALUES (?, ?, ?) "
		"ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
		(key, json.dumps(value), int(time.time())),
	)


def main() -> None:
	if len(sys.argv) != 3:
		print(__doc__)
		sys.exit(1)

	db_path = Path(sys.argv[1])
	pairs_str = sys.argv[2]

	# Parse "url1;key1;url2;key2" into [(url, key), ...]
	parts = [p.strip() for p in pairs_str.split(";") if p.strip()]
	if len(parts) % 2 != 0:
		print(f"error: expected url;key pairs, got {len(parts)} parts", file=sys.stderr)
		sys.exit(1)
	pairs = list(zip(parts[0::2], parts[1::2]))
	urls = [url for url, _ in pairs]
	keys = [key for _, key in pairs]

	# Build per-connection configs — all enabled, external, bearer auth.
	configs = {
		str(i): {
			"enable": True,
			"tags": [],
			"prefix_id": "",
			"model_ids": [],
			"connection_type": "external",
			"auth_type": "bearer",
			"passthrough_params": [],
		}
		for i in range(len(urls))
	}

	db_path.parent.mkdir(parents=True, exist_ok=True)
	conn = sqlite3.connect(str(db_path))

	# Create the config table if this is a fresh DB (open-webui will also
	# create it via alembic, but upserting first avoids a race on first run).
	conn.execute(
		"CREATE TABLE IF NOT EXISTS config ("
		'"key" TEXT NOT NULL, value JSON NOT NULL, updated_at BIGINT, '
		'PRIMARY KEY ("key"))'
	)

	upsert(conn, "openai.enable", True)
	upsert(conn, "openai.api_base_urls", urls)
	upsert(conn, "openai.api_keys", keys)
	upsert(conn, "openai.api_configs", configs)

	conn.commit()
	conn.close()

	print(f"webui-config: registered {len(urls)} backend(s) in {db_path}")
	for url, key in pairs:
		print(f"  {url}  (key: {key})")


if __name__ == "__main__":
	main()
