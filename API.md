# API Reference

All time values use the format `YYYYMMDDHH` (UTC).

---

## `GET /status`

Returns the current server status: what is actively being processed, what is queued, and when the next scheduled maintenance run is.

**Response**

```json
{
  "status": "working",
  "next_maintenance": "2026-04-12T00:30:00Z",
  "active": ["2026041212"],
  "queued": ["2026041300", "2026041306"]
}
```

| Field | Description |
|-------|-------------|
| `status` | `"idle"` when nothing is queued or active, otherwise `"working"` |
| `next_maintenance` | ISO 8601 UTC datetime of the next scheduled purge + auto-build run |
| `active` | Target IDs (`YYYYMMDDHH`) currently being processed |
| `queued` | Target IDs (`YYYYMMDDHH`) waiting in the queue, sorted ascending |

---

## `GET /available`

Returns all ready tile sets currently on disk, sorted by time.

**Response**

```json
{
  "items": [
    {
      "id": "2026040812",
      "status": "ready",
      "run": "2026040809",
      "step": 3,
      "path": "/2026040809_003/"
    },
    ...
  ]
}
```

| Field | Description |
|-------|-------------|
| `id` | Target hour (`YYYYMMDDHH`) |
| `status` | Always `ready` |
| `run` | Model run used (`YYYYMMDDHH`) |
| `step` | Forecast step (hours from run to target) |
| `path` | URL prefix for tile and shadow files |

---

## `GET /{folder}/tiles/{z}/{x}/{y}.ktx2`

Serves a cloud tile in KTX2 format. `{folder}` is the value of the `path` field returned by `/available` (e.g. `2026040809_003`).

## `GET /{folder}/shadow.ktx2`

Serves the shadow map for the given tile set.

All other paths return `403 Forbidden`.
