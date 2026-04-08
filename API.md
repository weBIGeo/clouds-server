# API Reference

All time values use the format `YYYYMMDDHH` (UTC).

---

## `GET /status?time=YYYYMMDDHH`

Returns the status of a single time slot.

**Query parameters**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `time` | yes | Target hour in `YYYYMMDDHH` format |

**Response fields**

| Field | Description |
|-------|-------------|
| `id` | The requested time (`YYYYMMDDHH`) |
| `status` | `ready`, `stale`, `pending`, `unknown`, or `error` |
| `run` | Model run used (`YYYYMMDDHH`) |
| `step` | Forecast step (hours from run to target) |
| `path` | Relative URL prefix for tile files (only when `status` is `ready` or `stale`) |
| `progress` | Object with `stage`, `detail`, `percent` (only when `status` is `pending`) |

**Status values**

| Value | Meaning |
|-------|---------|
| `ready` | Data is on disk and uses the best available model run |
| `stale` | Data is on disk but a newer model run is available |
| `pending` | Generation is currently queued or running |
| `unknown` | No data on disk and not currently generating |
| `error` | Time is outside the available window or no forecast exists |

---

## `POST /generate?time=YYYYMMDDHH`

Queues tile generation for the given target time. Returns immediately; poll `/status` to track progress.

**Query parameters**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `time` | yes | Target hour in `YYYYMMDDHH` format |

**Responses**

| Code | Meaning |
|------|---------|
| `202` | Generation queued (or already in progress) |
| `400` | Missing or invalid `time`, or no forecast available |

---

## `GET /available`

Lists all time slots in the window `[now − history_window, now + max_forecast_step]`, plus any additional ready slots found on disk.

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

Each item has the same fields as the `/status` response.

---

## `GET /{folder}/tiles/{z}/{x}/{y}.ktx2`
## `GET /{folder}/tiles/{z}/{x}/{y}.sdf.ktx2`

Serves a cloud tile in KTX2 format. `{folder}` is the `run_step` string returned in the `path` field of a status response (e.g. `2026040809_003`).

## `GET /{folder}/shadow.ktx2`

Serves the shadow map for the given tile set.

All other paths return `403 Forbidden`.
