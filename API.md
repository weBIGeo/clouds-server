# API Reference

All time values use the format `YYYYMMDDHH` (UTC).

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
