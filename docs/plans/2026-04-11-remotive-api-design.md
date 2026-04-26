# Remotive API Integration — Design

## Goal

Add Remotive API (https://remotive.com/api/remote-jobs) as an alternative job source alongside the existing local FAISS-based dataset. A config parameter controls which source is used.

## Approach

**Option 1 (chosen):** Use Remotive's API search params for initial retrieval, skip embed/vector-search entirely in API mode. The matching agent + critic agent handle fine-grained scoring, making local vector search redundant when the API already filters.

## Config

New settings in `.env` / `config/settings.py`:

- `JOB_SOURCE` — `local` (default) or `remotive`
- `REMOTIVE_SEARCH` — search query string (e.g. `"machine learning engineer"`), user-provided for now
- `REMOTIVE_LIMIT` — max jobs to fetch from API (default `20`)

Future: LLM-extracted keywords can replace user-provided search terms. The architecture supports this — the search term just needs to come from somewhere before the API call.

## New module: `tools/remotive_client.py`

Stateless functions (consistent with existing tools pattern):

- `fetch_remotive_jobs(search, limit) -> list[dict]` — GET request to Remotive API
- `normalize_remotive_job(job) -> dict` — field mapping to internal format:
  - `title` → `positionName`
  - `company_name` → `company`
  - `candidate_required_location` → `location`
  - `salary` → `salary`
  - `description` → HTML stripped to plain text (regex, no new dependency)
  - `id` → `id` (as string)
  - `url` → `url` (preserved for report)

## Graph changes

New node `fetch_remote_jobs` in `graph/nodes.py`:
- Calls remotive client, normalizes results
- Returns `{"retrieved_jobs": normalized_jobs}` — same state key as local path

Conditional routing in `graph/graph_builder.py`:
- `JOB_SOURCE == "remotive"`: `parse_cv` → `fetch_remote_jobs` → `matching_agent`
- `JOB_SOURCE == "local"`: `parse_cv` → `embed_cv` → `retrieve_jobs` → `matching_agent`

Everything from `matching_agent` onward is unchanged.

## Testing

- Mock `requests.get` for Remotive client tests
- Test field normalization and HTML stripping
- Test graph routing for both source modes
