import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


RUNPOD_API_BASE = "https://api.runpod.ai/v2"


def _request_json(method: str, url: str, token: str, payload: dict | None = None) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc}") from exc


def submit_job(endpoint_id: str, api_key: str, test_input: dict) -> str:
    url = f"{RUNPOD_API_BASE}/{endpoint_id}/run"
    result = _request_json("POST", url, api_key, payload=test_input)
    job_id = result.get("id")
    if not job_id:
        raise RuntimeError(f"Missing job id in run response: {result}")
    return job_id


def poll_job(
    endpoint_id: str,
    api_key: str,
    job_id: str,
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> dict:
    deadline = time.time() + timeout_seconds
    status_url = f"{RUNPOD_API_BASE}/{endpoint_id}/status/{job_id}"

    while time.time() < deadline:
        status = _request_json("GET", status_url, api_key)
        state = status.get("status")
        print(f"[quality-check] job={job_id} status={state}", flush=True)

        if state == "COMPLETED":
            return status
        if state in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            raise RuntimeError(f"RunPod job finished with failure state: {state}, payload={status}")

        time.sleep(poll_interval_seconds)

    raise TimeoutError(f"Quality check timed out after {timeout_seconds}s waiting for job {job_id}")


def validate_output(status_payload: dict) -> None:
    output = status_payload.get("output")
    if not isinstance(output, dict):
        raise RuntimeError(f"Invalid output payload: {status_payload}")

    image_url = output.get("image_url")
    images = output.get("images")
    seed = output.get("seed")

    if not image_url:
        raise RuntimeError(f"Missing output.image_url: {output}")
    if not isinstance(images, list) or len(images) == 0:
        raise RuntimeError(f"Missing or empty output.images: {output}")
    if seed is None:
        raise RuntimeError(f"Missing output.seed: {output}")


def load_input_payload(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "input" not in payload or not isinstance(payload["input"], dict):
        raise RuntimeError(f"Input payload must contain an object field 'input': {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RunPod non-blocking quality sanity check")
    parser.add_argument(
        "--test-input",
        default="quality_test_input.json",
        help="Path to a JSON payload shaped as {'input': {...}}",
    )
    parser.add_argument("--timeout-seconds", type=int, default=600, help="Polling timeout in seconds")
    parser.add_argument("--poll-interval-seconds", type=int, default=5, help="Polling interval in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
    if not api_key:
        print("RUNPOD_API_KEY is required", file=sys.stderr)
        return 2
    if not endpoint_id:
        print("RUNPOD_ENDPOINT_ID is required", file=sys.stderr)
        return 2

    try:
        payload = load_input_payload(args.test_input)
        job_id = submit_job(endpoint_id, api_key, payload)
        print(f"[quality-check] submitted job id: {job_id}", flush=True)

        final_status = poll_job(
            endpoint_id=endpoint_id,
            api_key=api_key,
            job_id=job_id,
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        validate_output(final_status)
        print("[quality-check] COMPLETED and output schema validated", flush=True)
        return 0
    except Exception as exc:
        print(f"[quality-check] FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
