#!/usr/bin/env python3
"""Embed processed document chunks using a SageMaker text-embedding endpoint
and upload embeddings to an S3 prefix suitable for vector indexing.

This script expects `document_processor.py` to have created JSON files under
an S3 prefix (default `processed/`) where each JSON contains a top-level
`chunks` list of objects with `text` and `metadata` fields.

The script calls a SageMaker endpoint (requires AWS credentials) and uploads
one JSON per chunk to S3 under `vector_prefix` containing `metadata` and
the numeric `embedding`.

Notes:
- The SageMaker endpoint must accept JSON input and return an embedding as
  JSON (either a list of numbers or a dict containing an "embedding" key).
- The script is conservative by default: use `--dry-run` to preview actions.
"""
import argparse
import json
import logging
import random
import time
from typing import Any, Callable, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def retry_with_backoff(fn: Callable[[], Any], max_attempts: int = 4, initial_delay: float = 1.0,
                       backoff_factor: float = 2.0, max_delay: float = 30.0, jitter: float = 0.1):
    attempt = 0
    delay = initial_delay
    while True:
        attempt += 1
        try:
            result = fn()
            # If result is a boto3 Response-like dict, treat it as success
            return result
        except Exception as exc:
            # If the error is a SageMaker ModelError indicating the model
            # rejected the content type, don't retry — this is a configuration
            # issue rather than a transient failure.
            from botocore.exceptions import ClientError
            if hasattr(exc, "response") and isinstance(exc, ClientError):
                code = exc.response.get("Error", {}).get("Code")
                msg = exc.response.get("Error", {}).get("Message", "")
                if code and code.lower().startswith("modelerror") or "unsupported content type" in str(msg).lower():
                    logger.error("Non-retriable model error: %s", exc)
                    raise

            if attempt >= max_attempts:
                raise
            jitter_val = random.uniform(-jitter * delay, jitter * delay)
            sleep_for = min(max_delay, delay + jitter_val)
            logger.warning("Attempt %d failed: %s — retrying in %.1fs", attempt, exc, sleep_for)
            time.sleep(max(0.0, sleep_for))
            delay *= backoff_factor


def parse_sagemaker_response(body_bytes: bytes) -> Any:
    """Try to parse the SageMaker endpoint response into an embedding (list of floats).

    The response body often is JSON; common shapes:
    - a raw JSON array: [0.1, 0.2, ...]
    - a dict containing 'embedding' or 'embeddings' key
    """
    text = body_bytes.decode("utf-8")
    try:
        parsed = json.loads(text)
    except Exception:
        # Not JSON
        raise ValueError("Unparseable response from endpoint: not JSON")

    # If parsed is dict, try to find embedding key
    if isinstance(parsed, dict):
        for key in ("embedding", "embeddings", "vector", "output", "outputs", "predictions"):
            if key in parsed:
                return parsed[key]
        # sometimes predictions is a list of dicts
        if "predictions" in parsed and isinstance(parsed["predictions"], list):
            p = parsed["predictions"]
            if p and isinstance(p[0], list):
                return p[0]
    # If parsed is a list of numbers, return it
    if isinstance(parsed, list) and parsed and all(isinstance(x, (int, float)) for x in parsed):
        return parsed

    raise ValueError("Could not locate embedding in endpoint response")


def get_embedding(endpoint_name: str, text: str, runtime_client) -> List[float]:
    # This client hard-codes the Content-Type to application/x-text
    # because that is the expected value for the target SageMaker endpoint.
    # We still attempt a couple of payload shapes (JSON wrapper then raw
    # text) but send the same content-type header.
    attempts = []

    ct = "application/x-text"
    # 1) JSON wrapper many endpoints sometimes accept
    attempts.append((json.dumps({"inputs": text}).encode("utf-8"), ct))
    # 2) Plain text payload
    attempts.append((text.encode("utf-8"), ct))

    last_exc: Exception | None = None
    for body_payload, content_type in attempts:
        try:
            resp = runtime_client.invoke_endpoint(EndpointName=endpoint_name,
                                                  ContentType=content_type,
                                                  Body=body_payload)
            body_stream = resp.get("Body")
            if body_stream is None:
                raise ValueError("No Body in SageMaker response")
            body = body_stream.read()
            emb = parse_sagemaker_response(body)
            # If emb is a flat list of numbers, return it as floats
            if isinstance(emb, list) and emb and all(isinstance(x, (int, float)) for x in emb):
                return [float(x) for x in emb]

            # If embedding wrapped in another list (e.g., [[...]]), unwrap
            if isinstance(emb, list) and emb and isinstance(emb[0], list):
                inner = emb[0]
                if all(isinstance(x, (int, float)) for x in inner):
                    return [float(x) for x in inner]

            # If parse succeeded but shape unexpected, raise to try other types
            raise ValueError(f"Unexpected embedding shape: {type(emb)}")
        except Exception as exc:
            # If the model explicitly rejected the content type, try the next
            # payload/content-type pair. Otherwise record the last exception
            # and continue (it will be raised after trying all attempts).
            last_exc = exc
            logger.debug("Attempt with Content-Type %s failed: %s", content_type, exc)
            continue

    # If we got here, all attempts failed. Raise the last exception.
    if last_exc:
        raise last_exc

    # If emb is a flat list of numbers, return it as floats
    if isinstance(emb, list) and emb and all(isinstance(x, (int, float)) for x in emb):
        return [float(x) for x in emb]

    # If embedding wrapped in another list (e.g., [[...]]), unwrap
    if isinstance(emb, list) and emb and isinstance(emb[0], list):
        inner = emb[0]
        if all(isinstance(x, (int, float)) for x in inner):
            return [float(x) for x in inner]

    raise ValueError(f"Unexpected embedding shape: {type(emb)}")


def sanitize_key_component(s: str) -> str:
    # Conservative sanitizer to avoid nested prefixes and path traversal
    return s.replace("/", "_").replace(" ", "_").replace("\\", "_")


def process_bucket(bucket: str, processed_prefix: str, vector_prefix: str, endpoint_name: str,
                   chunk_size: int = 512, max_files: int | None = None, dry_run: bool = True,
                   content_type: str | None = None,
                   vector_bucket: str | None = None, vector_index: str | None = None,
                   region_name: str | None = None) -> None:
    # We'll list objects in the main thread, but process each file concurrently
    paginator = boto3.client("s3").get_paginator("list_objects_v2")
    list_kwargs = {"Bucket": bucket, "Prefix": processed_prefix}
    processed_docs = 0

    def process_object(key: str) -> None:
        """Process a single S3 object key: download, embed chunks, and upload vectors."""
        # Create per-thread clients to avoid sharing sockets across threads
        s3_client = boto3.client("s3")
        runtime_client = boto3.client("sagemaker-runtime")
        s3vectors_client = None
        if vector_bucket and vector_index:
            kwargs = {"region_name": region_name} if region_name else {}
            try:
                s3vectors_client = boto3.client("s3vectors", **kwargs)
            except Exception:
                logger.exception("Failed to create s3vectors client for worker; will fall back to regular S3")

        logger.info("Processing processed file: s3://%s/%s", bucket, key)

        try:
            resp = s3_client.get_object(Bucket=bucket, Key=key)
            data = json.loads(resp["Body"].read().decode("utf-8"))
        except (BotoCoreError, ClientError) as e:
            logger.exception("Failed to download %s: %s", key, e)
            return

        chunks = data.get("chunks", [])
        for chunk in chunks:
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            identifier = metadata.get("pmid") or metadata.get("doi") or metadata.get("title") or key
            identifier = sanitize_key_component(str(identifier))
            chunk_id = metadata.get("chunk_id", 0)

            try:
                emb = retry_with_backoff(lambda: get_embedding(endpoint_name, text, runtime_client))
            except Exception as e:
                logger.exception("Embedding failed for %s chunk %s: %s", identifier, chunk_id, e)
                continue

            out_key = f"{vector_prefix.rstrip('/')}/{identifier}_{chunk_id}.json"
            body = json.dumps({"metadata": metadata, "embedding": emb}, ensure_ascii=False)

            try:
                if s3vectors_client and vector_bucket and vector_index:
                    vec_key = f"{identifier}_{chunk_id}"
                    vector_entry = {
                        "key": vec_key,
                        "data": {"float32": emb},
                        "metadata": {"id": vec_key, "source_s3": f"s3://{bucket}/{key}", "chunk_id": chunk_id}
                    }
                    try:
                        s3vectors_client.put_vectors(vectorBucketName=vector_bucket, indexName=vector_index, vectors=[vector_entry])
                        logger.info("Inserted vector %s into index %s/%s", vec_key, vector_bucket, vector_index)
                    except (BotoCoreError, ClientError) as e:
                        logger.exception("Failed to put vector %s into S3 vectors index: %s", vec_key, e)
                else:
                    s3_client.put_object(Bucket=bucket, Key=out_key, Body=body.encode("utf-8"), ContentType="application/json")
                    logger.info("Uploaded embedding to s3://%s/%s", bucket, out_key)
            except (BotoCoreError, ClientError) as e:
                logger.exception("Failed to upload embedding %s: %s", out_key, e)

    # Collect keys to process
    keys: List[str] = []
    for page in paginator.paginate(**list_kwargs):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.lower().endswith(".json"):
                logger.debug("Skipping non-json key: %s", key)
                continue
            keys.append(key)

    if not keys:
        logger.info("No processed json files found under prefix %s", processed_prefix)
        return

    # If dry-run, just list the keys and return
    if max_files:
        keys = keys[:max_files]

    if dry_run:
        for key in keys:
            logger.info("Dry-run: would download and embed chunks from %s", key)
        return

    # Run processing in a ThreadPoolExecutor. Determine worker count: default 4 or based on CPU
    worker_count = min(len(keys), getattr(process_bucket, "_worker_count", 4))
    with ThreadPoolExecutor(max_workers=worker_count) as exc:
        futures = {exc.submit(process_object, k): k for k in keys}
        for fut in as_completed(futures):
            processed_docs += 1
            key = futures[fut]
            try:
                fut.result()
            except Exception as e:
                logger.exception("Error processing %s: %s", key, e)
            if max_files and processed_docs >= max_files:
                logger.info("Reached max_files=%d, stopping", max_files)
                break


def main():
    parser = argparse.ArgumentParser(description="Embed processed document chunks with SageMaker and upload vectors to S3")
    parser.add_argument("--bucket", default="medical-research-agent-documents", help="S3 bucket name")
    parser.add_argument("--processed-prefix", default="processed/", help="S3 prefix where processed JSON files live")
    parser.add_argument("--vector-prefix", default="vectors/", help="S3 prefix to write embeddings to")
    parser.add_argument("--endpoint", required=True, help="SageMaker endpoint name for embeddings")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size used for processing (informational)")
    parser.add_argument("--max-files", type=int, default=10, help="Max number of processed files to handle")
    parser.add_argument("--dry-run", action="store_true", help="Don't call endpoint or upload, just list")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent worker threads to use")
    parser.add_argument("--vector-bucket", default=None, help="S3 Vectors bucket name to insert vectors into (optional)")
    parser.add_argument("--vector-index", default=None, help="S3 Vectors index name to insert vectors into (optional)")
    parser.add_argument("--region", default=None, help="AWS region to use for s3vectors client (optional)")

    args = parser.parse_args()

    # Attach worker count to the function object for simple configuration
    setattr(process_bucket, "_worker_count", max(1, args.workers))
    process_bucket(bucket=args.bucket, processed_prefix=args.processed_prefix, vector_prefix=args.vector_prefix,
                   endpoint_name=args.endpoint, chunk_size=args.chunk_size, max_files=args.max_files,
                   dry_run=args.dry_run, vector_bucket=args.vector_bucket, vector_index=args.vector_index,
                   region_name=args.region)


if __name__ == "__main__":
    main()

# Example usage:
# python3 embed_documents.py --bucket medical-research-agent-documents --processed-prefix processed/ --endpoint jumpstart-dft-robertafin-large-wiki-20251029-230437 --vector-bucket medical-research-agent-vector-bucket --vector-index medical-research-agent-vector-index --region us-east-2 --max-files 1000
