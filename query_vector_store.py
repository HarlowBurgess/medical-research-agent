#!/usr/bin/env python3
"""Query an S3 Vector Index using embeddings from a SageMaker endpoint.

This script:
- Calls a SageMaker endpoint (runtime.invoke_endpoint) to get an embedding for an input query text.
- Calls the S3Vectors API `query_vectors` to perform a similarity search against an S3 Vector Index.

Example:
python3 query_vector_store.py --vector-bucket my-vector-bucket --vector-index my-vector-index --endpoint my-sagemaker-endpoint --query "find papers about covid vaccines" --region us-east-2 --top-k 5
"""
import argparse
import json
import logging
from typing import Any, Dict, List

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_sagemaker_response(body_bytes: bytes) -> Any:
    text = body_bytes.decode("utf-8")
    parsed = json.loads(text)
    # Try to locate embedding in common shapes
    if isinstance(parsed, dict):
        for key in ("embedding", "embeddings", "vector", "output", "outputs", "predictions"):
            if key in parsed:
                return parsed[key]
        if "predictions" in parsed and isinstance(parsed["predictions"], list):
            p = parsed["predictions"]
            if p and isinstance(p[0], list):
                return p[0]
    if isinstance(parsed, list) and parsed and all(isinstance(x, (int, float)) for x in parsed):
        return parsed
    raise ValueError("Could not locate embedding in SageMaker response")


def get_embedding_from_sagemaker(endpoint_name: str, text: str, runtime_client) -> List[float]:
    # Try JSON wrapper then raw text; mirror the caller script's approach
    attempts = [json.dumps({"inputs": text}).encode("utf-8"), text.encode("utf-8")]
    ct = "application/x-text"
    last_exc = None
    for body in attempts:
        try:
            resp = runtime_client.invoke_endpoint(EndpointName=endpoint_name, ContentType=ct, Body=body)
            body_stream = resp.get("Body")
            if body_stream is None:
                raise ValueError("No Body in SageMaker response")
            body = body_stream.read()
            emb = parse_sagemaker_response(body)
            # unwrap if necessary
            if isinstance(emb, list) and emb and all(isinstance(x, (int, float)) for x in emb):
                return [float(x) for x in emb]
            if isinstance(emb, list) and emb and isinstance(emb[0], list):
                inner = emb[0]
                if all(isinstance(x, (int, float)) for x in inner):
                    return [float(x) for x in inner]
            raise ValueError(f"Unexpected embedding shape: {type(emb)}")
        except Exception as exc:
            last_exc = exc
            logger.debug("Attempt failed: %s", exc)
            continue
    if last_exc:
        raise last_exc
    raise ValueError("Failed to obtain embedding from SageMaker")


def query_vectors_index(s3vectors_client, vector_bucket: str, vector_index: str, query_embedding: List[float], top_k: int = 5, filter_expr: Dict[str, Any] | None = None) -> Dict[str, Any]:
    # Build query payload
    query_vector = {"float32": query_embedding}
    kwargs = {
        "vectorBucketName": vector_bucket,
        "indexName": vector_index,
        "queryVector": query_vector,
        "topK": top_k,
        "returnDistance": True,
        "returnMetadata": True,
    }
    if filter_expr:
        kwargs["filter"] = filter_expr
    return s3vectors_client.query_vectors(**kwargs)


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    p = urlparse(s3_uri)
    if p.scheme != "s3":
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    bucket = p.netloc
    key = p.path.lstrip("/")
    return bucket, key


def fetch_document_chunks(s3_client, s3_uri: str, cache: Dict[str, Any]) -> list[dict]:
    """Fetch a JSON document from S3 and return its 'chunks' list. Uses cache to avoid repeated downloads."""
    if s3_uri in cache:
        return cache[s3_uri]
    bucket, key = parse_s3_uri(s3_uri)
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        body = resp["Body"].read()
        doc = json.loads(body.decode("utf-8"))
        chunks = doc.get("chunks") or []
        cache[s3_uri] = chunks
        return chunks
    except Exception as e:
        logger.exception("Failed to fetch or parse document %s: %s", s3_uri, e)
        cache[s3_uri] = []
        return []


def main():
    parser = argparse.ArgumentParser(description="Query S3 Vector Index using SageMaker endpoint embeddings")
    parser.add_argument("--vector-bucket", required=True, help="S3 Vectors bucket name")
    parser.add_argument("--vector-index", required=True, help="S3 Vectors index name")
    parser.add_argument("--endpoint", required=True, help="SageMaker endpoint name to create query embedding")
    parser.add_argument("--query", required=True, help="Text query to embed and search")
    parser.add_argument("--region", default=None, help="AWS region for s3vectors and runtime client (optional)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest neighbors to return")
    parser.add_argument("--filter", default=None, help="Optional JSON filter to apply to query (e.g. '{\"genre\": \"scifi\"}')")
    parser.add_argument("--dry-run", action="store_true", help="Don't call AWS, just print what would be done")
    parser.add_argument("--output-mode", choices=["metadata", "chunk", "document"], default="chunk", help="What to print for each match: metadata (original vector metadata), chunk (the matched chunk text), or document (full concatenated document text)")

    args = parser.parse_args()

    region_kwargs = {"region_name": args.region} if args.region else {}

    if args.dry_run:
        logger.info("Dry-run: would create clients and run query")
        logger.info("SageMaker endpoint: %s", args.endpoint)
        logger.info("Vector bucket/index: %s / %s", args.vector_bucket, args.vector_index)
        logger.info("Query text: %s", args.query)
        return

    # Create clients
    runtime = boto3.client("sagemaker-runtime", **region_kwargs)
    try:
        s3vectors = boto3.client("s3vectors", **region_kwargs)
    except Exception:
        logger.exception("Failed to create s3vectors client")
        raise

    # Get embedding
    try:
        embedding = get_embedding_from_sagemaker(args.endpoint, args.query, runtime)
    except Exception as e:
        logger.exception("Failed to get embedding from SageMaker endpoint: %s", e)
        raise

    # Parse filter JSON if provided
    filter_obj = None
    if args.filter:
        try:
            filter_obj = json.loads(args.filter)
        except Exception as e:
            logger.error("Invalid filter JSON: %s", e)
            raise

    # Query the index
    try:
        resp = query_vectors_index(s3vectors, args.vector_bucket, args.vector_index, embedding, top_k=args.top_k, filter_expr=filter_obj)
    except (BotoCoreError, ClientError) as e:
        logger.exception("Failed to query vectors index: %s", e)
        raise

    vectors = resp.get("vectors") or []

    # Create S3 client to fetch original documents
    s3 = boto3.client("s3", **region_kwargs)

    # Cache documents already fetched
    doc_cache: Dict[str, Any] = {}

    output_mode = args.output_mode

    output = []
    for item in vectors:
        metadata = item.get("metadata") or {}
        s3_uri = metadata.get("source_s3")
        chunk_id = metadata.get("chunk_id")

        entry = {
            "key": item.get("key"),
            "distance": item.get("distance"),
            "chunk_text": None,
            "source_s3": s3_uri,
            "chunk_id": chunk_id,
        }

        # Handle modes:
        # - metadata: return original vector metadata only
        # - chunk: fetch document and return matched chunk text
        # - document: fetch document and concatenate all chunk texts
        if output_mode == "metadata":
            # Keep only metadata and distance
            out = {
                "key": entry["key"],
                "distance": entry["distance"],
                "metadata": metadata,
            }
            output.append(out)
            continue

        if s3_uri is None:
            entry["chunk_text"] = None
            entry.setdefault("note", "no source_s3 in metadata")
            output.append(entry)
            continue

        chunks = fetch_document_chunks(s3, s3_uri, doc_cache)

        if output_mode == "document":
            # concatenate chunk texts into a single document text
            all_texts = [t for t in (c.get("text") for c in chunks) if isinstance(t, str) and t]
            entry["document_text"] = "\n\n".join(all_texts) if all_texts else None
            if entry["document_text"] is None:
                entry.setdefault("note", "no chunks or failed to fetch document")
            output.append(entry)
            continue

        # output_mode == "chunk"
        # Try to find the chunk by chunk_id. chunk_id may be int or string.
        found_text = None
        if chunks:
            for c in chunks:
                # chunk metadata may contain chunk_id or the chunk dict may have 'chunk_id' at top-level
                c_meta = c.get("metadata") or {}
                c_chunk_id = c_meta.get("chunk_id") if isinstance(c_meta, dict) else None
                if c_chunk_id is None:
                    c_chunk_id = c.get("chunk_id")
                # Compare as int or str
                try:
                    # Ensure both ids are present before numeric comparison
                    if c_chunk_id is not None and chunk_id is not None and int(c_chunk_id) == int(chunk_id):
                        found_text = c.get("text")
                        break
                except Exception:
                    # fallback to string comparison
                    if str(c_chunk_id) == str(chunk_id):
                        found_text = c.get("text")
                        break

        if found_text is None:
            entry["chunk_text"] = None
            entry["note"] = "chunk not found"
        else:
            entry["chunk_text"] = found_text

        output.append(entry)

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

# Example usage:
# python3 query_vector_store.py --vector-bucket medical-research-agent-vector-bucket --vector-index medical-research-agent-vector-index --endpoint jumpstart-dft-robertafin-large-wiki-20251029-050135 --query "Does psoriasis impair left ventricular (LV) systolic function" --region us-east-2 --top-k 5
