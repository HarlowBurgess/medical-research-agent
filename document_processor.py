# document_processor.py
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MedicalPaperProcessor:
    """Process medical papers into structured chunks"""
    
    def parse_pmc_xml(self, xml_content: str) -> Dict:
        """Extract structured content from PMC XML"""
        root = ET.fromstring(xml_content)
        
        metadata = {
            "pmid": self._get_text(root, ".//article-id[@pub-id-type='pmid']"),
            "doi": self._get_text(root, ".//article-id[@pub-id-type='doi']"),
            "title": self._get_text(root, ".//article-title"),
            "pub_date": self._extract_date(root),
        }
        
        # abstract may contain nested <p> elements; join their text
        abstract_elem = root.find(".//abstract")
        if abstract_elem is not None:
            abstract = " ".join(abstract_elem.itertext()).strip()
        else:
            abstract = self._get_text(root, ".//abstract")
        sections = self._extract_sections(root)
        
        return {
            "metadata": metadata,
            "abstract": abstract,
            "sections": sections,
        }
    
    def chunk_document(self, doc: Dict, chunk_size: int = 512) -> List[Dict]:
        """Create semantic chunks with overlap"""
        chunks = []
        
        # Abstract as first chunk
        chunks.append({
            "text": doc.get("abstract", ""),
            "metadata": {**doc.get("metadata", {}), "section": "abstract", "chunk_id": 0}
        })
        
        # Process sections
        chunk_id = 1
        for section_name, section_text in doc.get("sections", {}).items():
            paragraphs = section_text.split("\n\n")
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) < chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "metadata": {
                                **doc.get("metadata", {}),
                                "section": section_name,
                                "chunk_id": chunk_id
                            }
                        })
                        chunk_id += 1
                    current_chunk = para + "\n\n"
            
            if current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": {**doc.get("metadata", {}), "section": section_name, "chunk_id": chunk_id}
                })
                chunk_id += 1
        
        return chunks
    
    def _get_text(self, element, xpath: str) -> str:
        result = element.find(xpath)
        return result.text.strip() if result is not None and result.text else ""
    
    def _extract_date(self, root) -> str:
        pub_date = root.find(".//pub-date")
        if pub_date is not None:
            year = self._get_text(pub_date, "./year")
            month = self._get_text(pub_date, "./month") or "01"
            day = self._get_text(pub_date, "./day") or "01"
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return ""
    
    def _extract_sections(self, root) -> Dict:
        sections = {}
        for section in root.findall(".//sec"):
            title = self._get_text(section, "./title") or "unnamed"
            content = " ".join(section.itertext()).strip()
            sections[title.lower()] = content
        return sections


def process_s3_bucket(bucket: str = "medical-research-agent-documents",
                      input_prefix: str = "raw/",
                      output_prefix: str = "processed/",
                      chunk_size: int = 512,
                      max_files: Optional[int] = None,
                      dry_run: bool = True) -> None:
    """List XML files in S3 under input_prefix, process them, and upload JSON to output_prefix.

    This will not delete or modify the original objects. When dry_run is True the
    S3 download/upload calls are logged but not executed.
    """
    s3 = None
    paginator = None
    BotoCoreError = Exception
    ClientError = Exception

    if dry_run:
        # In dry-run mode we try to import boto3 to list objects, but it's optional.
        try:
            import boto3  # type: ignore
            from botocore.exceptions import BotoCoreError as _BotoCoreError, ClientError as _ClientError
            BotoCoreError = _BotoCoreError
            ClientError = _ClientError
            s3 = boto3.client("s3")
            paginator = s3.get_paginator("list_objects_v2")
        except ImportError:
            logger.info("Dry-run and boto3 not available; skipping S3 listing. To run a real dry-run listing, install boto3 or run without --dry-run when boto3 is installed.")
            return
    else:
        try:
            import boto3  # imported lazily so --dry-run doesn't require boto3
            from botocore.exceptions import BotoCoreError as _BotoCoreError, ClientError as _ClientError
            BotoCoreError = _BotoCoreError
            ClientError = _ClientError
            s3 = boto3.client("s3")
            paginator = s3.get_paginator("list_objects_v2")
        except ImportError as e:
            logger.error("boto3 is required to access S3 when not in dry-run mode: %s", e)
            return
    processor = MedicalPaperProcessor()
    processed = 0

    kwargs = {"Bucket": bucket, "Prefix": input_prefix}

    try:
        for page in paginator.paginate(**kwargs):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.lower().endswith(".xml"):
                    logger.debug("Skipping non-xml key: %s", key)
                    continue

                logger.info("Found: s3://%s/%s", bucket, key)

                if dry_run:
                    logger.info("Dry-run: would download and process %s", key)
                else:
                    try:
                        resp = s3.get_object(Bucket=bucket, Key=key)
                        xml_bytes = resp["Body"].read()
                        xml_text = xml_bytes.decode("utf-8")
                    except (BotoCoreError, ClientError) as e:
                        logger.exception("Failed to download %s: %s", key, e)
                        continue

                    try:
                        doc = processor.parse_pmc_xml(xml_text)
                        chunks = processor.chunk_document(doc, chunk_size=chunk_size)
                        out_key = key.replace(input_prefix, output_prefix).rstrip('.xml') + ".json"
                        body = json.dumps({"chunks": chunks}, ensure_ascii=False)

                        logger.info("Uploading processed result to s3://%s/%s", bucket, out_key)
                        s3.put_object(Bucket=bucket, Key=out_key, Body=body.encode('utf-8'), ContentType='application/json')
                    except Exception as e:
                        logger.exception("Processing failed for %s: %s", key, e)

                processed += 1
                if max_files and processed >= max_files:
                    logger.info("Reached max_files=%d, stopping", max_files)
                    return

    except (BotoCoreError, ClientError) as e:
        logger.exception("Failed listing objects in bucket %s with prefix %s: %s", bucket, input_prefix, e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process PMC XML files from S3 and upload chunked JSON")
    parser.add_argument("--bucket", default="medical-research-agent-documents", help="S3 bucket name")
    parser.add_argument("--prefix", default="raw/", help="S3 input prefix to look for XML files")
    parser.add_argument("--output-prefix", default="processed/", help="S3 output prefix for JSON files")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in characters")
    parser.add_argument("--max-files", type=int, default=10, help="Maximum number of files to process")
    parser.add_argument("--dry-run", action="store_true", help="If set, don't download or upload, just list")

    args = parser.parse_args()

    process_s3_bucket(bucket=args.bucket,
                      input_prefix=args.prefix,
                      output_prefix=args.output_prefix,
                      chunk_size=args.chunk_size,
                      max_files=args.max_files,
                      dry_run=args.dry_run)