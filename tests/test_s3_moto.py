import json
import boto3
import moto

from document_processor import process_s3_bucket

SAMPLE_PMC_XML = '''<?xml version="1.0"?>
<article>
  <front>
    <article-meta>
      <article-id pub-id-type="pmid">99999</article-id>
      <article-id pub-id-type="doi">10.1000/motodoi</article-id>
      <title-group>
        <article-title>Moto Test Paper</article-title>
      </title-group>
      <pub-date>
        <year>2023</year>
        <month>11</month>
        <day>01</day>
      </pub-date>
      <abstract>
        <p>Moto abstract text.</p>
      </abstract>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Results</title>
      <p>Results paragraph.</p>
    </sec>
  </body>
</article>
'''


def test_process_s3_with_moto(tmp_path):
  bucket = "medical-research-agent-documents"

  # Choose the correct moto context manager available in this environment
  if hasattr(moto, 'mock_s3'):
    ctx = moto.mock_s3()
  else:
    # mock_aws() without args returns a MockAWS object that can be used as a context manager
    ctx = moto.mock_aws()

  with ctx:
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=bucket)

    # put a sample XML under raw/
    key = "raw/test-moto-1.xml"
    s3.put_object(Bucket=bucket, Key=key, Body=SAMPLE_PMC_XML.encode('utf-8'))

    # run processing (not dry-run)
    process_s3_bucket(bucket=bucket, input_prefix="raw/", output_prefix="processed/", dry_run=False, max_files=5)

    # check processed object exists
    out_key = "processed/test-moto-1.json"
    resp = s3.get_object(Bucket=bucket, Key=out_key)
    body = resp['Body'].read().decode('utf-8')
    data = json.loads(body)

    assert 'chunks' in data
    assert len(data['chunks']) > 0
