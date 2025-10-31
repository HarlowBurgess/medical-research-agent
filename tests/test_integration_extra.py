import json
import boto3
import moto
from document_processor import process_s3_bucket

SAMPLE_XML_TEMPLATE = '''<?xml version="1.0"?>
<article>
  <front>
    <article-meta>
      <article-id pub-id-type="pmid">{pmid}</article-id>
      <article-id pub-id-type="doi">10.1000/{pmid}</article-id>
      <title-group>
        <article-title>Title {pmid}</article-title>
      </title-group>
      <pub-date>
        <year>2024</year>
        <month>01</month>
        <day>15</day>
      </pub-date>
      <abstract>
        <p>Abstract for {pmid}.</p>
      </abstract>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Content</title>
      <p>{body}</p>
    </sec>
  </body>
</article>
'''


def _get_ctx():
    return moto.mock_s3() if hasattr(moto, 'mock_s3') else moto.mock_aws()


def test_process_multiple_files():
    bucket = 'medical-research-agent-documents'
    ctx = _get_ctx()
    with ctx:
        s3 = boto3.client('s3', region_name='us-east-1')
        s3.create_bucket(Bucket=bucket)

        # upload two xmls
        s3.put_object(Bucket=bucket, Key='raw/a.xml', Body=SAMPLE_XML_TEMPLATE.format(pmid='a', body='a body'))
        s3.put_object(Bucket=bucket, Key='raw/b.xml', Body=SAMPLE_XML_TEMPLATE.format(pmid='b', body='b body'))

        process_s3_bucket(bucket=bucket, input_prefix='raw/', output_prefix='processed/', dry_run=False)

        # check outputs
        out_a = s3.get_object(Bucket=bucket, Key='processed/a.json')
        out_b = s3.get_object(Bucket=bucket, Key='processed/b.json')
        assert out_a and out_b


def test_skip_non_xml_files():
    bucket = 'medical-research-agent-documents'
    ctx = _get_ctx()
    with ctx:
        s3 = boto3.client('s3', region_name='us-east-1')
        s3.create_bucket(Bucket=bucket)

        s3.put_object(Bucket=bucket, Key='raw/ignore.txt', Body=b'should be ignored')
        s3.put_object(Bucket=bucket, Key='raw/keep.xml', Body=SAMPLE_XML_TEMPLATE.format(pmid='keep', body='keep body'))

        process_s3_bucket(bucket=bucket, input_prefix='raw/', output_prefix='processed/', dry_run=False)

        # should exist
        _ = s3.get_object(Bucket=bucket, Key='processed/keep.json')
        # non-xml should not be processed (expecting 404 if we try to get it)
        try:
            s3.get_object(Bucket=bucket, Key='processed/ignore.json')
            got = True
        except Exception:
            got = False
        assert not got


def test_chunk_size_boundary():
    bucket = 'medical-research-agent-documents'
    ctx = _get_ctx()
    long_par = ' '.join(['word'] * 200)
    xml = SAMPLE_XML_TEMPLATE.format(pmid='boundary', body=long_par)

    with ctx:
        s3 = boto3.client('s3', region_name='us-east-1')
        s3.create_bucket(Bucket=bucket)
        s3.put_object(Bucket=bucket, Key='raw/boundary.xml', Body=xml)

        # use small chunk size to force multiple chunks
        process_s3_bucket(bucket=bucket, input_prefix='raw/', output_prefix='processed/', dry_run=False, chunk_size=100)

        out = s3.get_object(Bucket=bucket, Key='processed/boundary.json')
        data = json.loads(out['Body'].read().decode('utf-8'))
        assert 'chunks' in data
        assert len(data['chunks']) >= 2
