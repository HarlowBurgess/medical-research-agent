import pytest

from document_processor import MedicalPaperProcessor, process_s3_bucket


@pytest.fixture
def sample_pmc_xml():
    return '''<?xml version="1.0"?>
<article>
  <front>
    <article-meta>
      <article-id pub-id-type="pmid">12345</article-id>
      <article-id pub-id-type="doi">10.1000/testdoi</article-id>
      <title-group>
        <article-title>Test Paper Title</article-title>
      </title-group>
      <pub-date>
        <year>2022</year>
        <month>7</month>
        <day>15</day>
      </pub-date>
      <abstract>
        <p>This is the abstract text.</p>
      </abstract>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Introduction</title>
      <p>Intro paragraph one.</p>
      <p>Intro paragraph two.</p>
    </sec>
    <sec>
      <title>Methods</title>
      <p>Methods paragraph.</p>
    </sec>
  </body>
</article>
'''


def test_parse_pmc_xml_basic(sample_pmc_xml):
    proc = MedicalPaperProcessor()
    result = proc.parse_pmc_xml(sample_pmc_xml)

    assert "metadata" in result
    assert result["metadata"]["pmid"] == "12345"
    assert result["metadata"]["doi"] == "10.1000/testdoi"
    assert result["metadata"]["title"] == "Test Paper Title"
    assert result["metadata"]["pub_date"] == "2022-07-15"

    assert "abstract" in result
    assert "This is the abstract text." in result["abstract"]

    assert "sections" in result
    assert "introduction" in result["sections"]
    assert "methods" in result["sections"]


def test_chunk_document_basic(sample_pmc_xml):
    proc = MedicalPaperProcessor()
    doc = proc.parse_pmc_xml(sample_pmc_xml)
    chunks = proc.chunk_document(doc, chunk_size=50)

    # abstract should be first chunk
    assert chunks[0]["metadata"]["section"] == "abstract"
    assert any(c["metadata"]["section"] == "introduction" for c in chunks)
    assert any(c["metadata"]["section"] == "methods" for c in chunks)


def test_process_s3_bucket_dry_run_no_boto(monkeypatch, caplog):
  # Simulate ImportError for boto3 by monkeypatching builtins.__import__
  import builtins

  real_import = builtins.__import__

  def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'boto3' or (fromlist and 'boto3' in fromlist):
      raise ImportError('No module named boto3')
    return real_import(name, globals, locals, fromlist, level)

  monkeypatch.setattr(builtins, '__import__', fake_import)
  caplog.set_level('INFO')

  # Call process_s3_bucket in dry-run; code should catch ImportError and log accordingly
  process_s3_bucket(bucket="dummy", input_prefix="raw/", output_prefix="processed/", dry_run=True, max_files=1)

  assert any("Dry-run and boto3 not available" in rec.message for rec in caplog.records)
