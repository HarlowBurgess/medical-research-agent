# Lightweight test runner for environments without pytest
import sys
import os

# Ensure repository root is on sys.path so tests can import the package
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from tests.test_document_processor import (
    test_parse_pmc_xml_basic,
    test_chunk_document_basic,
    test_process_s3_bucket_dry_run_no_boto,
)

import logging

logging.basicConfig(level=logging.INFO)

def main():
    try:
        # Provide the sample_pmc_xml fixture value expected by the test
        from tests.test_document_processor import sample_pmc_xml
        test_parse_pmc_xml_basic(sample_pmc_xml())
        print('test_parse_pmc_xml_basic: PASS')
    except AssertionError as e:
        print('test_parse_pmc_xml_basic: FAIL', e)
        sys.exit(1)

    try:
        test_chunk_document_basic()
        print('test_chunk_document_basic: PASS')
    except AssertionError as e:
        print('test_chunk_document_basic: FAIL', e)
        sys.exit(1)

    try:
        # ensure boto3 not importable for the dry-run test
        sys.modules.pop('boto3', None)
        test_process_s3_bucket_dry_run_no_boto()
        print('test_process_s3_bucket_dry_run_no_boto: PASS')
    except AssertionError as e:
        print('test_process_s3_bucket_dry_run_no_boto: FAIL', e)
        sys.exit(1)
    except Exception as e:
        print('test_process_s3_bucket_dry_run_no_boto: ERROR', e)
        sys.exit(1)

    print('All tests passed')


if __name__ == '__main__':
    main()
