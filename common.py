import logging

logger = logging.getLogger(__name__)
import os
from opensearchpy import OpenSearch


def get_opensearch_url():
    domain = os.environ.get("OPENSEARCH_DOMAIN", "localhost")
    port = os.environ.get("OPENSEARCH_PORT", "9200")
    url = f"https://{domain}:{port}/"
    return url


def opensearch_client():
    vector_store_url = get_opensearch_url()
    username = os.environ.get("OPENSEARCH_USERNAME")
    password = os.environ.get("OPENSEARCH_PASSWORD")
    logger.info(f"OpenSearch vector_store_url: {vector_store_url}")
    logger.info(f"OpenSearch username: {username}")
    client = OpenSearch(
        hosts=[vector_store_url],
        http_auth=(username, password),
        http_compress=True,
        use_ssl=False,
        verify_certs=False,  # DONT USE IN PRODUCTION
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )
    logger.info(f"OpenSearch client instantiated: {client.info()}")
    return client


os_client = opensearch_client()
