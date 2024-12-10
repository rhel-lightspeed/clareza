"""Test the embed methods."""

from unittest.mock import patch

from clareza import embed


def test_clear_documents():
    """Test the clear_documents method."""
    with patch("clareza.embed.get_vector_store") as mock_get_vector_store:
        embed.clear_documents()
        mock_get_vector_store.assert_called_once()
