"""Tests for OpenAlex validation."""

import pytest

from authen.core.schemas import Author, ReferenceData


@pytest.mark.asyncio
async def test_calculate_title_similarity():
    """Test title similarity calculation."""
    from authen.validation.openalex import OpenAlexValidator

    validator = OpenAlexValidator()

    # Exact match
    sim = validator._calculate_title_similarity(
        "Machine Learning for Drug Discovery",
        "Machine Learning for Drug Discovery",
    )
    assert sim == 1.0

    # Similar titles
    sim = validator._calculate_title_similarity(
        "Machine Learning for Drug Discovery",
        "Machine learning for drug discovery",
    )
    assert sim > 0.95

    # Different titles
    sim = validator._calculate_title_similarity(
        "Machine Learning for Drug Discovery",
        "Quantum Computing Applications",
    )
    assert sim < 0.5

    await validator.close()


@pytest.mark.asyncio
async def test_calculate_author_similarity():
    """Test author similarity calculation."""
    from authen.validation.openalex import OpenAlexValidator

    validator = OpenAlexValidator()

    authors1 = [
        Author(first_name="John", last_name="Smith"),
        Author(first_name="Jane", last_name="Doe"),
    ]

    authorships = [
        {"author": {"display_name": "John Smith"}},
        {"author": {"display_name": "Jane Doe"}},
    ]

    sim = validator._calculate_author_similarity(authors1, authorships)
    assert sim == 1.0

    # Partial match
    authorships_partial = [
        {"author": {"display_name": "John Smith"}},
        {"author": {"display_name": "Alice Johnson"}},
    ]

    sim_partial = validator._calculate_author_similarity(authors1, authorships_partial)
    assert 0 < sim_partial < 1

    await validator.close()


@pytest.mark.asyncio
async def test_extract_reference_data():
    """Test extracting reference data from OpenAlex work."""
    from authen.validation.openalex import OpenAlexValidator

    validator = OpenAlexValidator()

    original = ReferenceData(
        title="Test Paper",
        raw_text="[1] Test Paper...",
    )

    work = {
        "id": "https://openalex.org/W123456",
        "doi": "https://doi.org/10.1234/test",
        "title": "Test Paper: A Comprehensive Study",
        "publication_year": 2023,
        "type": "article",
        "authorships": [
            {
                "author": {
                    "id": "https://openalex.org/A123",
                    "display_name": "John Smith",
                    "orcid": "https://orcid.org/0000-0001-2345-6789",
                },
                "institutions": [
                    {
                        "id": "https://openalex.org/I123",
                        "display_name": "MIT",
                        "country_code": "US",
                        "ror": "https://ror.org/042nb2s44",
                    }
                ],
            }
        ],
        "primary_location": {
            "source": {
                "display_name": "Nature",
                "publisher": "Springer Nature",
            }
        },
        "biblio": {
            "volume": "595",
            "issue": "7865",
            "first_page": "123",
            "last_page": "130",
        },
        "cited_by_count": 42,
        "is_oa": True,
    }

    ref = validator._extract_reference_data(work, original)

    assert ref.title == "Test Paper: A Comprehensive Study"
    assert ref.doi == "10.1234/test"
    assert ref.year == "2023"
    assert ref.publication == "Nature"
    assert ref.publisher == "Springer Nature"
    assert ref.volume == "595"
    assert ref.cited_by_count == 42
    assert ref.is_open_access is True
    assert len(ref.authors) == 1
    assert ref.authors[0].full_name == "John Smith"
    assert ref.authors[0].orcid == "https://orcid.org/0000-0001-2345-6789"
    assert len(ref.authors[0].affiliations) == 1
    assert ref.authors[0].affiliations[0].name == "MIT"

    # Check that original fields are preserved
    assert ref.raw_text == "[1] Test Paper..."

    await validator.close()
