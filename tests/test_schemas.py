"""Tests for core schemas."""

import pytest

from authen.core.schemas import (
    Affiliation,
    Author,
    ReferenceData,
    ValidationResult,
    ValidationStatus,
)


class TestAffiliation:
    """Tests for Affiliation model."""

    def test_create_affiliation(self):
        """Test creating an affiliation."""
        aff = Affiliation(
            name="MIT",
            department="CSAIL",
            country="USA",
            city="Cambridge",
        )
        assert aff.name == "MIT"
        assert aff.department == "CSAIL"

    def test_to_dict(self):
        """Test converting affiliation to dict."""
        aff = Affiliation(name="Harvard", country="USA")
        d = aff.to_dict()
        assert d["name"] == "Harvard"
        assert d["country"] == "USA"
        assert "department" not in d  # None values excluded


class TestAuthor:
    """Tests for Author model."""

    def test_display_name_from_parts(self):
        """Test display name from first and last name."""
        author = Author(first_name="John", last_name="Doe")
        assert author.display_name == "John Doe"

    def test_display_name_from_full(self):
        """Test display name from full_name."""
        author = Author(full_name="Dr. Jane Smith")
        assert author.display_name == "Dr. Jane Smith"

    def test_display_name_fallback(self):
        """Test display name fallback to Unknown."""
        author = Author()
        assert author.display_name == "Unknown"

    def test_author_with_affiliations(self):
        """Test author with multiple affiliations."""
        author = Author(
            first_name="Alice",
            last_name="Johnson",
            affiliations=[
                Affiliation(name="Stanford"),
                Affiliation(name="Google"),
            ],
        )
        assert len(author.affiliations) == 2


class TestReferenceData:
    """Tests for ReferenceData model."""

    def test_create_reference(self):
        """Test creating a reference."""
        ref = ReferenceData(
            title="Test Paper",
            year="2023",
            doi="10.1234/test",
            authors=[Author(first_name="John", last_name="Doe")],
        )
        assert ref.title == "Test Paper"
        assert ref.doi == "10.1234/test"
        assert len(ref.authors) == 1

    def test_has_identifier(self):
        """Test has_identifier method."""
        ref_with_doi = ReferenceData(doi="10.1234/test")
        assert ref_with_doi.has_identifier() is True

        ref_with_pmid = ReferenceData(pmid="12345678")
        assert ref_with_pmid.has_identifier() is True

        ref_no_id = ReferenceData(title="No ID Paper")
        assert ref_no_id.has_identifier() is False

    def test_get_search_title(self):
        """Test get_search_title method."""
        ref = ReferenceData(title='"Quoted Title"')
        assert ref.get_search_title() == "Quoted Title"

        ref2 = ReferenceData(title="  Normal Title  ")
        assert ref2.get_search_title() == "Normal Title"

        ref3 = ReferenceData()
        assert ref3.get_search_title() is None


class TestValidationResult:
    """Tests for ValidationResult model."""

    def test_is_valid(self):
        """Test is_valid method."""
        validated = ValidationResult(
            original=ReferenceData(title="Test"),
            status=ValidationStatus.VALIDATED,
        )
        assert validated.is_valid() is True

        partial = ValidationResult(
            original=ReferenceData(title="Test"),
            status=ValidationStatus.PARTIAL_MATCH,
        )
        assert partial.is_valid() is True

        not_found = ValidationResult(
            original=ReferenceData(title="Test"),
            status=ValidationStatus.NOT_FOUND,
        )
        assert not_found.is_valid() is False

    def test_get_best_reference(self):
        """Test get_best_reference method."""
        original = ReferenceData(title="Original")
        validated = ReferenceData(title="Validated", doi="10.1234/test")

        result = ValidationResult(
            original=original,
            validated=validated,
            status=ValidationStatus.VALIDATED,
        )
        assert result.get_best_reference().title == "Validated"

        result_not_found = ValidationResult(
            original=original,
            status=ValidationStatus.NOT_FOUND,
        )
        assert result_not_found.get_best_reference().title == "Original"
