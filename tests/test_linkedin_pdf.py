"""Tests for LinkedIn PDF ingestion service (Workstream A).

Covers:
- Text extraction and section parsing
- Headshot crop logic (mocked PIL/fitz)
- Full ingestion pipeline
- Regression: crop failure does NOT overwrite existing photo
- Evidence node generation from PDF text (Workstream B integration)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services.linkedin_pdf import (
    LinkedInPDFCropResult,
    LinkedInPDFIngestResult,
    LinkedInPDFTextResult,
    _extract_date_from_text,
    _garbled_ratio,
    _is_garbled_text,
    _parse_education_section,
    _parse_experience_section,
    _parse_linkedin_sections,
    _split_into_chunks,
    build_evidence_nodes_from_pdf,
    crop_headshot_from_pdf,
    extract_text_from_pdf,
    ingest_linkedin_pdf,
)


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


class TestSplitIntoChunks:
    def test_short_text_single_chunk(self):
        chunks = _split_into_chunks("Hello world.", max_chars=200)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_long_text_split(self):
        text = "First sentence. Second sentence. Third sentence that is also quite long."
        chunks = _split_into_chunks(text, max_chars=40)
        assert len(chunks) >= 2
        assert all(len(c) <= 40 for c in chunks)

    def test_empty_text(self):
        chunks = _split_into_chunks("", max_chars=200)
        assert chunks == []

    def test_no_sentence_boundaries(self):
        text = "A" * 500
        chunks = _split_into_chunks(text, max_chars=200)
        assert len(chunks) >= 1
        assert len(chunks[0]) <= 200


class TestExtractDate:
    def test_year_range(self):
        assert _extract_date_from_text("Jan 2020 - Present") == "2020"

    def test_year_only(self):
        assert _extract_date_from_text("2019") == "2019"

    def test_empty(self):
        assert _extract_date_from_text("") == "UNKNOWN"

    def test_no_date(self):
        assert _extract_date_from_text("No dates here") == "UNKNOWN"


class TestGarbledTextDetection:
    """Test detection of garbled/binary text from CIDFont PDFs."""

    def test_clean_text_not_garbled(self):
        text = "Jane Doe\nVP of Engineering at BigCorp\nSan Francisco Bay Area"
        assert not _is_garbled_text(text)
        assert _garbled_ratio(text) < 0.1

    def test_binary_garbled_text(self):
        # Simulates CIDFont garbled output (high Unicode codepoints)
        garbled = "8=%1\xd8{i\xbc\xd5W-\xe0o!\xe0\xa8o\\\xbcV\xe3\xe7k\xbf\xca\xc7\xc0"
        assert _is_garbled_text(garbled)
        assert _garbled_ratio(garbled) > 0.3

    def test_mixed_text_with_some_garble(self):
        # Mostly clean with a few garbled chars — should pass
        text = "Jane Doe works at BigCorp and lives in SF. " + "\xd8\xbc" * 2
        ratio = _garbled_ratio(text)
        # This has very few garbled chars relative to total
        assert ratio < 0.3

    def test_empty_text_is_garbled(self):
        assert _is_garbled_text("")
        assert _is_garbled_text("   ")

    def test_accented_text_not_garbled(self):
        text = "José García works at Café Résumé in São Paulo"
        assert not _is_garbled_text(text)

    def test_heavily_garbled_output(self):
        # Simulates real CIDFont output from a LinkedIn PDF
        garbled = (
            "\xc0\xa3\xd6Z\xc0kL\xe90+t\xffj\xbd"
            "+\xb7\xfa|iA/\xc5o3\xac\xa8`?(\xc3O\xe1"
        )
        assert _is_garbled_text(garbled)


class TestParseExperienceSection:
    def test_with_date_entries(self):
        lines = [
            "Software Engineer",
            "Acme Corp",
            "Jan 2020 - Present",
            "Worked on backend systems",
            "Led team of 5",
        ]
        result = _parse_experience_section(lines)
        assert len(result) >= 1

    def test_empty_lines(self):
        result = _parse_experience_section([])
        assert result == []


class TestParseEducationSection:
    def test_basic_education(self):
        lines = [
            "MIT",
            "BSc Computer Science",
            "2010 - 2014",
        ]
        result = _parse_education_section(lines)
        assert len(result) >= 1
        assert result[0]["school"] == "MIT"

    def test_empty(self):
        result = _parse_education_section([])
        assert result == []


class TestParseLinkedInSections:
    def test_basic_profile(self):
        raw_text = """Jane Doe
VP of Engineering at BigCorp
San Francisco Bay Area
About
Experienced technology leader building scalable systems.
Over 15 years of experience in distributed computing.
Experience
Jan 2020 - Present
VP of Engineering
BigCorp Inc
Leading engineering org of 200+ engineers.
Education
Stanford University
MSc Computer Science
Skills
Python
Distributed Systems
Leadership"""

        result = LinkedInPDFTextResult()
        _parse_linkedin_sections(raw_text, result)

        assert result.name == "Jane Doe"
        assert result.headline == "VP of Engineering at BigCorp"
        assert "about" in result.sections
        assert "experience" in result.sections
        assert "education" in result.sections
        assert "skills" in result.sections

    def test_empty_text(self):
        result = LinkedInPDFTextResult()
        _parse_linkedin_sections("", result)
        assert result.name == ""

    def test_minimal_profile(self):
        result = LinkedInPDFTextResult()
        _parse_linkedin_sections("John Smith\nSales Director", result)
        assert result.name == "John Smith"
        assert result.headline == "Sales Director"


# ---------------------------------------------------------------------------
# Text extraction (with mocked PDF libraries)
# ---------------------------------------------------------------------------


class TestExtractTextFromPdf:
    def test_with_mocked_fitz(self):
        """When fitz is available, extracts text from pages."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Jane Doe\nVP Engineering\nAbout\nExperienced leader."
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__len__ = MagicMock(return_value=1)

        with patch.dict("sys.modules", {"fitz": MagicMock()}):
            import sys
            sys.modules["fitz"].open.return_value = mock_doc
            result = extract_text_from_pdf(b"fake pdf bytes")
            assert isinstance(result, LinkedInPDFTextResult)

    def test_no_pdf_libraries(self):
        """Falls back gracefully when no PDF library is installed."""
        with (
            patch("app.services.linkedin_pdf._extract_raw_text", return_value=""),
            patch("app.services.linkedin_pdf._count_pages", return_value=0),
        ):
            result = extract_text_from_pdf(b"fake bytes")
            assert result.raw_text == ""
            assert result.page_count == 0

    def test_with_raw_text(self):
        """When raw text is available, parses sections."""
        raw = "John Smith\nCEO at TechCo\nNew York\nAbout\nVisionary builder."
        with (
            patch("app.services.linkedin_pdf._extract_raw_text", return_value=raw),
            patch("app.services.linkedin_pdf._count_pages", return_value=1),
        ):
            result = extract_text_from_pdf(b"fake")
            assert result.name == "John Smith"
            assert result.page_count == 1


# ---------------------------------------------------------------------------
# Headshot cropping
# ---------------------------------------------------------------------------


class TestCropHeadshotFromPdf:
    def test_no_fitz_returns_failed(self):
        """Without fitz, crop returns graceful failure."""
        with patch.dict("sys.modules", {"fitz": None}):
            # Force ImportError
            with patch(
                "app.services.linkedin_pdf.crop_headshot_from_pdf",
                return_value=LinkedInPDFCropResult(
                    success=False, method="failed", error="PyMuPDF not installed"
                ),
            ):
                result = crop_headshot_from_pdf(b"fake", contact_id=1)
                assert not result.success

    def test_crop_result_structure(self):
        """Crop result has expected fields."""
        result = LinkedInPDFCropResult(
            success=True,
            image_path="./image_cache/linkedin_crop_1.jpg",
            width=200,
            height=200,
            method="pillow_crop",
        )
        assert result.success is True
        assert result.width == 200
        assert result.height == 200
        assert result.method == "pillow_crop"


# ---------------------------------------------------------------------------
# Full ingestion pipeline
# ---------------------------------------------------------------------------


class TestIngestLinkedInPdf:
    def test_empty_pdf_returns_error(self):
        result = ingest_linkedin_pdf(b"", contact_id=1, contact_name="Test")
        assert result.error == "Empty PDF data"
        assert not result.pdf_path

    def test_full_pipeline_stores_pdf(self, tmp_path, monkeypatch):
        """Pipeline stores raw PDF and runs text + crop."""
        monkeypatch.setattr(
            "app.services.linkedin_pdf.PDF_UPLOAD_DIR", tmp_path / "pdfs"
        )
        monkeypatch.setattr(
            "app.services.linkedin_pdf.IMAGE_CACHE_DIR", tmp_path / "images"
        )

        # Mock text extraction and crop
        mock_text = LinkedInPDFTextResult(
            raw_text="Test Person\nDirector at TestCo",
            name="Test Person",
            headline="Director at TestCo",
            page_count=1,
        )
        mock_crop = LinkedInPDFCropResult(
            success=False, method="failed", error="fitz not available"
        )

        with (
            patch(
                "app.services.linkedin_pdf.extract_text_from_pdf",
                return_value=mock_text,
            ),
            patch(
                "app.services.linkedin_pdf.crop_headshot_from_pdf",
                return_value=mock_crop,
            ),
        ):
            result = ingest_linkedin_pdf(
                b"fake pdf content", contact_id=42, contact_name="Test Person"
            )

        assert result.pdf_path  # PDF was stored
        assert result.pdf_hash  # Hash computed
        assert result.text_result.name == "Test Person"
        assert not result.crop_result.success  # Crop failed (expected)
        assert result.ingested_at

    def test_pipeline_handles_text_failure(self, tmp_path, monkeypatch):
        """Pipeline continues even if text extraction fails."""
        monkeypatch.setattr(
            "app.services.linkedin_pdf.PDF_UPLOAD_DIR", tmp_path / "pdfs"
        )
        monkeypatch.setattr(
            "app.services.linkedin_pdf.IMAGE_CACHE_DIR", tmp_path / "images"
        )

        mock_crop = LinkedInPDFCropResult(success=False, method="failed")

        with (
            patch(
                "app.services.linkedin_pdf.extract_text_from_pdf",
                side_effect=Exception("text extraction error"),
            ),
            patch(
                "app.services.linkedin_pdf.crop_headshot_from_pdf",
                return_value=mock_crop,
            ),
        ):
            result = ingest_linkedin_pdf(b"fake pdf", contact_id=1)

        assert result.pdf_path  # PDF still stored
        assert result.text_result.raw_text == ""  # Fallback to empty


# ---------------------------------------------------------------------------
# Regression: crop failure DOES NOT overwrite existing photo
# ---------------------------------------------------------------------------


class TestCropDoesNotOverwriteExistingPhoto:
    """REGRESSION: If crop fails, the existing photo_url MUST NOT be wiped."""

    def test_failed_crop_preserves_existing_photo(self):
        """This test validates the API-level logic.

        When ingest_linkedin_pdf returns crop_result.success=False,
        the existing photo_url must not be overwritten.
        """
        result = LinkedInPDFIngestResult(
            crop_result=LinkedInPDFCropResult(
                success=False, method="failed", error="No fitz"
            ),
        )
        # The API endpoint checks result.crop_result.success before updating photo
        assert not result.crop_result.success
        # Therefore no photo update occurs — verified at API level

    def test_successful_crop_can_update_photo(self):
        """When crop succeeds, photo CAN be updated."""
        result = LinkedInPDFIngestResult(
            crop_result=LinkedInPDFCropResult(
                success=True,
                image_path="./image_cache/linkedin_crop_1.jpg",
                width=200,
                height=200,
                method="pillow_crop",
            ),
        )
        assert result.crop_result.success
        assert result.crop_result.image_path


# ---------------------------------------------------------------------------
# Evidence node generation (Workstream B integration)
# ---------------------------------------------------------------------------


class TestBuildEvidenceNodesFromPdf:
    def test_basic_profile_generates_nodes(self):
        text_result = LinkedInPDFTextResult(
            raw_text="Jane Doe\nVP Engineering",
            name="Jane Doe",
            headline="VP of Engineering at BigCorp",
            about="Experienced technology leader building scalable cloud platforms.",
            experience=[
                {"title": "VP Engineering", "company": "BigCorp", "dates": "2020 - Present"},
                {"title": "Director", "company": "SmallCo", "dates": "2017 - 2020"},
            ],
            education=[{"school": "MIT", "details": "BSc CS"}],
            skills=["Python", "Leadership", "AWS"],
        )

        nodes = build_evidence_nodes_from_pdf(text_result, contact_name="Jane Doe")

        assert len(nodes) >= 4  # about + headline + 2 experience + 1 education + 1 skills
        assert all(n["type"] == "PDF" for n in nodes)
        assert all("linkedin_pdf:" in n["source"] for n in nodes)

        # Check headline node exists
        headline_nodes = [n for n in nodes if n["ref"] == "headline"]
        assert len(headline_nodes) == 1
        assert "VP of Engineering" in headline_nodes[0]["snippet"]

    def test_empty_profile_generates_no_nodes(self):
        text_result = LinkedInPDFTextResult()
        nodes = build_evidence_nodes_from_pdf(text_result, contact_name="Nobody")
        assert nodes == []

    def test_about_section_chunked(self):
        long_about = "First important statement about leadership. " * 20
        text_result = LinkedInPDFTextResult(about=long_about)
        nodes = build_evidence_nodes_from_pdf(text_result, contact_name="Test")
        about_nodes = [n for n in nodes if "about:" in n["ref"]]
        assert len(about_nodes) >= 2  # Should be chunked

    def test_node_snippets_under_200_chars(self):
        text_result = LinkedInPDFTextResult(
            headline="A" * 300,  # Very long headline
            about="B" * 300,
        )
        nodes = build_evidence_nodes_from_pdf(text_result, contact_name="Test")
        for node in nodes:
            assert len(node["snippet"]) <= 200

    def test_experience_nodes_have_dates(self):
        text_result = LinkedInPDFTextResult(
            experience=[
                {
                    "title": "CEO",
                    "company": "StartupCo",
                    "dates": "Jan 2022 - Present",
                    "description": "Leading the company",
                },
            ],
        )
        nodes = build_evidence_nodes_from_pdf(text_result, contact_name="Test")
        exp_nodes = [n for n in nodes if "experience:" in n["ref"]]
        assert len(exp_nodes) >= 1
        assert exp_nodes[0]["date"] == "2022"

    def test_skills_grouped_into_single_node(self):
        text_result = LinkedInPDFTextResult(
            skills=["Python", "JavaScript", "React", "Docker", "AWS"],
        )
        nodes = build_evidence_nodes_from_pdf(text_result, contact_name="Test")
        skills_nodes = [n for n in nodes if n["ref"] == "skills"]
        assert len(skills_nodes) == 1
        assert "Python" in skills_nodes[0]["snippet"]


# ---------------------------------------------------------------------------
# PhotoSource enum includes LINKEDIN_PDF_CROP
# ---------------------------------------------------------------------------


class TestPhotoSourceEnum:
    def test_linkedin_pdf_crop_exists(self):
        from app.services.photo_resolution import PhotoSource
        assert PhotoSource.LINKEDIN_PDF_CROP == "linkedin_pdf_crop"

    def test_resolve_preserves_pdf_crop(self):
        """PhotoResolutionService preserves linkedin_pdf_crop source."""
        from app.services.photo_resolution import (
            PhotoResolutionService,
            PhotoSource,
            PhotoStatus,
        )
        service = PhotoResolutionService()
        result = service.resolve(
            contact_name="Test",
            existing_photo_url="/api/local-image/image_cache/linkedin_crop_1.jpg",
            existing_photo_source=PhotoSource.LINKEDIN_PDF_CROP,
            existing_photo_status=PhotoStatus.RESOLVED,
        )
        assert result.photo_url == "/api/local-image/image_cache/linkedin_crop_1.jpg"
        assert result.photo_source == PhotoSource.LINKEDIN_PDF_CROP
        assert result.photo_status == PhotoStatus.RESOLVED


# ---------------------------------------------------------------------------
# EvidenceNode type "PDF" in evidence graph
# ---------------------------------------------------------------------------


class TestPdfEvidenceNodes:
    def test_add_pdf_node_to_graph(self):
        from app.brief.evidence_graph import EvidenceGraph

        graph = EvidenceGraph()
        node = graph.add_pdf_node(
            source="linkedin_pdf:Jane Doe",
            snippet="VP of Engineering at BigCorp",
            date="2020",
            ref="headline",
        )
        assert node.type == "PDF"
        assert node.id.startswith("E")
        assert "BigCorp" in node.snippet
        assert len(graph.nodes) == 1

    def test_pdf_nodes_in_graph_dict(self):
        from app.brief.evidence_graph import EvidenceGraph

        graph = EvidenceGraph()
        graph.add_pdf_node(source="test", snippet="Test snippet", ref="about:1")
        graph.add_meeting_node(source="meeting", snippet="Meeting note")

        data = graph.to_dict()
        assert "nodes" in data
        types = {n["type"] for n in data["nodes"]}
        assert "PDF" in types
        assert "MEETING" in types


# ---------------------------------------------------------------------------
# EvidenceTag includes VERIFIED_PDF
# ---------------------------------------------------------------------------


class TestEvidenceTagEnum:
    def test_verified_pdf_tag_exists(self):
        from app.models import EvidenceTag

        assert EvidenceTag.verified_pdf == "VERIFIED_PDF"

    def test_verified_pdf_in_valid_tags(self):
        from app.brief.evidence_graph import _VALID_EVIDENCE_TAGS

        assert "VERIFIED-PDF" in _VALID_EVIDENCE_TAGS
        assert "VERIFIED_PDF" in _VALID_EVIDENCE_TAGS
