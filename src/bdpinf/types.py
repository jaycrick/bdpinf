from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, RootModel, field_validator

if TYPE_CHECKING:
    from collections.abc import Iterable

from .constants import CANONICAL_BOOKS, CANONICAL_SECTIONS


class ValidBookSet(RootModel[set[str]]):
    """Wrapper around a set of valid book names with validation."""

    root: set[str] = Field(default_factory=set)

    @field_validator("root")
    @classmethod
    def validate_books(cls, v: set[str]) -> set[str]:
        """Validate that all books are in the canonical set."""
        invalid = v - CANONICAL_BOOKS
        if invalid:
            msg = f"Invalid books: {invalid}"
            raise ValueError(msg)
        return v

    def add(self, book: str) -> None:
        """Add a single book to the set after validation."""
        validated = self.validate_books({book})
        self.root.update(validated)

    def update(self, books: Iterable[str]) -> None:
        """Update the set with multiple books after validation."""
        validated = self.validate_books(set(books))
        self.root.update(validated)

    def __contains__(self, book: str) -> bool:
        """Check if a book is in the set."""
        return book in self.root

    def __len__(self) -> int:
        """Return the number of books in the set."""
        return len(self.root)


class ValidSectionSet(RootModel[set[str]]):
    """Wrapper around a set of valid section names with validation."""

    root: set[str] = Field(default_factory=set)

    @field_validator("root")
    @classmethod
    def validate_sections(cls, v: set[str]) -> set[str]:
        """Validate that all sections are in the canonical set."""
        invalid = v - CANONICAL_SECTIONS
        if invalid:
            msg = f"Invalid sections: {invalid}"
            raise ValueError(msg)
        return v

    def add(self, section: str) -> None:
        """Add a single section to the set after validation."""
        validated = self.validate_sections({section})
        self.root.update(validated)

    def update(self, sections: Iterable[str]) -> None:
        """Update the set with multiple sections after validation."""
        validated = self.validate_sections(set(sections))
        self.root.update(validated)

    def __contains__(self, section: str) -> bool:
        """Check if a section is in the set."""
        return section in self.root

    def __len__(self) -> int:
        """Return the number of sections in the set."""
        return len(self.root)


class ParsedContentOverview(BaseModel):
    """Holds parsed content overview information (books + sections)."""

    books: ValidBookSet = Field(default_factory=ValidBookSet)
    sections: ValidSectionSet = Field(default_factory=ValidSectionSet)

    def merge(self, other: ParsedContentOverview) -> None:
        """In-place merge from another ParsedContentOverview."""
        self.books.update(other.books.root)
        self.sections.update(other.sections.root)


class BasicMetadata(BaseModel):
    """Holds basic metadata information for parsed content.

    Attributes
    ----------
    doc_id : int
        The document identifier.
    ga_num : str | None
        The GA number, if available.
    primary_name : str | None
        The primary name, if available.
    lang : str | None
        The language code, if available.
    early_year : int | None
        The earliest year, if available.
    late_year : int | None
        The latest year, if available.

    """

    doc_id: int
    ga_num: str | None = None
    primary_name: str | None = None
    lang: str | None = None
    early_year: int | None = None
    late_year: int | None = None
