from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field, RootModel, field_validator

from .constants import CANONICAL_BOOKS, CANONICAL_SECTIONS


class ValidBookSet(RootModel[set[str]]):
    """Wrapper around a set of valid book names with validation."""

    root: set[str] = Field(default_factory=set)

    @field_validator("root")
    @classmethod
    def validate_books(cls, v: set[str]) -> set[str]:
        invalid = v - CANONICAL_BOOKS
        if invalid:
            raise ValueError(f"Invalid books: {invalid}")
        return v

    def add(self, book: str) -> None:
        validated = self.validate_books({book})
        self.root.update(validated)

    def update(self, books: Iterable[str]) -> None:
        validated = self.validate_books(set(books))
        self.root.update(validated)

    def __contains__(self, book: str) -> bool:
        return book in self.root

    def __len__(self) -> int:
        return len(self.root)


class ValidSectionSet(RootModel[set[str]]):
    """Wrapper around a set of valid section names with validation."""

    root: set[str] = Field(default_factory=set)

    @field_validator("root")
    @classmethod
    def validate_sections(cls, v: set[str]) -> set[str]:
        invalid = v - CANONICAL_SECTIONS
        if invalid:
            raise ValueError(f"Invalid sections: {invalid}")
        return v

    def add(self, section: str) -> None:
        validated = self.validate_sections({section})
        self.root.update(validated)

    def update(self, sections: Iterable[str]) -> None:
        validated = self.validate_sections(set(sections))
        self.root.update(validated)

    def __contains__(self, section: str) -> bool:
        return section in self.root

    def __len__(self) -> int:
        return len(self.root)


class ParsedContentOverview(BaseModel):
    """Holds parsed content overview information (books + sections)."""

    books: ValidBookSet = Field(default_factory=lambda: ValidBookSet())
    sections: ValidSectionSet = Field(default_factory=lambda: ValidSectionSet())

    def merge(self, other: ParsedContentOverview) -> None:
        """In-place merge from another ParsedContentOverview."""
        self.books.update(other.books.root)
        self.sections.update(other.sections.root)


class BasicMetadata(BaseModel):
    docID: int
    gaNum: str | None = None
    primaryName: str | None = None
    lang: str | None = None
    earlyYear: int | None = None
    lateYear: int | None = None
