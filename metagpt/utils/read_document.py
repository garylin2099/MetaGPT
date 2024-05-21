#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/29 15:45
@Author  : alexanderwu
@File    : read_document.py
"""

import docx
import nbformat


def read_docx(file_path: str) -> list:
    """Open a docx file"""
    doc = docx.Document(file_path)

    # Create an empty list to store paragraph contents
    paragraphs_list = []

    # Iterate through the paragraphs in the document and add their content to the list
    for paragraph in doc.paragraphs:
        paragraphs_list.append(paragraph.text)

    return paragraphs_list


class NotebookReader:
    def __init__(self, path: str):
        self.path = path
        self.notebook = self.load_notebook()

    def load_notebook(self):
        """Load the notebook from the file."""
        with open(self.path, "r", encoding="utf-8") as f:
            return nbformat.read(f, as_version=4)

    def extract_code_cells(self, error_exclude: bool = True):
        """
        Extract code cells from the notebook.

        Args:
            error_exclude: whether to exclude error-raising code cells. Default is True.
        """
        code_cells = []
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                if not error_exclude or "raise Exception" not in cell.source:
                    code_cells.append(cell.source)
        return code_cells

    def extract_markdown_cells(self):
        """Extract markdown cells from the notebook."""
        return [
            cell.source for cell in self.notebook.cells if cell.cell_type == "markdown"
        ]

    def extract_all_cells(self):
        """Extract all cells from the notebook."""
        return [cell.source for cell in self.notebook.cells]
