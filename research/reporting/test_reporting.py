# research/reporting/test_reporting.py
import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from .reporting import ExcelReporter, HTMLReporter, SummaryReporter, ReportFactory


def test_excel_reporter():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        reporter = ExcelReporter(tmp.name)
        reporter.add_sheet(df, "Test", index=False)
        reporter.save()
        assert os.path.exists(tmp.name)
    os.unlink(tmp.name)


def test_html_reporter():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        reporter = HTMLReporter(tmp.name, title="Test Raporu")
        reporter.add_text("Test Başlığı", level="h1")
        reporter.add_table(df, "Test Tablosu")
        reporter.save()
        assert os.path.exists(tmp.name)
    os.unlink(tmp.name)


def test_summary_reporter():
    reporter = SummaryReporter()
    reporter.add_metric("test", 42, "Test metrik")
    df = reporter.to_dataframe()
    assert not df.empty
    assert df.iloc[0]["Metrik"] == "test"


def test_factory():
    excel = ReportFactory.create_excel("test.xlsx")
    assert isinstance(excel, ExcelReporter)
    html = ReportFactory.create_html("test.html")
    assert isinstance(html, HTMLReporter)
    summary = ReportFactory.create_summary()
    assert isinstance(summary, SummaryReporter)