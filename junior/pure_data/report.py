"""DQ Report."""

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
from user_input.metrics import Metric

import pandas as pd
import pyspark.sql as ps

LimitType = Dict[str, Tuple[float, float]]
CheckType = Tuple[str, Metric, LimitType]


@dataclass
class Report:
    """DQ report class."""

    checklist: List[CheckType]
    engine: str = "pandas"

    def fit(self, tables: Dict[str, Union[pd.DataFrame, ps.DataFrame]]) -> Dict:
        """Calculate DQ metrics and build report."""

        if self.engine == "pandas":
            return self._fit_pandas(tables)

        if self.engine == "pyspark":
            return self._fit_pyspark(tables)

        raise NotImplementedError("Only pandas and pyspark APIs currently supported!")

    def _fit_pandas(self, tables: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate DQ metrics and build report.  Engine: Pandas"""
        self.report_ = {}
        report = self.report_

        title = "DQ Report for tables ['big_table', 'sales', 'views']"
        passed = 0
        failed = 0
        errors = 0
        total = 0
        results = []

        for table, metric, limits in self.checklist:
            try:
                result = metric(tables[table])
                total += 1

                status = "."
                for metric_key, (lower_limit, upper_limit) in limits.items():
                    value = result.get(metric_key)
                    if value is None:
                        status = "F"
                        failed += 1
                        break
                    if not (lower_limit <= value <= upper_limit):
                        status = "F"
                        failed += 1
                        break

                results.append(
                    {
                        "table_name": table,
                        "metric": str(metric),
                        "limits": str(limits),
                        "values": result,
                        "status": status,
                        "error": '',
                    }
                )
                if status == ".":
                    passed += 1
            except Exception as e:
                results.append(
                    {
                        "table_name": table,
                        "metric": str(metric),
                        "limits": str(limits),
                        "values": None,
                        "status": "E",
                        "error": str(e),
                    }
                )
                errors += 1

        passed_pct = (passed / total) * 100 if total > 0 else 0
        failed_pct = (failed / total) * 100 if total > 0 else 0
        errors_pct = (errors / total) * 100 if total > 0 else 0

        report["title"] = title
        report["result"] = pd.DataFrame(results)
        report["passed"] = passed
        report["failed"] = failed
        report["errors"] = errors
        report["total"] = total
        report["passed_pct"] = passed_pct
        report["failed_pct"] = failed_pct
        report["errors_pct"] = errors_pct

        return report


    def _fit_pyspark(self, tables: Dict[str, ps.DataFrame]) -> Dict:
        """Calculate DQ metrics and build report.  Engine: PySpark"""
        self.report_ = {}
        report = self.report_

        title = "DQ Report for tables ['big_table', 'sales', 'views']"
        passed = 0
        failed = 0
        errors = 0
        total = 0
        results = []

        for table, metric, limits in self.checklist:
            try:
                result = metric(tables[table])
                total += 1

                status = "."
                for metric_key, (lower_limit, upper_limit) in limits.items():
                    value = result.get(metric_key)
                    if value is None:
                        status = "F"
                        failed += 1
                        break
                    if not (lower_limit <= value <= upper_limit):
                        status = "F"
                        failed += 1
                        break

                results.append(
                    {
                        "table_name": table,
                        "metric": str(metric),
                        "limits": str(limits),
                        "values": result,
                        "status": status,
                        "error": '',
                    }
                )
                if status == ".":
                    passed += 1
            except Exception as e:
                results.append(
                    {
                        "table_name": table,
                        "metric": str(metric),
                        "limits": str(limits),
                        "values": {},
                        "status": "E",
                        "error": str(e),
                    }
                )
                errors += 1

        passed_pct = (passed / total) * 100 if total > 0 else 0
        failed_pct = (failed / total) * 100 if total > 0 else 0
        errors_pct = (errors / total) * 100 if total > 0 else 0

        report["title"] = title
        report["result"] = pd.DataFrame(results)
        report["passed"] = passed
        report["failed"] = failed
        report["errors"] = errors
        report["total"] = total
        report["passed_pct"] = passed_pct
        report["failed_pct"] = failed_pct
        report["errors_pct"] = errors_pct

        return report

    def to_str(self) -> None:
        """Convert report to string format."""
        report = self.report_

        msg = (
            "This Report instance is not fitted yet. "
            "Call 'fit' before usong this method."
        )

        assert isinstance(report, dict), msg

        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)
        pd.set_option("display.max_colwidth", 20)
        pd.set_option("display.width", 1000)

        return (
            f"{report['title']}\n\n"
            f"{report['result']}\n\n"
            f"Passed: {report['passed']} ({report['passed_pct']}%)\n"
            f"Failed: {report['failed']} ({report['failed_pct']}%)\n"
            f"Errors: {report['errors']} ({report['errors_pct']}%)\n"
            "\n"
            f"Total: {report['total']}"
        )
