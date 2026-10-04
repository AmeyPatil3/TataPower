"""
Safe Parameter-to-SQL Pipeline for Tata Power Platform.
Extracts structured intent from analytical queries, validates against strict Pydantic schemas,
and builds read-only parameterized SELECT queries preventing SQL injection.
"""

import os
import re
import sys
from typing import Optional, List, Dict, Any, Tuple
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from pydantic import BaseModel, Field

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, PROJECT_ROOT)

from backend.db import get_db_connection


class QueryIntent(BaseModel):
    metric: Literal["AVG", "MAX", "MIN", "SUM", "COUNT"] = Field(
        default="AVG", description="SQL aggregation function"
    )
    variable: Literal["wdLoad", "temp", "rh"] = Field(
        default="wdLoad", description="Target measurement column"
    )
    season: Optional[Literal["Winter", "Summer", "Monsoon", "Autumn"]] = None
    year: Optional[int] = None
    date: Optional[str] = None          # 'YYYY-MM-DD'
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    hour: Optional[int] = None
    weekend: Optional[int] = None
    holiday: Optional[int] = None
    festival: Optional[int] = None
    office_hours: Optional[int] = None
    group_by: Optional[Literal["hour", "date", "day_of_week", "season"]] = None
    comparison_type: Optional[Literal["dates", "seasons", "none"]] = "none"
    compare_dates: Optional[List[str]] = None
    compare_seasons: Optional[List[str]] = None


class SafeSQLAgent:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def extract_intent(self, question: str) -> QueryIntent:
        """
        Extracts validated query intent using keyword/regex rule heuristics.
        Can also be augmented by local LLM function calling.
        """
        q = question.lower()

        # 1. Determine Metric
        metric = "AVG"
        if re.search(r"\b(max|maximum|peak|highest)\b", q):
            metric = "MAX"
        elif re.search(r"\b(min|minimum|trough|lowest)\b", q):
            metric = "MIN"
        elif re.search(r"\b(total|sum)\b", q):
            metric = "SUM"
        elif re.search(r"\b(count|number of intervals|how many)\b", q):
            metric = "COUNT"
        elif re.search(r"\b(avg|average|mean)\b", q):
            metric = "AVG"

        # 2. Determine Variable
        variable = "wdLoad"
        if any(w in q for w in ["temp", "temperature", "heat", "degree"]):
            variable = "temp"
        elif any(w in q for w in ["humidity", "rh", "moisture"]):
            variable = "rh"

        # 3. Determine Seasons
        seasons_found = []
        season_positions = []
        for s_name, words in [
            ("Summer", ["summer"]),
            ("Monsoon", ["monsoon", "rainy"]),
            ("Autumn", ["autumn", "fall"]),
            ("Winter", ["winter"])
        ]:
            for w in words:
                pos = q.find(w)
                if pos != -1:
                    season_positions.append((pos, s_name))
                    break
        season_positions.sort(key=lambda x: x[0])
        for _, s_name in season_positions:
            if s_name not in seasons_found:
                seasons_found.append(s_name)

        # 4. Calendar & Activity Filters
        festival = 1 if "festival" in q or "festive" in q or "diwali" in q else None
        holiday = 1 if "holiday" in q else None
        weekend = 1 if "weekend" in q or "saturday" in q or "sunday" in q else None
        if "weekday" in q or "working day" in q:
            weekend = 0

        office_hours = 1 if "office hour" in q or "work hours" in q or "business hours" in q else None

        # 5. Extract Specific Dates
        month_map = {
            "jan": "01", "feb": "02", "mar": "03", "apr": "04",
            "may": "05", "jun": "06", "jul": "07", "aug": "08",
            "sep": "09", "oct": "10", "nov": "11", "dec": "12"
        }
        all_dates: List[str] = []

        # Find ISO dates
        for m in re.finditer(r"\b(202[3-5]-\d{2}-\d{2})\b", question):
            d = m.group(1)
            if d not in all_dates:
                all_dates.append(d)

        # Find text dates like "15 Oct 2024" or "15th October 2024"
        for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?:\s+(202[3-5]))?\b", q):
            day_num = m.group(1).zfill(2)
            mon_str = month_map.get(m.group(2)[:3])
            yr_str = m.group(3) or "2024"
            if mon_str:
                d = f"{yr_str}-{mon_str}-{day_num}"
                if d not in all_dates:
                    all_dates.append(d)

        # Also find "October 15, 2024"
        for m in re.finditer(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+(202[3-5]))?\b", q):
            day_num = m.group(2).zfill(2)
            mon_str = month_map.get(m.group(1)[:3])
            yr_str = m.group(3) or "2024"
            if mon_str:
                d = f"{yr_str}-{mon_str}-{day_num}"
                if d not in all_dates:
                    all_dates.append(d)

        # 6. Check for Comparative Intent
        is_compare_query = any(w in q for w in ["compare", "comparison", "difference between", "differ", " vs ", " versus ", "than"])
        comparison_type = "none"
        compare_dates = None
        compare_seasons = None
        date_val = None
        season_val = None

        if len(all_dates) >= 2 and (is_compare_query or " and " in q or " to " in q or " vs " in q):
            comparison_type = "dates"
            compare_dates = all_dates[:2]
            date_val = all_dates[0]
        elif len(seasons_found) >= 2 and (is_compare_query or " and " in q or " vs " in q or "between" in q):
            comparison_type = "seasons"
            compare_seasons = seasons_found[:2]
            season_val = seasons_found[0]
        elif len(all_dates) == 1:
            date_val = all_dates[0]
        elif seasons_found:
            season_val = seasons_found[0]

        # If date is found and season is None, infer season from date
        if date_val and not season_val:
            m_num = int(date_val.split("-")[1])
            if m_num in [12, 1, 2]:
                season_val = "Winter"
            elif m_num in [3, 4, 5]:
                season_val = "Summer"
            elif m_num in [6, 7, 8, 9]:
                season_val = "Monsoon"
            elif m_num in [10, 11]:
                season_val = "Autumn"

        # 7. Year
        year_match = re.search(r"\b(202[3-5])\b", question)
        year_val = int(year_match.group(1)) if year_match else (int(date_val.split("-")[0]) if date_val else 2024)

        # 8. Grouping
        group_by = None
        if "by hour" in q or "hourly" in q or "each hour" in q:
            group_by = "hour"
        elif "by date" in q or "daily" in q or "each day" in q:
            group_by = "date"
        elif "by season" in q or "seasonally" in q:
            group_by = "season"
        elif "by day of week" in q:
            group_by = "day_of_week"

        # 9. Hour filter (e.g. "at 2 pm", "at 14:00", "at 3 pm", "15:00")
        hour_val = None
        hour_match = re.search(r"\bat\s+(\d{1,2})(?::00)?\s*(am|pm)?\b", q)
        if not hour_match:
            hour_match = re.search(r"\b(\d{1,2})\s*(am|pm)\b", q)
        if hour_match:
            h = int(hour_match.group(1))
            ampm = hour_match.group(2)
            if ampm == "pm" and h < 12:
                h += 12
            elif ampm == "am" and h == 12:
                h = 0
            if 0 <= h <= 23:
                hour_val = h

        return QueryIntent(
            metric=metric,
            variable=variable,
            season=season_val,
            year=year_val,
            date=date_val,
            hour=hour_val,
            weekend=weekend,
            holiday=holiday,
            festival=festival,
            office_hours=office_hours,
            group_by=group_by,
            comparison_type=comparison_type,
            compare_dates=compare_dates,
            compare_seasons=compare_seasons
        )

    def build_safe_query(self, intent: QueryIntent) -> Tuple[str, List[Any]]:
        """
        Builds a safe, parameterized SQL query from validated intent.
        Strictly guarantees SELECT only, with no injection vectors.
        """
        metric = intent.metric
        var = intent.variable

        # Handle Date Comparison
        if intent.comparison_type == "dates" and intent.compare_dates:
            sql = f"""
            SELECT date, ROUND({metric}({var}), 2) as aggregate_value, ROUND(AVG(temp), 2) as mean_temp, ROUND(AVG(rh), 2) as mean_rh, COUNT(*) as sample_count
            FROM load_data
            WHERE date IN (?, ?)
            GROUP BY date
            ORDER BY date ASC
            """
            return " ".join(sql.split()), [intent.compare_dates[0], intent.compare_dates[1]]

        # Handle Season Comparison
        if intent.comparison_type == "seasons" and intent.compare_seasons:
            sql = f"""
            SELECT season, ROUND({metric}({var}), 2) as aggregate_value, ROUND(AVG(temp), 2) as mean_temp, ROUND(AVG(rh), 2) as mean_rh, COUNT(*) as sample_count
            FROM load_data
            WHERE season IN (?, ?) AND year = ?
            GROUP BY season
            ORDER BY season ASC
            """
            return " ".join(sql.split()), [intent.compare_seasons[0], intent.compare_seasons[1], intent.year or 2024]

        where_clauses = []
        params = []

        if intent.date:
            where_clauses.append("date = ?")
            params.append(intent.date)

        if intent.season:
            where_clauses.append("season = ?")
            params.append(intent.season)

        if intent.year:
            where_clauses.append("year = ?")
            params.append(intent.year)

        if intent.hour is not None:
            where_clauses.append("hour = ?")
            params.append(intent.hour)

        if intent.weekend is not None:
            where_clauses.append("weekend = ?")
            params.append(intent.weekend)

        if intent.holiday is not None:
            where_clauses.append("holiday = ?")
            params.append(intent.holiday)

        if intent.festival is not None:
            where_clauses.append("festival = ?")
            params.append(intent.festival)

        if intent.office_hours is not None:
            where_clauses.append("office_hours = ?")
            params.append(intent.office_hours)

        where_stmt = " AND ".join(where_clauses)
        if where_stmt:
            where_stmt = "WHERE " + where_stmt

        if intent.group_by:
            grp = intent.group_by
            sql = f"""
            SELECT {grp}, ROUND({metric}({var}), 2) as aggregate_value, COUNT(*) as sample_count
            FROM load_data
            {where_stmt}
            GROUP BY {grp}
            ORDER BY {grp} ASC
            """
        else:
            sql = f"""
            SELECT ROUND({metric}({var}), 2) as aggregate_value, COUNT(*) as sample_count
            FROM load_data
            {where_stmt}
            """

        # Clean excess indentation/spaces
        sql = " ".join(sql.split())
        return sql, params

    def execute_query(self, question: str) -> Dict[str, Any]:
        """Processes an analytical question end-to-end and returns structured results."""
        intent = self.extract_intent(question)
        sql, params = self.build_safe_query(intent)

        unit_map = {
            "wdLoad": "MW",
            "temp": "°C",
            "rh": "%"
        }

        with get_db_connection(self.db_path) as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

        if intent.group_by or intent.comparison_type in ["dates", "seasons"]:
            results = [dict(r) for r in rows]
            summary_val = None
            total_samples = sum(r["sample_count"] for r in rows)
        else:
            row = rows[0] if rows else None
            summary_val = row["aggregate_value"] if row else None
            total_samples = row["sample_count"] if row else 0
            results = [{"aggregate_value": summary_val, "sample_count": total_samples}]

        return {
            "question": question,
            "intent": intent.model_dump(),
            "sql": sql,
            "params": params,
            "value": summary_val,
            "unit": unit_map.get(intent.variable, ""),
            "sample_count": total_samples,
            "records": results
        }

