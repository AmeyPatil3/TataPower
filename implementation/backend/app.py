"""
Tata Power Platform API & Web Server.
Pure Python standard library HTTP server (zero external dependency requirements).
Serves REST API endpoints and interactive web dashboard.
"""

import os
import sys
import json
import urllib.parse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from typing import Dict, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
sys.path.insert(0, PROJECT_ROOT)

from backend.db import get_available_dates, get_daily_records
from backend.services.dtw import DTWService
from backend.services.causal import CausalService
from backend.services.xai import XAIService
from backend.services.sql_agent import SafeSQLAgent
from backend.services.explainer import ExplanationOrchestrator

PORT = int(os.environ.get("PORT", 8000))


# Initialize singleton services once globally
dtw_svc = DTWService()
causal_svc = CausalService()
xai_svc = XAIService()
sql_agent = SafeSQLAgent()
explainer = ExplanationOrchestrator()


class TataPowerRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        # 1. Health Endpoint
        if path == "/api/health":
            endpoint = explainer.llm_client.detect_active_endpoint()
            llm_info = f"{endpoint['type']} ({endpoint['url']})" if endpoint else "Local synthesizer active (LLM server offline)"
            self._send_json({
                "status": "healthy",
                "database": "connected (empirical models & 15-min series)",
                "llm_service": llm_info
            })
            return

        # 2. Available Dates
        elif path == "/api/dates":
            season = qs.get("season", [None])[0]
            dates = get_available_dates(season=season)
            self._send_json({"dates": dates, "total": len(dates)})
            return

        # 3. Day Load Curve (96 intervals)
        elif path.startswith("/api/day/"):
            date_str = path.replace("/api/day/", "").strip()
            curve_data = dtw_svc.get_day_curve(date_str)
            if not curve_data:
                self._send_json({"error": "Date not found"}, status=404)
                return
            # Convert numpy arrays to list for JSON serialization
            curve_data["raw_load"] = curve_data["raw_load"].tolist()
            curve_data["normalized_load"] = curve_data["normalized_load"].tolist()
            self._send_json(curve_data)
            return

        # 4. DTW Matches
        elif path.startswith("/api/dtw/"):
            date_str = path.replace("/api/dtw/", "").strip()
            try:
                matches = dtw_svc.find_top_k_similar_days(date_str, k=3, mask_future=True)
                self._send_json({"query_date": date_str, "matches": matches})
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)
            return

        # 5. Causal Effects & Seasonal Hourly HTE
        elif path.startswith("/api/causal/"):
            season = path.replace("/api/causal/", "").capitalize().strip()
            effects = causal_svc.get_season_effects(season)
            self._send_json(effects)
            return

        # 5b. Day-Specific 24-Hour Diurnal HTE (sociowinter_v2.ipynb Cell 13)
        elif path.startswith("/api/hte/day/"):
            date_str = path.replace("/api/hte/day/", "").strip()
            hte_day = causal_svc.get_day_hourly_hte(date_str)
            self._send_json({"date": date_str, "hourly_hte": hte_day})
            return

        # 6. XAI SHAP Features
        elif path.startswith("/api/xai/"):
            season = path.replace("/api/xai/", "").capitalize().strip()
            shap_data = xai_svc.get_global_importance(season)
            self._send_json({"season": season, "shap_features": shap_data})
            return

        # Serve static files (Dashboard UI)
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")
        data = json.loads(body) if body else {}

        # 1. Safe Parameter-to-SQL Query
        if path == "/api/query":
            question = data.get("question", "")
            if not question:
                self._send_json({"error": "Empty question"}, status=400)
                return
            result = sql_agent.execute_query(question)
            self._send_json(result)
            return

        # 2. Grounded Explanation Endpoint (LM Studio gemma-4-e2b)
        elif path == "/api/explain":
            date_str = data.get("date")
            question = data.get("question")
            if not date_str:
                self._send_json({"error": "Missing target date"}, status=400)
                return
            try:
                explanation = explainer.explain(query_date=date_str, user_question=question)
                self._send_json(explanation)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        # 3. Interactive Prompt / Chat Reasoning Endpoint (gemma-4-e2b via LM Studio)
        elif path in ["/api/chat", "/api/prompt"]:
            prompt = data.get("prompt") or data.get("question")
            if not prompt:
                self._send_json({"error": "Missing prompt"}, status=400)
                return
            date_str = data.get("date")
            season = data.get("season")
            try:
                result = explainer.answer_prompt(prompt=prompt, date=date_str, season=season)
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        self._send_json({"error": "Endpoint not found"}, status=404)


def run_server(port: int = PORT):
    server = ThreadingHTTPServer(("0.0.0.0", port), TataPowerRequestHandler)
    print("==================================================")
    print(f"Tata Power Load Analysis Server is RUNNING")
    print(f"Dashboard URL: http://localhost:{port}")
    print(f"LM Studio local endpoint: http://localhost:1234/v1 (gemma-4-e2b)")
    print("==================================================")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
