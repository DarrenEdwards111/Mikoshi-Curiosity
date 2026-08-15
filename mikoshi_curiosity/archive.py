"""Persistent SQLite memory for research candidates and failures."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List

from mikoshi_curiosity.research import CandidateEvaluation, Conjecture


class ResearchArchive:
    def __init__(self, path):
        self.path = str(Path(path))
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS candidates (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, statement TEXT NOT NULL,
          text TEXT NOT NULL, conjecture_json TEXT NOT NULL,
          verdict TEXT NOT NULL, score REAL NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS critiques (
          candidate_id TEXT NOT NULL, critic TEXT NOT NULL, severity TEXT NOT NULL,
          message TEXT NOT NULL, evidence TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS proofs (
          candidate_id TEXT NOT NULL, adapter TEXT NOT NULL, status TEXT NOT NULL,
          output TEXT NOT NULL, artifact TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_candidates_verdict ON candidates(verdict);
        """)
        self.connection.commit()

    def save(self, conjecture: Conjecture, evaluation: CandidateEvaluation) -> None:
        data = {
            "name": conjecture.name, "statement": conjecture.statement,
            "definitions": conjecture.definitions, "assumptions": conjecture.assumptions,
            "proof_sketch": conjecture.proof_sketch, "tags": conjecture.tags,
            "provenance": conjecture.provenance, "parent_id": conjecture.parent_id,
            "generation": conjecture.generation,
        }
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO candidates(id,name,statement,text,conjecture_json,verdict,score) VALUES(?,?,?,?,?,?,?)",
                (conjecture.id, conjecture.name, conjecture.statement, conjecture.text,
                 json.dumps(data), evaluation.verdict, evaluation.score),
            )
            self.connection.execute("DELETE FROM critiques WHERE candidate_id=?", (conjecture.id,))
            self.connection.execute("DELETE FROM proofs WHERE candidate_id=?", (conjecture.id,))
            self.connection.executemany(
                "INSERT INTO critiques VALUES(?,?,?,?,?)",
                [(conjecture.id, x.critic, x.severity, x.message, x.evidence) for x in evaluation.critiques],
            )
            self.connection.executemany(
                "INSERT INTO proofs VALUES(?,?,?,?,?)",
                [(conjecture.id, x.adapter, x.status, x.output, x.artifact) for x in evaluation.proofs],
            )

    def similar_failures(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        tokens = {token.lower() for token in query.split() if len(token) >= 4}
        rows = self.connection.execute(
            "SELECT id,name,text FROM candidates WHERE verdict IN ('rejected','falsified') ORDER BY created_at DESC"
        ).fetchall()
        scored = []
        for row in rows:
            overlap = len(tokens.intersection(row["text"].lower().split()))
            reason_row = self.connection.execute(
                "SELECT message FROM critiques WHERE candidate_id=? ORDER BY severity DESC LIMIT 1", (row["id"],)
            ).fetchone()
            scored.append((overlap, {"id": row["id"], "name": row["name"],
                                     "reason": reason_row["message"] if reason_row else "formal check failed"}))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:max(0, limit)]]

    def count(self, verdict: str = "") -> int:
        if verdict:
            return int(self.connection.execute("SELECT COUNT(*) FROM candidates WHERE verdict=?", (verdict,)).fetchone()[0])
        return int(self.connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
