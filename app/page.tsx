import type { ReactNode } from "react";
import { comparisonRows, firstWrongPossession, q4GroundTruth, q4LineupLocks, resolvedWindows, v12Rules } from "../lib/data";

function Table({ headers, rows }: { headers: string[]; rows: (string | number | ReactNode)[][] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>{headers.map((h) => <th key={h}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Page() {
  return (
    <main>
      <h1>Rockets Live Game Dashboard</h1>
      <div className="section">
        <h2>AUDIT-GRADE CONTROL PARSER V12</h2>
        <p className="muted">Boundary-trim pass. Q4 lineup identities, possession counts, and scores are now treated as correct. This pass should only convert raw sub windows into counted display windows.</p>
        <div className="badges">
          <span className="badge">Q4 only first</span>
          <span className="badge">boundary trim only</span>
          <span className="badge">no possession logic changes</span>
          <span className="badge">displayStart / displayEnd</span>
        </div>
      </div>

      <section className="section">
        <h3>Locked Q4 ground truth</h3>
        <Table headers={["Split", "MIN", "Off Poss", "Def Poss", "Score"]} rows={q4GroundTruth} />
      </section>

      <section className="section">
        <h3>Q4 lineup locks</h3>
        <Table headers={["Lineup", "MIN", "Off Poss", "Def Poss", "Score"]} rows={q4LineupLocks} />
      </section>

      <section className="section">
        <h3>Q4 resolved lineup windows</h3>
        <p className="muted">Raw windows are now correct enough for lineup identity and possession ownership. The remaining job is to trim displayed minutes to counted control boundaries.</p>
        <Table headers={["Lineup", "Start", "End", "Duration", "Off Poss", "Def Poss", "HOU pts", "POR pts", "Source raw ids", "Merge reason"]} rows={resolvedWindows} />
      </section>

      <section className="section">
        <h3>Q4 parser vs locked lineup rows</h3>
        <Table headers={["Lineup", "Parser MIN", "Target MIN", "Δ MIN", "Parser Off", "Target Off", "Δ Off", "Parser Def", "Target Def", "Δ Def", "Parser Score", "Target Score"]} rows={comparisonRows} />
      </section>

      <section className="section">
        <h3>First wrong possession</h3>
        <Table headers={["Possession id", "Period", "Start", "End", "Offense", "Trigger lineup", "Counted lineup", "Previous window", "Next window", "Why parser put it here"]} rows={firstWrongPossession} />
      </section>

      <section className="section">
        <h3>V12 implementation rules</h3>
        <div className="code">{v12Rules.map((r, i) => `${i + 1}. ${r}`).join("\n")}</div>
      </section>
    </main>
  );
}
