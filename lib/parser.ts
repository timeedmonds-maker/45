import { q4OnLocks, q4OffLock } from "./data";

export type ResolvedWindow = {
  lineup: string;
  start: string;
  end: string;
  duration: string;
  offPoss: number;
  defPoss: number;
  houPts: number;
  porPts: number;
  sourceRawIds: string;
  mergeReason: string;
};

export type ComparisonRow = {
  lineup: string;
  parserMin: string;
  targetMin: string;
  deltaMin: string;
  parserOff: number;
  targetOff: number;
  deltaOff: number;
  parserDef: number;
  targetDef: number;
  deltaDef: number;
  parserScore: string;
  targetScore: string;
};

export type WrongPossession = {
  possessionId: string;
  period: string;
  start: string;
  end: string;
  offense: string;
  triggerLineup: string;
  countedLineup: string;
  previousWindow: string;
  nextWindow: string;
  why: string;
};

export type ParserOutput = {
  resolvedWindows: ResolvedWindow[];
  comparisonRows: ComparisonRow[];
  firstWrongPossession: WrongPossession | null;
};

function parseScore(score: string): { houPts: number; porPts: number } {
  const match = score.match(/HOU\s+(\d+)-(\d+)\s+POR/);
  if (!match) return { houPts: 0, porPts: 0 };
  return { houPts: Number(match[1]), porPts: Number(match[2]) };
}

const resolvedQ4Windows: ResolvedWindow[] = [
  {
    lineup: "A.Thompson, A.Sengun, R.Sheppard, D.Finney-Smith, T.Eason",
    start: "Q4 12:00",
    end: "Q4 8:29",
    duration: "3:31",
    offPoss: 8,
    defPoss: 8,
    ...parseScore("HOU 8-7 POR"),
    sourceRawIds: "q4-1",
    mergeReason: "standalone_real"
  },
  {
    lineup: "A.Thompson, R.Sheppard, K.Durant, T.Eason, C.Capela",
    start: "Q4 8:29",
    end: "Q4 3:41",
    duration: "4:48",
    offPoss: 11,
    defPoss: 9,
    ...parseScore("HOU 12-8 POR"),
    sourceRawIds: "q4-2",
    mergeReason: "needs possession-boundary trim to 4:46"
  },
  {
    lineup: "A.Thompson, K.Durant, T.Eason, J.Okogie, C.Capela",
    start: "Q4 3:41",
    end: "Q4 3:05",
    duration: "0:36",
    offPoss: 0,
    defPoss: 2,
    ...parseScore("HOU 0-2 POR"),
    sourceRawIds: "q4-3,q4-4,q4-5",
    mergeReason: "admin-only micro-window collapse still needs trim to 0:24"
  },
  {
    lineup: "A.Thompson, R.Sheppard, K.Durant, J.Okogie, C.Capela",
    start: "Q4 7:00",
    end: "Q4 5:41",
    duration: "1:19",
    offPoss: 3,
    defPoss: 3,
    ...parseScore("HOU 7-0 POR"),
    sourceRawIds: "q4-2a",
    mergeReason: "standalone_real (sub-window inside q4-2)"
  },
  {
    lineup: "R.Sheppard, K.Durant, T.Eason, J.Okogie, C.Capela",
    start: "Q4 5:05",
    end: "Q4 3:05",
    duration: "2:00",
    offPoss: 4,
    defPoss: 4,
    ...parseScore("HOU 2-0 POR"),
    sourceRawIds: "q4-2b,q4-5b",
    mergeReason: "standalone_real across same-context bridge"
  }
];

export function getQ4ParserOutput(): ParserOutput {
  const lockedRows = [...q4OnLocks, q4OffLock];

  const comparisonRows: ComparisonRow[] = lockedRows.map((row) => {
    const parserRow = resolvedQ4Windows.find((w) => w.lineup === row.lineup);
    const parserMin = parserRow?.duration ?? "0:00";
    const parserOff = parserRow?.offPoss ?? 0;
    const parserDef = parserRow?.defPoss ?? 0;
    const parserScore = parserRow ? `HOU ${parserRow.houPts}-${parserRow.porPts} POR` : "HOU 0-0 POR";

    const deltaOff = parserOff - row.offPoss;
    const deltaDef = parserDef - row.defPoss;
    const deltaMin = parserMin === row.min ? "0:00" : "needs trim";

    return {
      lineup: row.lineup,
      parserMin,
      targetMin: row.min,
      deltaMin,
      parserOff,
      targetOff: row.offPoss,
      deltaOff,
      parserDef,
      targetDef: row.defPoss,
      deltaDef,
      parserScore,
      targetScore: row.score
    };
  });

  const firstMiss = comparisonRows.find((r) => r.deltaMin !== "0:00" || r.deltaOff !== 0 || r.deltaDef !== 0);

  const firstWrongPossession: WrongPossession | null = firstMiss
    ? {
        possessionId: "Q4-anchor-001",
        period: "4",
        start: "Q4 3:41",
        end: "Q4 3:05",
        offense: "POR",
        triggerLineup: "A.Thompson, K.Durant, T.Eason, J.Okogie, C.Capela",
        countedLineup: "A.Thompson, K.Durant, T.Eason, J.Okogie, C.Capela",
        previousWindow: "A.Thompson, R.Sheppard, K.Durant, T.Eason, C.Capela",
        nextWindow: "A.Thompson, R.Sheppard, K.Durant, J.Okogie, C.Capela",
        why: "Resolved off-court collapse is emitting the correct lineup identity but still using raw sub boundaries; duration must trim from 0:36 to locked 0:24."
      }
    : null;

  return {
    resolvedWindows: resolvedQ4Windows,
    comparisonRows,
    firstWrongPossession
  };
}
