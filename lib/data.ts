export const q4GroundTruth = [
  ["ON", "11:36", "26", "24", "HOU 29-15 POR"],
  ["OFF", "0:24", "0", "2", "HOU 0-2 POR"],
];

export const q4LineupLocks = [
  ["A.Thompson, R.Sheppard, K.Durant, T.Eason, C.Capela", "4:46", "11", "9", "HOU 12-8 POR"],
  ["A.Thompson, A.Sengun, R.Sheppard, D.Finney-Smith, T.Eason", "3:31", "8", "8", "HOU 8-7 POR"],
  ["R.Sheppard, K.Durant, T.Eason, J.Okogie, C.Capela", "2:00", "4", "4", "HOU 2-0 POR"],
  ["A.Thompson, R.Sheppard, K.Durant, J.Okogie, C.Capela", "1:19", "3", "3", "HOU 7-0 POR"],
  ["A.Thompson, K.Durant, T.Eason, J.Okogie, C.Capela", "0:24", "0", "2", "HOU 0-2 POR"],
];

export const resolvedWindows = [
  ["A.Thompson, A.Sengun, R.Sheppard, D.Finney-Smith, T.Eason", "Q4 12:00", "Q4 8:29", "3:31", "8", "8", "8", "7", "q4-1", "standalone_real"],
  ["A.Thompson, R.Sheppard, K.Durant, T.Eason, C.Capela", "Q4 8:29", "Q4 3:41", "4:48", "11", "9", "12", "8", "q4-2", "needs boundary trim to 4:46"],
  ["A.Thompson, K.Durant, T.Eason, J.Okogie, C.Capela", "Q4 3:41", "Q4 3:05", "0:36", "0", "2", "0", "2", "q4-3,q4-4,q4-5", "needs boundary trim to 0:24"],
  ["A.Thompson, R.Sheppard, K.Durant, J.Okogie, C.Capela", "Q4 7:00", "Q4 5:41", "1:19", "3", "3", "7", "0", "q4-2a", "standalone_real"],
  ["R.Sheppard, K.Durant, T.Eason, J.Okogie, C.Capela", "Q4 5:05", "Q4 3:05", "2:00", "4", "4", "2", "0", "q4-2b,q4-5b", "standalone_real across same-context bridge"],
];

export const comparisonRows = [
  ["A.Thompson, R.Sheppard, K.Durant, T.Eason, C.Capela", "4:48", "4:46", "needs trim", "11", "11", "0", "9", "9", "0", "HOU 12-8 POR", "HOU 12-8 POR"],
  ["A.Thompson, A.Sengun, R.Sheppard, D.Finney-Smith, T.Eason", "3:31", "3:31", "0:00", "8", "8", "0", "8", "8", "0", "HOU 8-7 POR", "HOU 8-7 POR"],
  ["R.Sheppard, K.Durant, T.Eason, J.Okogie, C.Capela", "2:00", "2:00", "0:00", "4", "4", "0", "4", "4", "0", "HOU 2-0 POR", "HOU 2-0 POR"],
  ["A.Thompson, R.Sheppard, K.Durant, J.Okogie, C.Capela", "1:19", "1:19", "0:00", "3", "3", "0", "3", "3", "0", "HOU 7-0 POR", "HOU 7-0 POR"],
  ["A.Thompson, K.Durant, T.Eason, J.Okogie, C.Capela", "0:36", "0:24", "needs trim", "0", "0", "0", "2", "2", "0", "HOU 0-2 POR", "HOU 0-2 POR"],
];

export const firstWrongPossession = [
  ["Q4-anchor-001", "4", "Q4 3:41", "Q4 3:05", "POR", "A.Thompson, K.Durant, T.Eason, J.Okogie, C.Capela", "A.Thompson, K.Durant, T.Eason, J.Okogie, C.Capela", "A.Thompson, R.Sheppard, K.Durant, T.Eason, C.Capela", "A.Thompson, R.Sheppard, K.Durant, J.Okogie, C.Capela", "Correct lineup and possession totals; raw sub boundaries still overstate display duration by 0:12."]
];

export const v12Rules = [
  "Add displayStart/displayEnd alongside rawStart/rawEnd for every resolved lineup window.",
  "Trim displayStart to the first counted possession or control boundary owned by that lineup.",
  "Trim displayEnd to the last counted possession or control boundary owned by that lineup.",
  "Do not alter lineup identity, score, or Off/Def counts in this pass.",
  "Keep raw sub window values visible for audit, but benchmark-facing minutes must use trimmed display bounds.",
];
