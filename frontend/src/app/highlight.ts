import { Finding } from './models';

/** A run of contract text covered by the same set of clause types (empty set = plain text). */
export interface Segment {
  text: string;
  clauseTypes: string[];
}

/**
 * Cuts the contract text at every start/end offset of the verified findings, so each
 * segment can be rendered as plain text or as a highlight. Overlapping excerpts are
 * supported: a segment lists every clause type that covers it.
 */
export function buildSegments(text: string, findings: Finding[]): Segment[] {
  const spans = findings.flatMap((f) =>
    f.verified && f.start_char !== null && f.end_char !== null
      ? [{ start: f.start_char, end: f.end_char, clauseType: f.clause_type }]
      : [],
  );

  const cuts = new Set<number>([0, text.length]);
  for (const span of spans) {
    cuts.add(span.start);
    cuts.add(span.end);
  }
  const points = [...cuts].sort((a, b) => a - b);

  const segments: Segment[] = [];
  for (let i = 0; i < points.length - 1; i++) {
    const [start, end] = [points[i], points[i + 1]];
    const clauseTypes = [
      ...new Set(spans.filter((s) => s.start <= start && s.end >= end).map((s) => s.clauseType)),
    ];
    const previous = segments.at(-1);
    if (previous && sameMembers(previous.clauseTypes, clauseTypes)) {
      previous.text += text.slice(start, end);
    } else {
      segments.push({ text: text.slice(start, end), clauseTypes });
    }
  }
  return segments;
}

function sameMembers(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((x) => b.includes(x));
}
