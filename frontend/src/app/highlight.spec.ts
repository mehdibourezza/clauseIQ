import { buildSegments } from './highlight';
import { Finding } from './models';

function finding(clause_type: string, start: number | null, end: number | null, verified = true): Finding {
  return { clause_type, quote: '', start_char: start, end_char: end, verified };
}

describe('buildSegments', () => {
  const text = 'Governed by Delaware law. Liability capped at $1M.';

  it('returns the whole text as one plain segment when nothing is verified', () => {
    expect(buildSegments(text, [finding('Governing Law', null, null, false)])).toEqual([
      { text, clauseTypes: [] },
    ]);
  });

  it('isolates each verified excerpt and keeps the text intact', () => {
    const segments = buildSegments(text, [finding('Governing Law', 0, 25), finding('Cap On Liability', 26, 50)]);
    expect(segments.map((s) => s.text).join('')).toBe(text);
    expect(segments).toEqual([
      { text: 'Governed by Delaware law.', clauseTypes: ['Governing Law'] },
      { text: ' ', clauseTypes: [] },
      { text: 'Liability capped at $1M.', clauseTypes: ['Cap On Liability'] },
    ]);
  });

  it('lists every clause type on overlapping excerpts', () => {
    const segments = buildSegments(text, [finding('Governing Law', 0, 25), finding('Parties', 12, 20)]);
    expect(segments.find((s) => s.text === 'Delaware')?.clauseTypes).toEqual(['Governing Law', 'Parties']);
  });
});
