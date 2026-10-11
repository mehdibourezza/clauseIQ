// Mirrors api/schemas/contracts.py: keep both in sync when the API contract changes.

export type ContractStatus = 'processing' | 'done' | 'failed';

export const STATUS_LABELS: Record<ContractStatus, string> = {
  processing: 'En cours',
  done: 'Terminé',
  failed: 'Échec',
};

export interface ClauseType {
  name: string;
  description: string;
}

export interface Finding {
  clause_type: string;
  quote: string;
  start_char: number | null;
  end_char: number | null;
  verified: boolean;
}

export interface ContractSummary {
  id: string;
  filename: string;
  status: ContractStatus;
  created_at: string;
  completed_at: string | null;
  clauses_found: number;
}

export interface ContractDetail extends ContractSummary {
  text: string;
  error: string | null;
  model: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
  findings: Finding[];
}
