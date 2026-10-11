import { DecimalPipe } from '@angular/common';
import { HttpErrorResponse } from '@angular/common/http';
import { Component, ElementRef, computed, inject, input, signal } from '@angular/core';
import { takeUntilDestroyed, toObservable, toSignal } from '@angular/core/rxjs-interop';
import { RouterLink } from '@angular/router';
import { exhaustMap, switchMap, takeWhile, timer } from 'rxjs';

import { ContractsService } from '../contracts.service';
import { Segment, buildSegments } from '../highlight';
import { ContractDetail, STATUS_LABELS } from '../models';

const POLL_INTERVAL_MS = 2000;

@Component({
  selector: 'app-contract-detail',
  standalone: true,
  imports: [DecimalPipe, RouterLink],
  templateUrl: './contract-detail.component.html',
  styleUrl: './contract-detail.component.css',
})
export class ContractDetailComponent {
  /** Bound from the route parameter :id (withComponentInputBinding). */
  readonly id = input.required<string>();

  private readonly service = inject(ContractsService);
  private readonly host: ElementRef<HTMLElement> = inject(ElementRef);

  readonly statusLabels = STATUS_LABELS;
  readonly contract = signal<ContractDetail | null>(null);
  readonly loadError = signal<string | null>(null);
  readonly selected = signal<string | null>(null);
  readonly clauseTypes = toSignal(this.service.clauseTypes(), { initialValue: [] });

  /** One row per clause type: found ones first (catalogue order), then absent ones. */
  readonly rows = computed(() => {
    const findings = this.contract()?.findings ?? [];
    const rows = this.clauseTypes().map((type) => {
      const own = findings.filter((f) => f.clause_type === type.name);
      return { ...type, verified: own.filter((f) => f.verified), rejected: own.filter((f) => !f.verified) };
    });
    return [...rows.filter((r) => r.verified.length > 0), ...rows.filter((r) => r.verified.length === 0)];
  });

  readonly selectedRow = computed(() => this.rows().find((r) => r.name === this.selected()) ?? null);

  readonly segments = computed(() => {
    const contract = this.contract();
    return contract?.status === 'done' ? buildSegments(contract.text, contract.findings) : [];
  });

  constructor() {
    // Poll every 2 s while the analysis runs; the final emission (done/failed) is kept.
    // exhaustMap skips a tick if the previous request has not answered yet.
    toObservable(this.id)
      .pipe(
        switchMap((id) =>
          timer(0, POLL_INTERVAL_MS).pipe(
            exhaustMap(() => this.service.get(id)),
            takeWhile((contract) => contract.status === 'processing', true),
          ),
        ),
        takeUntilDestroyed(),
      )
      .subscribe({
        next: (contract) => this.contract.set(contract),
        error: (err: HttpErrorResponse) =>
          this.loadError.set(err.status === 404 ? 'Contrat introuvable.' : 'API injoignable.'),
      });
  }

  select(clauseType: string): void {
    this.selected.set(clauseType);
    const index = this.segments().findIndex((s) => s.clauseTypes.includes(clauseType));
    if (index >= 0) {
      this.host.nativeElement
        .querySelector(`#seg-${index}`)
        ?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  isActive(segment: Segment): boolean {
    const selected = this.selected();
    return selected !== null && segment.clauseTypes.includes(selected);
  }
}
