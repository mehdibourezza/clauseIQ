import { DatePipe } from '@angular/common';
import { HttpErrorResponse } from '@angular/common/http';
import { Component, OnInit, inject, signal } from '@angular/core';
import { Router, RouterLink } from '@angular/router';

import { ContractsService } from '../contracts.service';
import { ContractSummary, STATUS_LABELS } from '../models';

@Component({
  selector: 'app-contract-list',
  standalone: true,
  imports: [DatePipe, RouterLink],
  templateUrl: './contract-list.component.html',
  styleUrl: './contract-list.component.css',
})
export class ContractListComponent implements OnInit {
  private readonly service = inject(ContractsService);
  private readonly router = inject(Router);

  readonly statusLabels = STATUS_LABELS;
  readonly contracts = signal<ContractSummary[]>([]);
  readonly selectedFile = signal<File | null>(null);
  readonly uploading = signal(false);
  readonly error = signal<string | null>(null);

  ngOnInit(): void {
    this.service.list().subscribe({
      next: (contracts) => this.contracts.set(contracts),
      error: () => this.error.set("API injoignable : le backend tourne-t-il sur le port 8000 ?"),
    });
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.selectedFile.set(input.files?.[0] ?? null);
    this.error.set(null);
  }

  upload(): void {
    const file = this.selectedFile();
    if (!file) return;

    this.uploading.set(true);
    this.service.upload(file).subscribe({
      next: (contract) => this.router.navigate(['/contracts', contract.id]),
      error: (err: HttpErrorResponse) => {
        this.error.set(err.error?.detail ?? 'Échec du dépôt.');
        this.uploading.set(false);
      },
    });
  }
}
