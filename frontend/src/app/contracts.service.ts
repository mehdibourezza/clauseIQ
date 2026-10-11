import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { ClauseType, ContractDetail, ContractSummary } from './models';

// Relative URL: in development the Angular dev server proxies /api to FastAPI (proxy.conf.json).
const API = '/api/v1';

@Injectable({ providedIn: 'root' })
export class ContractsService {
  private readonly http = inject(HttpClient);

  list(): Observable<ContractSummary[]> {
    return this.http.get<ContractSummary[]>(`${API}/contracts`);
  }

  get(id: string): Observable<ContractDetail> {
    return this.http.get<ContractDetail>(`${API}/contracts/${id}`);
  }

  upload(file: File): Observable<ContractSummary> {
    const form = new FormData();
    form.append('file', file);
    return this.http.post<ContractSummary>(`${API}/contracts`, form);
  }

  clauseTypes(): Observable<ClauseType[]> {
    return this.http.get<ClauseType[]>(`${API}/clause-types`);
  }
}
