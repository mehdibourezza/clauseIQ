import { Routes } from '@angular/router';

import { ContractDetailComponent } from './contract-detail/contract-detail.component';
import { ContractListComponent } from './contract-list/contract-list.component';

export const routes: Routes = [
  { path: '', component: ContractListComponent, title: 'ClauseIQ · Contrats' },
  { path: 'contracts/:id', component: ContractDetailComponent, title: 'ClauseIQ · Analyse' },
  { path: '**', redirectTo: '' },
];
