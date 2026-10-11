"""
The 41 clause categories reviewed by ClauseIQ.

Names and descriptions come verbatim from the CUAD dataset (Hendrycks et al., 2021),
where they are the instructions given to the lawyers who annotated the contracts.
This module is the single source of truth: the LLM prompt, the output schema
and the frontend (via GET /api/clause-types) all read from it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ClauseType:
    name: str
    description: str


CLAUSE_TYPES: tuple[ClauseType, ...] = (
    ClauseType('Document Name', 'The name of the contract'),
    ClauseType('Parties', 'The two or more parties who signed the contract'),
    ClauseType('Agreement Date', 'The date of the contract'),
    ClauseType('Effective Date', 'The date when the contract is effective'),
    ClauseType('Expiration Date', "On what date will the contract's initial term expire?"),
    ClauseType('Renewal Term', 'What is the renewal term after the initial term expires? This includes automatic extensions and unilateral extensions with prior notice.'),
    ClauseType('Notice Period To Terminate Renewal', 'What is the notice period required to terminate renewal?'),
    ClauseType('Governing Law', "Which state/country's law governs the interpretation of the contract?"),
    ClauseType('Most Favored Nation', 'Is there a clause that if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms?'),
    ClauseType('Non-Compete', 'Is there a restriction on the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector?'),
    ClauseType('Exclusivity', 'Is there an exclusive dealing commitment with the counterparty? This includes a commitment to procure all “requirements” from one party of certain technology, goods, or services or a prohibition on licensing or selling technology, goods or services to third parties, or a prohibition on collaborating or working with other parties), whether during the contract or after the contract ends (or both).'),
    ClauseType('No-Solicit Of Customers', 'Is a party restricted from contracting or soliciting customers or partners of the counterparty, whether during the contract or after the contract ends (or both)?'),
    ClauseType('Competitive Restriction Exception', 'This category includes the exceptions or carveouts to Non-Compete, Exclusivity and No-Solicit of Customers above.'),
    ClauseType('No-Solicit Of Employees', 'Is there a restriction on a party’s soliciting or hiring employees and/or contractors from the counterparty, whether during the contract or after the contract ends (or both)?'),
    ClauseType('Non-Disparagement', 'Is there a requirement on a party not to disparage the counterparty?'),
    ClauseType('Termination For Convenience', 'Can a party terminate this contract without cause (solely by giving a notice and allowing a waiting period to expire)?'),
    ClauseType('Rofr/Rofo/Rofn', 'Is there a clause granting one party a right of first refusal, right of first offer or right of first negotiation to purchase, license, market, or distribute equity interest, technology, assets, products or services?'),
    ClauseType('Change Of Control', 'Does one party have the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law?'),
    ClauseType('Anti-Assignment', 'Is consent or notice required of a party if the contract is assigned to a third party?'),
    ClauseType('Revenue/Profit Sharing', 'Is one party required to share revenue or profit with the counterparty for any technology, goods, or services?'),
    ClauseType('Price Restrictions', 'Is there a restriction on the ability of a party to raise or reduce prices of technology, goods, or services provided?'),
    ClauseType('Minimum Commitment', 'Is there a minimum order size or minimum amount or units per-time period that one party must buy from the counterparty under the contract?'),
    ClauseType('Volume Restriction', 'Is there a fee increase or consent requirement, etc. if one party’s use of the product/services exceeds certain threshold?'),
    ClauseType('Ip Ownership Assignment', 'Does intellectual property created by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events?'),
    ClauseType('Joint Ip Ownership', 'Is there any clause providing for joint or shared ownership of intellectual property between the parties to the contract?'),
    ClauseType('License Grant', 'Does the contract contain a license granted by one party to its counterparty?'),
    ClauseType('Non-Transferable License', 'Does the contract limit the ability of a party to transfer the license being granted to a third party?'),
    ClauseType('Affiliate License-Licensor', 'Does the contract contain a license grant by affiliates of the licensor or that includes intellectual property of affiliates of the licensor?'),
    ClauseType('Affiliate License-Licensee', 'Does the contract contain a license grant to a licensee (incl. sublicensor) and the affiliates of such licensee/sublicensor?'),
    ClauseType('Unlimited/All-You-Can-Eat-License', 'Is there a clause granting one party an “enterprise,” “all you can eat” or unlimited usage license?'),
    ClauseType('Irrevocable Or Perpetual License', 'Does the contract contain a license grant that is irrevocable or perpetual?'),
    ClauseType('Source Code Escrow', 'Is one party required to deposit its source code into escrow with a third party, which can be released to the counterparty upon the occurrence of certain events (bankruptcy, insolvency, etc.)?'),
    ClauseType('Post-Termination Services', 'Is a party subject to obligations after the termination or expiration of a contract, including any post-termination transition, payment, transfer of IP, wind-down, last-buy, or similar commitments?'),
    ClauseType('Audit Rights', 'Does a party have the right to audit the books, records, or physical locations of the counterparty to ensure compliance with the contract?'),
    ClauseType('Uncapped Liability', 'Is a party’s liability uncapped upon the breach of its obligation in the contract? This also includes uncap liability for a particular type of breach such as IP infringement or breach of confidentiality obligation.'),
    ClauseType('Cap On Liability', 'Does the contract include a cap on liability upon the breach of a party’s obligation? This includes time limitation for the counterparty to bring claims or maximum amount for recovery.'),
    ClauseType('Liquidated Damages', 'Does the contract contain a clause that would award either party liquidated damages for breach or a fee upon the termination of a contract (termination fee)?'),
    ClauseType('Warranty Duration', 'What is the duration of any warranty against defects or errors in technology, products, or services provided under the contract?'),
    ClauseType('Insurance', 'Is there a requirement for insurance that must be maintained by one party for the benefit of the counterparty?'),
    ClauseType('Covenant Not To Sue', 'Is a party restricted from contesting the validity of the counterparty’s ownership of intellectual property or otherwise bringing a claim against the counterparty for matters unrelated to the contract?'),
    ClauseType('Third Party Beneficiary', 'Is there a non-contracting party who is a beneficiary to some or all of the clauses in the contract and therefore can enforce its rights against a contracting party?'),
)

CLAUSE_NAMES: tuple[str, ...] = tuple(c.name for c in CLAUSE_TYPES)
