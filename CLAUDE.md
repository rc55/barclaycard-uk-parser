# CLAUDE.md - Project Context for AI Assistance

## Project Overview

**Barclaycard UK Statement Parser** — Extracts transactions from Barclaycard PDF statements into CSV/JSON format with automatic validation.

Two implementations:
- `barclaycard_parser.py` — Python CLI tool for local/batch processing
- `barclaycard_parser.html` — Browser-based tool (100% client-side, no server)

## Key Technical Details

### PDF Layout
- Barclaycard statements use a **two-column layout** on transaction pages
- Column split is at **x=295px**
- "Ways to pay" box appears on page 2 (usually) at **y≈570** — must be excluded from parsing
- Transactions span pages 2-3+ depending on volume

### Transaction Format
```
DD Mon Description £Amount[CR]
```
- Date and description often have no space between them in extracted text
- `CR` suffix indicates credit (refund or payment)
- Foreign transactions have continuation lines with exchange rate info
- Contactless indicator `e` appears at end of some descriptions

### Payment Types Recognised
- "Payment, Thank You" — manual payments
- "Payment By Direct Debit" — automatic payments

### Validation Logic
Three checks against statement summary figures:
1. **Payments match** — Sum of payment transactions = "Payments towards your account"
2. **New activity match** — (Purchases - Refunds + Interest + Charges) = "Your new activity"
3. **Balance equation** — Previous - Payments + NewActivity = New Balance

### Known Edge Cases Handled
- Credit balances (£X.XX**CR** suffix)
- Duplicate legitimate transactions (allows up to 2 identical)
- Multi-line merchant names
- Foreign currency with exchange rates
- Year boundary handling (transactions in previous month)

## File Structure

```
barclaycard_parser.py   # Python CLI (requires pdfplumber, pandas)
barclaycard_parser.html # Browser version (uses pdf.js from CDN)
README.md               # User documentation
CLAUDE.md               # This file - AI context
```

## Common Modifications

### Adding new payment types
In Python (`barclaycard_parser.py` ~line 213):
```python
if 'Payment' in txn['description'] and ('Thank You' in txn['description'] or 'Direct Debit' in txn['description']):
```

In JavaScript (`barclaycard_parser.html`):
```javascript
if (txn.description.includes('Payment') && 
    (txn.description.includes('Thank You') || txn.description.includes('Direct Debit'))) {
```

### Adjusting column split
If layout changes, modify `column_split = 295` (Python) or `columnSplit = 295` (JS)

### Adjusting line grouping tolerance
Python: `y_key = round(c['top'] / 3) * 3` — change divisor
JavaScript: `const yKey = Math.round(item.y / 5) * 5` — change divisor

## Testing

Validation automatically checks extraction accuracy. If a statement fails:
1. Check which validation check failed and by how much
2. The difference often equals one missing transaction
3. Look for new payment types or layout changes

Tested successfully with statements from 2023-2026 (7+ statements, 400+ transactions).

## Dependencies

### Python
- pdfplumber (PDF text extraction with coordinates)
- pandas (DataFrame/CSV handling)

### JavaScript
- pdf.js v3.11.174 (Mozilla's PDF library, loaded from CDN)

## Repository

https://github.com/rc55/barclaycard-uk-parser
