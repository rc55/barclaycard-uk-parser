# Barclaycard Statement Parser

A Python tool to extract transactions from Barclaycard PDF statements into CSV or JSON format.

## Features

- **Accurate extraction** from Barclaycard's two-column PDF layout
- **Automatic validation** against statement totals (payments, new activity, balance)
- **Batch processing** for multiple statements
- **Combined output** merging all transactions with source file tracking
- **Foreign currency support** with exchange rate details preserved
- **Transaction categorisation** (purchase, payment, refund)

## Installation

### Requirements

- Python 3.8+
- pdfplumber
- pandas

### Setup

```bash
# Install dependencies
pip install pdfplumber pandas

# Download the parser
# (save barclaycard_parser.py to your working directory)

# Make executable (optional, Linux/Mac)
chmod +x barclaycard_parser.py
```

## Usage

### Basic Usage

```bash
# Parse a single statement (validates and prints summary)
python barclaycard_parser.py statement.pdf

# Save to CSV
python barclaycard_parser.py statement.pdf -o transactions.csv

# Save to JSON (includes metadata and validation results)
python barclaycard_parser.py statement.pdf -o statement.json
```

### Batch Processing

```bash
# Process all PDFs and combine into single CSV
python barclaycard_parser.py *.pdf -o all_transactions.csv

# Process all PDFs and save each separately
python barclaycard_parser.py *.pdf --output-dir ./exports/
```

### Options

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Output file (.csv or .json) |
| `--output-dir DIR` | Save each file separately to directory |
| `--no-validate` | Skip validation checks |
| `-v, --verbose` | Show full validation report |
| `-q, --quiet` | Only show errors |
| `--version` | Show version number |
| `-h, --help` | Show help message |

## Output Format

### CSV Columns

| Column | Description |
|--------|-------------|
| `source_file` | Original PDF filename (batch mode only) |
| `date` | Transaction date (e.g., "04 Nov") |
| `full_date` | Full date with year (e.g., "04 Nov 2025") |
| `description` | Merchant/transaction description |
| `amount` | Transaction amount (always positive) |
| `type` | Transaction type: `purchase`, `payment`, or `refund` |
| `signed_amount` | Amount with sign (negative for payments/refunds) |
| `extra_info` | Additional info (forex details, continuation text) |

### JSON Structure

```json
{
  "metadata": {
    "statement_date": "02 December 2025",
    "previous_balance": 1685.49,
    "new_balance": 1100.59,
    "stated_payments": 3083.44,
    "stated_new_activity": 2498.54,
    "stated_interest": 0.0,
    "stated_charges": 0.0
  },
  "transactions": [...],
  "validation": {
    "valid": true,
    "checks": [...],
    "summary": {...}
  }
}
```

## Validation

The parser automatically validates extracted data against the statement's summary figures:

1. **Payments match** — Total payments vs "Payments towards your account"
2. **New activity match** — (Purchases - Refunds + Interest + Charges) vs "Your new activity"  
3. **Balance equation** — Previous balance - Payments + New activity = New balance

If validation fails, you'll see which check failed and by how much — this usually indicates a parsing issue or unusual transaction format.

```
============================================================
VALIDATION REPORT
============================================================

Overall Status: ✓ PASSED

--- Checks ---
  ✓ Payments: stated £3083.44 vs extracted £3083.44 (diff: £0.00)
  ✓ New activity: stated £2498.54 vs extracted £2498.54 (diff: £0.00)
  ✓ Balance check: calculated £1100.59 vs stated £1100.59 (diff: £0.00)
  ✓ Extracted 86 transactions
```

## Python API

You can also use the parser as a library:

```python
from barclaycard_parser import parse_statement, validate_extraction, to_dataframe

# Parse a statement
result = parse_statement('statement.pdf')

# Access metadata
print(result['metadata']['statement_date'])
print(result['metadata']['new_balance'])

# Access transactions
for txn in result['transactions']:
    print(f"{txn['date']}: {txn['description']} £{txn['amount']}")

# Validate
validation = validate_extraction(result)
if validation['valid']:
    print("All checks passed!")

# Convert to pandas DataFrame
df = to_dataframe(result)
df.to_excel('transactions.xlsx', index=False)
```

## Troubleshooting

### Validation Failed

If validation fails, common causes are:

- **New payment type** — The parser recognises "Payment, Thank You" and "Payment By Direct Debit". Other formats may need adding.
- **Credit balance** — Balances with "CR" suffix (e.g., "£1.54CR") indicate you're in credit.
- **Duplicate transactions** — Legitimate duplicates (e.g., two identical parking charges) are preserved, but parsing artifacts are filtered.

### Missing Transactions

If transactions are missing:

1. Check if they're in an unusual format
2. Try `--no-validate` to see what was extracted
3. Check for page layout changes in newer statements

### Unicode/Encoding Issues

The CSV output uses UTF-8 encoding. If you see garbled characters in Excel:

1. Open Excel
2. Data → From Text/CSV
3. Select UTF-8 encoding

## Tested With

- Barclaycard Platinum Visa statements (2023-2026)
- 4-5 page statements
- Statements with foreign currency transactions
- Statements with credit balances
- Statements with refunds

## License

MIT License — feel free to modify and distribute.

## Contributing

If you encounter a statement that doesn't parse correctly:

1. Check if validation passes
2. Note which check failed and by how much
3. The difference often points to the specific missing transaction

Pull requests welcome for new edge cases!
