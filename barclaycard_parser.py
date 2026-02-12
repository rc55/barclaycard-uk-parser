#!/usr/bin/env python3
"""
Barclaycard PDF Statement Parser

Handles the two-column transaction layout by splitting the page
and extracting transactions from each column separately.
"""

import pdfplumber
import pandas as pd
import re
import json
from datetime import datetime
from pathlib import Path


def extract_column_text(page, x_min, x_max, y_max=None):
    """Extract text from a specific horizontal region of the page.
    
    Args:
        page: pdfplumber page object
        x_min: Left boundary
        x_max: Right boundary  
        y_max: Optional bottom boundary (to exclude footer content like "Ways to pay")
    """
    # Filter characters within the x-bounds (and y-bounds if specified)
    if y_max:
        chars = [c for c in page.chars if x_min <= c['x0'] < x_max and c['top'] < y_max]
    else:
        chars = [c for c in page.chars if x_min <= c['x0'] < x_max]
    
    if not chars:
        return ""
    
    # Group characters by line (y-position with tolerance)
    lines = {}
    for c in chars:
        # Round y to group characters on the same line
        y_key = round(c['top'] / 3) * 3  # 3-point tolerance
        if y_key not in lines:
            lines[y_key] = []
        lines[y_key].append(c)
    
    # Sort each line by x position and build text
    result_lines = []
    for y_key in sorted(lines.keys()):
        line_chars = sorted(lines[y_key], key=lambda c: c['x0'])
        line_text = ''.join(c['text'] for c in line_chars)
        result_lines.append(line_text)
    
    return '\n'.join(result_lines)


def parse_transaction_line(line):
    """
    Parse a transaction line into components.
    Returns dict with date, description, amount, is_credit, or None if not a transaction.
    """
    # Transaction pattern: DD Mon[optional space]Description [-]£Amount[CR]
    # The PDF extraction often removes the space between date and description
    # Examples:
    #   04 NovSQ *Exeter Science Park L, Exeter£4.45
    #   06 NovFS *Illformed.com, Fsprg.UK£60.00CR
    #   05 Nov Payment, Thank You£34.99
    #   17 MayPayrev Payment Reversal-£1,900.68  (negative amount)
    
    # Pattern handles optional space after month, optional negative sign before £
    pattern = r'^(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\s*(.+?)(-)?£([\d,]+\.\d{2})(CR)?$'
    
    match = re.match(pattern, line.strip())
    if match:
        date_str = match.group(1)
        description = match.group(2).strip()
        is_negative = match.group(3) == '-'
        amount_str = match.group(4).replace(',', '')
        amount = float(amount_str)
        is_credit_suffix = match.group(5) == 'CR'
        
        # Either CR suffix or negative prefix indicates credit
        is_credit = is_credit_suffix or is_negative
        
        # Clean up description - remove trailing 'e' which appears for contactless
        # Also clean up truncated descriptions
        description = re.sub(r'\s*e\s*$', '', description)
        description = re.sub(r'\s+$', '', description)  # Trailing whitespace
        
        # Fix common truncation patterns
        if description.endswith(','):
            description = description[:-1]
        
        return {
            'date': date_str,
            'description': description,
            'amount': amount,
            'is_credit': is_credit
        }
    
    return None


def parse_statement(pdf_path, statement_year=None):
    """
    Parse a Barclaycard statement PDF and extract all transactions.
    
    Args:
        pdf_path: Path to the PDF file
        statement_year: Year for the statement (extracted from PDF if not provided)
    
    Returns:
        dict with metadata and list of transactions
    """
    transactions = []
    metadata = {}
    
    with pdfplumber.open(pdf_path) as pdf:
        # Page 1: Extract metadata
        page1_text = pdf.pages[0].extract_text()
        
        # Extract statement date
        date_match = re.search(r'(\d{2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})', page1_text)
        if date_match:
            metadata['statement_date'] = date_match.group(1)
            if not statement_year:
                year_match = re.search(r'(\d{4})$', date_match.group(1))
                if year_match:
                    statement_year = int(year_match.group(1))
        
        # Extract balances (handle CR suffix for credit balances)
        balance_match = re.search(r'Your new balance[:\s]+£([\d,]+\.\d{2})(CR)?', page1_text)
        if balance_match:
            amount = float(balance_match.group(1).replace(',', ''))
            if balance_match.group(2) == 'CR':
                amount = -amount
            metadata['new_balance'] = amount
        
        prev_balance_match = re.search(r'Your previous balance[:\s]+£([\d,]+\.\d{2})(CR)?', page1_text)
        if prev_balance_match:
            amount = float(prev_balance_match.group(1).replace(',', ''))
            # CR means credit balance (negative from customer perspective, they're owed money)
            if prev_balance_match.group(2) == 'CR':
                amount = -amount
            metadata['previous_balance'] = amount
        
        # Extract "Your activity" summary figures for validation
        payments_match = re.search(r'Payments towards your account[:\s]+£([\d,]+\.\d{2})', page1_text)
        if payments_match:
            metadata['stated_payments'] = float(payments_match.group(1).replace(',', ''))
        
        new_activity_match = re.search(r'Your new activity[:\s]+£([\d,]+\.\d{2})', page1_text)
        if new_activity_match:
            metadata['stated_new_activity'] = float(new_activity_match.group(1).replace(',', ''))
        
        interest_match = re.search(r'Interest charged[:\s]+£([\d,]+\.\d{2})', page1_text)
        if interest_match:
            metadata['stated_interest'] = float(interest_match.group(1).replace(',', ''))
        
        charges_match = re.search(r'Other charges[:\s]+£([\d,]+\.\d{2})', page1_text)
        if charges_match:
            metadata['stated_charges'] = float(charges_match.group(1).replace(',', ''))
        
        # Process transaction pages (typically pages 2-3)
        # The layout uses two columns, split roughly at x=300
        column_split = 295  # Adjust based on examination
        
        for page_num in range(1, len(pdf.pages)):  # Skip page 0 (summary)
            page = pdf.pages[page_num]
            
            # Check if this page has transactions
            page_text = page.extract_text() or ""
            if "Your transactions" not in page_text and not re.search(
                r'\d{2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', page_text
            ):
                continue
            
            # Detect if this page has the "Ways to pay" / payment instructions box
            # If so, find its y-position and cut off extraction there
            y_cutoff = None
            if "Ways to pay" in page_text or ("Direct Debit" in page_text and "Sort code" in page_text):
                # Find the y-position where "Ways to pay" starts
                for char in page.chars:
                    if char['text'] == 'W' and char['x0'] < 100:
                        nearby = [c for c in page.chars 
                                  if abs(c['top'] - char['top']) < 3 
                                  and c['x0'] >= char['x0'] 
                                  and c['x0'] < char['x0'] + 100]
                        text = ''.join(c['text'] for c in sorted(nearby, key=lambda c: c['x0']))
                        if 'Waystopay' in text.replace(' ', '') or 'Ways to pay' in text:
                            y_cutoff = char['top'] - 5  # Cut off slightly above
                            break
            
            # Extract left and right columns
            left_text = extract_column_text(page, 0, column_split, y_max=y_cutoff)
            right_text = extract_column_text(page, column_split, page.width, y_max=y_cutoff)
            
            # Process each column
            for column_text in [left_text, right_text]:
                lines = column_text.split('\n')
                
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Skip non-transaction lines
                    if not line or line.startswith('Page ') or line.startswith('Ways to pay'):
                        i += 1
                        continue
                    
                    # Try to parse as transaction
                    txn = parse_transaction_line(line)
                    
                    if txn:
                        # Payments are credits (money coming in to pay off balance)
                        # Handle "Payment, Thank You", "Payment By Direct Debit", and "Payrev Payment Reversal"
                        if 'Payment' in txn['description'] and ('Thank You' in txn['description'] or 'Direct Debit' in txn['description']):
                            txn['is_credit'] = True
                            txn['type'] = 'payment'
                        elif 'Payrev' in txn['description'] and 'Payment Reversal' in txn['description']:
                            # Payment reversal - this is a negative payment (reverses a previous payment)
                            # It should be treated as a debit (adds back to balance)
                            txn['type'] = 'payment_reversal'
                        else:
                            txn['type'] = 'refund' if txn['is_credit'] else 'purchase'
                        
                        # Check for continuation lines (multi-line descriptions, forex info)
                        extra_info = []
                        j = i + 1
                        while j < len(lines):
                            next_line = lines[j].strip()
                            # Stop if it's a new transaction or section header
                            if not next_line:
                                j += 1
                                continue
                            if re.match(r'^\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', next_line):
                                break
                            if next_line.startswith('£') or 'balance' in next_line.lower():
                                break
                            # Skip page footer content
                            if any(x in next_line.lower() for x in ['ways to pay', 'direct debit', 'debit card', 
                                   'please pay', 'barclaycard', 'promotional', 'interest and charges',
                                   'transactions, interest']):
                                break
                            # Forex info or continuation
                            if re.match(r'^[\d.]+\s+(?:U\.?S\.?\s*Dollar|Euro|Pound)', next_line) or \
                               'Exch Rate' in next_line or \
                               'Non Sterling' in next_line or \
                               'For £' in next_line:
                                extra_info.append(next_line)
                                j += 1
                            elif not re.match(r'^\d', next_line) and len(next_line) < 50:
                                # Likely a continuation of merchant name (but not footer text)
                                if not any(x in next_line.lower() for x in ['visit', 'call', 'sort code', 'account number']):
                                    extra_info.append(next_line)
                                j += 1
                            else:
                                break
                        
                        if extra_info:
                            # Clean up extra_info - remove stray contactless markers
                            cleaned_info = []
                            for info in extra_info:
                                info = info.strip()
                                if info and info != 'e' and info != 'e ':
                                    cleaned_info.append(info)
                            if cleaned_info:
                                txn['extra_info'] = ' | '.join(cleaned_info)
                        
                        # Add full date with year
                        if statement_year:
                            txn['full_date'] = f"{txn['date']} {statement_year}"
                        
                        transactions.append(txn)
                        i = j
                    else:
                        i += 1
    
    # Sort transactions by date
    month_order = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                   'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    
    def sort_key(t):
        parts = t['date'].split()
        day = int(parts[0])
        month = month_order.get(parts[1], 0)
        return (month, day)
    
    transactions.sort(key=sort_key)
    
    # Remove true duplicates only (exact same transaction appearing twice due to column overlap)
    # We use a counter to allow legitimate duplicate transactions (e.g., two Ringgo charges)
    from collections import Counter
    seen = Counter()
    unique_txns = []
    for t in transactions:
        key = (t['date'], t['description'], t['amount'], t['is_credit'])
        seen[key] += 1
        # Allow up to 2 of each (handles real duplicates like two Ringgo parking charges)
        # but remove triplicates+ which are likely parsing artifacts
        if seen[key] <= 2:
            unique_txns.append(t)
    
    return {
        'metadata': metadata,
        'transactions': unique_txns
    }


def to_dataframe(parsed_data):
    """Convert parsed data to a pandas DataFrame."""
    df = pd.DataFrame(parsed_data['transactions'])
    
    if not df.empty:
        # Calculate signed amount (negative for credits/payments)
        df['signed_amount'] = df.apply(
            lambda r: -r['amount'] if r['is_credit'] else r['amount'], 
            axis=1
        )
        
        # Reorder columns
        cols = ['date', 'description', 'amount', 'type', 'signed_amount']
        if 'extra_info' in df.columns:
            cols.append('extra_info')
        if 'full_date' in df.columns:
            cols.insert(1, 'full_date')
        
        df = df[[c for c in cols if c in df.columns]]
    
    return df


def validate_extraction(parsed_data, tolerance=0.01):
    """
    Validate extracted transactions against statement summary figures.
    
    Args:
        parsed_data: Dict with 'metadata' and 'transactions' from parse_statement()
        tolerance: Acceptable difference in pounds (default £0.01)
    
    Returns:
        dict with validation results including pass/fail status and details
    """
    metadata = parsed_data['metadata']
    df = to_dataframe(parsed_data)
    
    results = {
        'valid': True,
        'checks': [],
        'warnings': [],
        'summary': {}
    }
    
    if df.empty:
        results['valid'] = False
        results['checks'].append({
            'name': 'transactions_found',
            'passed': False,
            'message': 'No transactions were extracted'
        })
        return results
    
    # Calculate totals from extracted transactions
    total_payments = df[df['type'] == 'payment']['amount'].sum()
    # Payment reversals subtract from the payments total
    total_payment_reversals = df[df['type'] == 'payment_reversal']['amount'].sum()
    net_payments = total_payments - total_payment_reversals
    
    total_purchases = df[df['type'] == 'purchase']['amount'].sum()
    total_refunds = df[df['type'] == 'refund']['amount'].sum()
    
    # "New activity" = purchases + refunds (as shown) - but refunds are shown as credits
    # The statement shows: Your new activity = gross transaction value (purchases - refunds + interest + charges)
    calculated_new_activity = total_purchases - total_refunds
    
    results['summary'] = {
        'extracted_payments': net_payments,
        'extracted_purchases': total_purchases,
        'extracted_refunds': total_refunds,
        'extracted_new_activity': calculated_new_activity,
        'transaction_count': len(df)
    }
    
    # Check 1: Payments match
    if 'stated_payments' in metadata:
        stated = metadata['stated_payments']
        diff = abs(net_payments - stated)
        passed = diff <= tolerance
        results['checks'].append({
            'name': 'payments_match',
            'passed': passed,
            'stated': stated,
            'extracted': net_payments,
            'difference': diff,
            'message': f"Payments: stated £{stated:.2f} vs extracted £{net_payments:.2f} (diff: £{diff:.2f})"
        })
        if not passed:
            results['valid'] = False
    
    # Check 2: New activity matches (purchases - refunds + interest + charges)
    if 'stated_new_activity' in metadata:
        stated = metadata['stated_new_activity']
        # Add interest and charges if present
        interest = metadata.get('stated_interest', 0)
        charges = metadata.get('stated_charges', 0)
        calculated = calculated_new_activity + interest + charges
        diff = abs(calculated - stated)
        passed = diff <= tolerance
        results['checks'].append({
            'name': 'new_activity_match',
            'passed': passed,
            'stated': stated,
            'extracted': calculated,
            'difference': diff,
            'message': f"New activity: stated £{stated:.2f} vs extracted £{calculated:.2f} (diff: £{diff:.2f})"
        })
        if not passed:
            results['valid'] = False
    
    # Check 3: Balance equation
    # new_balance = previous_balance - payments + new_activity
    if 'previous_balance' in metadata and 'new_balance' in metadata:
        prev = metadata['previous_balance']
        stated_new = metadata['new_balance']
        interest = metadata.get('stated_interest', 0)
        charges = metadata.get('stated_charges', 0)
        
        # Calculate expected new balance (use net_payments which accounts for reversals)
        calculated_new = prev - net_payments + total_purchases - total_refunds + interest + charges
        diff = abs(calculated_new - stated_new)
        passed = diff <= tolerance
        
        results['checks'].append({
            'name': 'balance_equation',
            'passed': passed,
            'formula': f"£{prev:.2f} - £{net_payments:.2f} + £{total_purchases:.2f} - £{total_refunds:.2f} + £{interest:.2f} + £{charges:.2f}",
            'calculated_new_balance': calculated_new,
            'stated_new_balance': stated_new,
            'difference': diff,
            'message': f"Balance check: calculated £{calculated_new:.2f} vs stated £{stated_new:.2f} (diff: £{diff:.2f})"
        })
        if not passed:
            results['valid'] = False
    
    # Check 4: Transaction count sanity check
    txn_count = len(df)
    if txn_count < 1:
        results['warnings'].append("Very few transactions extracted - possible parsing issue")
    elif txn_count > 200:
        results['warnings'].append("Unusually high transaction count - possible duplicate extraction")
    
    results['checks'].append({
        'name': 'transaction_count',
        'passed': True,
        'count': txn_count,
        'message': f"Extracted {txn_count} transactions"
    })
    
    return results


def print_validation_report(validation_results):
    """Print a formatted validation report."""
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    
    status = "✓ PASSED" if validation_results['valid'] else "✗ FAILED"
    print(f"\nOverall Status: {status}")
    
    print("\n--- Checks ---")
    for check in validation_results['checks']:
        symbol = "✓" if check['passed'] else "✗"
        print(f"  {symbol} {check['message']}")
    
    if validation_results['warnings']:
        print("\n--- Warnings ---")
        for warning in validation_results['warnings']:
            print(f"  ⚠ {warning}")
    
    print("\n--- Extracted Summary ---")
    summary = validation_results['summary']
    print(f"  Payments: £{summary.get('extracted_payments', 0):.2f}")
    print(f"  Purchases: £{summary.get('extracted_purchases', 0):.2f}")
    print(f"  Refunds: £{summary.get('extracted_refunds', 0):.2f}")
    print(f"  Transaction count: {summary.get('transaction_count', 0)}")
    
    print("="*60)


def process_single_statement(pdf_path, output_path=None, skip_validation=False, quiet=False):
    """Process a single statement and optionally save output.
    
    Returns:
        tuple: (result_dict, validation_dict, dataframe) or None if failed
    """
    if not quiet:
        print(f"Parsing {pdf_path}...")
    
    try:
        result = parse_statement(pdf_path)
    except Exception as e:
        print(f"ERROR: Failed to parse {pdf_path}: {e}")
        return None
    
    if not quiet:
        print(f"  Statement Date: {result['metadata'].get('statement_date', 'Unknown')}")
        print(f"  Transactions: {len(result['transactions'])}")
    
    # Run validation
    validation = None
    if not skip_validation:
        validation = validate_extraction(result)
        if not quiet:
            status = "✓ PASSED" if validation['valid'] else "✗ FAILED"
            print(f"  Validation: {status}")
        
        if not validation['valid'] and not quiet:
            print(f"  ⚠ WARNING: Validation failed - data may be incomplete")
    
    df = to_dataframe(result)
    
    if output_path:
        if output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif output_path.endswith('.json'):
            output_data = result.copy()
            if validation:
                output_data['validation'] = validation
            
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    import numpy as np
                    if isinstance(obj, (np.bool_, bool)):
                        return bool(obj)
                    if isinstance(obj, (np.integer, int)):
                        return int(obj)
                    if isinstance(obj, (np.floating, float)):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super().default(obj)
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, cls=NumpyEncoder)
        
        if not quiet:
            print(f"  Saved to {output_path}")
    
    return result, validation, df


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Parse Barclaycard PDF statements and extract transactions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s statement.pdf                    Parse and validate, print summary
  %(prog)s statement.pdf -o output.csv      Save transactions to CSV
  %(prog)s statement.pdf -o output.json     Save full data with metadata to JSON
  %(prog)s *.pdf -o combined.csv            Batch process multiple statements
  %(prog)s *.pdf --output-dir ./exports     Save each statement separately
  %(prog)s statement.pdf --no-validate      Skip validation checks
  %(prog)s statement.pdf -v                 Verbose output with full validation report
        """
    )
    
    parser.add_argument('files', nargs='+', help='PDF statement file(s) to parse')
    parser.add_argument('-o', '--output', help='Output file (.csv or .json). For batch processing, combines all statements.')
    parser.add_argument('--output-dir', help='Output directory for batch processing (saves each file separately)')
    parser.add_argument('--no-validate', action='store_true', help='Skip validation checks')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output with full validation report')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet mode - only show errors')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    
    args = parser.parse_args()
    
    # Collect all results
    all_results = []
    all_dfs = []
    failed = []
    
    from pathlib import Path as PathLib  # Import here for use in loop
    
    for pdf_path in args.files:
        # Determine output path for this file
        if args.output_dir:
            PathLib(args.output_dir).mkdir(parents=True, exist_ok=True)
            base_name = PathLib(pdf_path).stem
            output_path = str(PathLib(args.output_dir) / f"{base_name}.csv")
        elif args.output and len(args.files) == 1:
            output_path = args.output
        else:
            output_path = None
        
        result = process_single_statement(
            pdf_path, 
            output_path=output_path,
            skip_validation=args.no_validate,
            quiet=args.quiet
        )
        
        if result:
            parsed, validation, df = result
            all_results.append((pdf_path, parsed, validation))
            
            # Add source file column for combined output
            df_with_source = df.copy()
            df_with_source.insert(0, 'source_file', PathLib(pdf_path).name)
            all_dfs.append(df_with_source)
            
            # Print full validation report if verbose
            if args.verbose and validation:
                print_validation_report(validation)
        else:
            failed.append(pdf_path)
    
    # Combined output for batch processing
    if args.output and len(args.files) > 1 and all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        if args.output.endswith('.csv'):
            combined_df.to_csv(args.output, index=False)
        elif args.output.endswith('.json'):
            combined_data = {
                'statements': [
                    {
                        'file': pdf_path,
                        'metadata': parsed['metadata'],
                        'validation': validation,
                        'transaction_count': len(parsed['transactions'])
                    }
                    for pdf_path, parsed, validation in all_results
                ],
                'transactions': combined_df.to_dict(orient='records')
            }
            with open(args.output, 'w') as f:
                json.dump(combined_data, f, indent=2, default=str)
        
        if not args.quiet:
            print(f"\nCombined output saved to {args.output}")
            print(f"  Total statements: {len(all_results)}")
            print(f"  Total transactions: {len(combined_df)}")
    
    # Summary
    if not args.quiet and len(args.files) > 1:
        print(f"\n{'='*50}")
        print(f"BATCH SUMMARY")
        print(f"{'='*50}")
        print(f"  Processed: {len(all_results)}/{len(args.files)} statements")
        if failed:
            print(f"  Failed: {len(failed)}")
            for f in failed:
                print(f"    - {f}")
        
        passed = sum(1 for _, _, v in all_results if v and v['valid'])
        print(f"  Validation: {passed}/{len(all_results)} passed")
    
    # Exit code
    if failed or any(v and not v['valid'] for _, _, v in all_results):
        exit(1)


if __name__ == '__main__':
    main()
