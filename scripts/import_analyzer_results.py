"""Strict join of analyzer results to synthesis variables; preserve QC and units."""
import argparse
import pandas as pd


def join_results(design_path, results_path, output_path):
    design = pd.read_csv(design_path, dtype={'sample_id': str})
    results = pd.read_csv(results_path, dtype={'sample_id': str})
    for frame in [design, results]:
        if 'sample_id' not in frame or frame.sample_id.isna().any() or frame.sample_id.str.strip().eq('').any() or frame.sample_id.duplicated().any():
            raise ValueError('Both files require unique nonempty sample_id values')
    if set(design.sample_id) != set(results.sample_id):
        raise ValueError('Design and results must contain identical sample-ID sets')
    overlap = (set(design) & set(results)) - {'sample_id'}
    if overlap:
        raise ValueError('Ambiguous shared columns: ' + ', '.join(sorted(overlap)))
    if 'qc_status' not in results:
        raise ValueError('Analyzer results must include qc_status')
    merged = design.merge(results, on='sample_id', how='left', validate='one_to_one', sort=False)
    merged.to_csv(output_path, index=False)
    return merged


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    try:
        joined = join_results(args.design, args.results, args.out)
    except ValueError as error:
        parser.error(str(error))
    print(f'Joined {len(joined)} samples; retained measurement units and QC')
