#!/usr/bin/env python3
"""Find connected passing regions in a completed fxopt Cartesian grid."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from fxopt.results import read_result_columns


def analyze(run: Path, metric: str, minimum: float, max_pdif: float | None,
            selected_axes: list[str], top: int) -> dict:
    metric = {'yb_gm': 'yb_apy_gm'}.get(metric, metric)
    metrics = list(dict.fromkeys([metric] + (['max_7d_rel_price_diff'] if max_pdif is not None else [])))
    columns = read_result_columns(run, metrics=metrics)
    axes = columns.metadata['axes']
    names = sorted(axes)
    shape = tuple(columns.metadata['shape'])
    if not names or shape != tuple(len(axes[name]) for name in names):
        raise ValueError('run has no valid Cartesian axis metadata')
    parameter_names = columns.metadata.get('compiled_policy', {}).get('parameter_names', [])
    aliases = {f'policy_params.{i}': name for i, name in enumerate(parameter_names)}
    selected = []
    for requested in selected_axes:
        matches = [name for name in names if requested in (
            name, aliases.get(name), aliases.get(name, '').split(' (')[0])]
        if len(matches) != 1:
            raise ValueError(f'axis {requested!r} is unknown or ambiguous; available: {names}')
        selected.append(matches[0])
    if len(set(selected)) != len(selected):
        raise ValueError('considered axes must be distinct')
    dimensions = [names.index(name) for name in selected]
    order = np.argsort(columns.ordinals)
    if not np.array_equal(columns.ordinals[order], np.arange(math.prod(shape))):
        raise ValueError('run must contain every Cartesian ordinal exactly once')
    values = columns.metrics[metric][order].reshape(shape)
    passing = columns.ok_mask[order].reshape(shape) & np.isfinite(values) & (values > minimum)
    pdif = None
    if max_pdif is not None:
        pdif = columns.metrics['max_7d_rel_price_diff'][order].reshape(shape)
        passing &= np.isfinite(pdif) & (pdif < max_pdif)

    # Only face neighbors on selected dimensions connect; other axes remain fixed.
    structure = np.zeros((3,) * len(shape), dtype=bool)
    center = [1] * len(shape)
    structure[tuple(center)] = True
    degree = np.zeros(shape, dtype=np.int32)
    for dimension in dimensions:
        for direction in (-1, 1):
            neighbor = center.copy()
            neighbor[dimension] += direction
            structure[tuple(neighbor)] = True
            adjacent = np.roll(passing, direction, axis=dimension)
            boundary = [slice(None)] * len(shape)
            boundary[dimension] = 0 if direction == 1 else -1
            adjacent[tuple(boundary)] = False
            degree += adjacent
    labels, component_count = label(passing, structure)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    ranked = sorted(range(1, component_count + 1), key=lambda i: (-int(counts[i]), i))[:top]
    blobs = []
    for component in ranked:
        ordinals = np.flatnonzero(labels.ravel() == component)
        coordinates = np.array(np.unravel_index(ordinals, shape)).T
        low, high = coordinates.min(axis=0), coordinates.max(axis=0)
        samples = values.ravel()[ordinals]
        representative = int(ordinals[np.argmax(degree.ravel()[ordinals])])
        blob = {
            'count': len(ordinals),
            'fixed_axes': {name: axes[name][int(low[i])] for i, name in enumerate(names)
                           if i not in dimensions},
            'axis_extents': {name: {'first': axes[name][int(low[i])],
                                   'last': axes[name][int(high[i])],
                                   'distinct_values': len(np.unique(coordinates[:, i]))}
                             for i, name in enumerate(names) if i in dimensions},
            'bounding_box_fill': len(ordinals) / int(np.prod(high - low + 1)),
            'metric_min_median_max': [float(samples.min()), float(np.median(samples)), float(samples.max())],
            'full_neighbor_centers': int(np.count_nonzero(degree.ravel()[ordinals] == 2 * len(dimensions))),
            'representative': {
                'ordinal': representative,
                'passing_neighbors': int(degree.ravel()[representative]),
                'metric': float(values.ravel()[representative]),
                'candidate': columns.candidate_at(representative).to_dict(ordinal=representative),
            },
            'ordinals': ordinals.tolist(),
        }
        if pdif is not None:
            blob['max_pdif'] = float(pdif.ravel()[ordinals].max())
        blobs.append(blob)
    return {
        'run': str(run.resolve()), 'metric': metric, 'min_value_exclusive': minimum,
        'max_7d_rel_price_diff_exclusive': max_pdif,
        'considered_axes': selected, 'axis_labels': {name: aliases.get(name, name) for name in selected},
        'adjacency': 'one adjacent sampled index in one considered axis; no diagonals or wraparound',
        'passing_points': int(passing.sum()), 'component_count': component_count,
        'non_ok_points': int((~columns.ok_mask).sum()), 'blobs': blobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='directory containing run.json and results.npz')
    parser.add_argument('--metric', default='yb_apy_gm', help='stored metric; yb_gm aliases yb_apy_gm')
    parser.add_argument('--min-value', required=True, type=float, help='exclusive minimum in raw units: 0.001 = 0.1%% APY')
    parser.add_argument('--max-pdif', type=float, help='exclusive max_7d_rel_price_diff: 0.15 = 15%%; omitted means no cap')
    parser.add_argument('--axes', nargs='+', required=True, help='canonical axis keys or stored policy parameter names')
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--output', type=Path, help='optional JSON report including exact member ordinals')
    args = parser.parse_args()
    if not math.isfinite(args.min_value) or args.top < 1:
        parser.error('--min-value must be finite and --top positive')
    if args.max_pdif is not None and (not math.isfinite(args.max_pdif) or args.max_pdif < 0):
        parser.error('--max-pdif must be finite and nonnegative')
    try:
        result = analyze(args.run, args.metric, args.min_value, args.max_pdif, args.axes, args.top)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    if args.output:
        if args.output.resolve() in {(args.run / 'run.json').resolve(), (args.run / 'results.npz').resolve()}:
            parser.error('--output must not overwrite the source artifacts')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f"{result['passing_points']} passing points; {result['component_count']} blobs; "
          f"{result['non_ok_points']} non-ok points excluded")
    print('Connectivity: adjacent sampled values, selected axes only. Values below use raw metric units.')
    for rank, blob in enumerate(result['blobs'], 1):
        print(f"\n#{rank}: {blob['count']} points; metric min/median/max={blob['metric_min_median_max']}; "
              f"box fill={blob['bounding_box_fill']:.1%}; full-neighbor centers={blob['full_neighbor_centers']}")
        print(f"  fixed: {blob['fixed_axes']}")
        for name, extent in blob['axis_extents'].items():
            print(f"  {result['axis_labels'][name]}: {extent}")
        print(f"  representative ordinal={blob['representative']['ordinal']}; "
              f"passing neighbors={blob['representative']['passing_neighbors']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
