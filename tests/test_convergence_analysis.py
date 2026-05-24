import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'analyze_convergence.py'
    spec = importlib.util.spec_from_file_location('analyze_convergence', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_settled_step_and_increase_steps():
    m = _load_module()
    energies = [0.0, -1.0, -2.0, -1.5, -2.1]
    assert m.settled_step(energies) == 4
    assert m.increase_steps(energies) == [4]


def test_parse_log_events(tmp_path):
    m = _load_module()
    log = tmp_path / 'water.log'
    log.write_text(
        '1 * Trust radius: 0.30\n'
        '2 * Trust radius: 0.15\n'
        '2 Linear-bend topology changed; rebuilding internal coordinates\n'
        '3 * No fit succeeded, returning to best point\n'
        '4 Minimization on sphere was performed:\n'
    )
    parsed = m.parse_log(log)
    assert parsed['trust_reduction_steps'] == {2}
    assert parsed['rebuild_steps'] == {2}
    assert parsed['nofit_steps'] == {3}
    assert parsed['sphere_steps'] == {4}


def test_analyze_and_render_markdown(tmp_path):
    m = _load_module()
    results = tmp_path / 'results'
    results.mkdir()
    (results / 'mopac-baker-b0.json').write_text(
        json.dumps(
            {
                'rows': [
                    {'name': 'water', 'energies': [0.0, -0.5, -0.7, -0.71]},
                    {
                        'name': 'methylamine',
                        'energies': [0.0, -0.1, -0.2, -0.15, -0.25],
                    },
                ]
            }
        )
    )
    rows = m.load_rows(results, 'baker', 'mopac')
    reference = json.loads((m.BENCHMARKS['baker'] / 'reference.json').read_text())
    analyzed = m.analyze(
        rows=rows,
        benchmark='baker',
        reference=reference,
        data_dir=m.BENCHMARKS['baker'],
        logs_dir=None,
    )
    assert analyzed['slowest'][0]['name'] == 'methylamine'
    assert analyzed['slowest'][0]['paper_delta'] == 1
    md = m.render_markdown('baker', 'mopac', analyzed)
    assert 'Slowest systems' in md
    assert 'Comparison vs Standard Method' in md
