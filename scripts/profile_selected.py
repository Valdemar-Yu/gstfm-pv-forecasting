"""Count and optionally time the exact H=1 validation-selected architectures."""
import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--counts-only', action='store_true')
    parser.add_argument('--out', default='results/selected_profile.csv')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()
    import torch
    from train_baseline import build_model, load_tuned_arch
    from models.iTransformer_LSTM_agcf import iTransformer_LSTM
    from train_gstfm import TUNABLE_KEYS

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    rows = []
    for season in ('Spring', 'Summer', 'Autumn', 'Winter'):
        for name in ('persistence', 'dlinear', 'timesnet', 'fedformer',
                     'itransformer', 'lstm', 'itransformer_lstm', 'gstfm'):
            torch.manual_seed(42)
            if name == 'gstfm':
                path = ROOT / 'results' / f'best_params_7-First-Solar_{season}_h1.json'
                cfg = {k: v for k, v in json.loads(path.read_text()).items() if k in TUNABLE_KEYS}
                model = iTransformer_LSTM(input_size=5, length_input=24,
                                         length_pre=1, use_agcf=True, **cfg)
            else:
                cfg = {} if name == 'persistence' else load_tuned_arch(name, '7-First-Solar', season, 1)
                if cfg is None:
                    raise FileNotFoundError(f'missing selected configuration: {name}/{season}')
                model = build_model(name, 24, 1, arch=cfg)
            model = model.to(device).eval()
            row = dict(season=season, model=name, pred_len=1,
                       params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                       config=json.dumps(cfg, sort_keys=True), device=str(device),
                       hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.machine(),
                       python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda)
            with torch.no_grad():
                assert model(torch.randn(1, 24, 5, device=device)).shape == (1, 1)
                if not args.counts_only:
                    if device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(device)
                    for batch in (1, 32):
                        x = torch.randn(batch, 24, 5, device=device)
                        for _ in range(args.warmup):
                            model(x)
                        elapsed = []
                        for _ in range(args.iters):
                            if device.type == 'cuda':
                                torch.cuda.synchronize(device)
                            start = time.perf_counter()
                            model(x)
                            if device.type == 'cuda':
                                torch.cuda.synchronize(device)
                            elapsed.append((time.perf_counter() - start) * 1000)
                        row[f'latency_b{batch}_ms'] = statistics.mean(elapsed)
                        row[f'latency_b{batch}_std_ms'] = statistics.stdev(elapsed)
                    row['peak_memory_mb'] = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else ''
            rows.append(row)
            print(f'{season}/{name}: {row["params"]:,} parameters', flush=True)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == '__main__':
    main()
