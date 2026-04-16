# Geant2001 Analysis Scripts — v3

## Thay đổi so với v2

| Hàm | File | Fix |
|-----|------|-----|
| `parse_demands()` | analysis + deep | Outer try/except cho file open; per-line số dòng; `errors='replace'` |
| `get_sp()` | deep | Check cả 2 chiều key; validate path là list ≥2; 3 exception types riêng biệt |
| `load_timesteps()` | plots | JSONDecodeError + OSError riêng; validate struct từng entry; cast guard |

## Chạy theo thứ tự

```bash
python3 geant2001_analysis_v3.py
python3 geant2001_plots_v3.py
python3 geant2001_deep_analysis_v3.py
```
