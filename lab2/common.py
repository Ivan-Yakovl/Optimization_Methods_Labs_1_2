"""Общий интерфейс HiGHS и файловая отчётность. Проверки находятся в main.py."""
from pathlib import Path
import csv, json, platform, time, os
import numpy as np
import highspy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOL = 1e-7
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

def require(condition, message):
    """Обязательная проверка; работает и при python -O."""
    if not bool(condition):
        raise ValueError(message)

def finite_array(label, values, shape=None, nonnegative=False):
    a=np.asarray(values,dtype=float)
    require(shape is None or a.shape==shape,f'{label}: неверная размерность {a.shape}, ожидается {shape}')
    require(np.all(np.isfinite(a)),f'{label}: NaN и бесконечные значения недопустимы')
    if nonnegative: require(np.all(a>=0),f'{label}: отрицательные значения недопустимы')
    return a

def solution_vector(x, expected):
    return finite_array('Вектор решения',x,(expected,))

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def csvwrite(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def solve(c, A, lo, up, upper, name, integer=None):
    """Все задачи минимизации. LP — dual simplex; MILP — branch-and-cut."""
    A = np.asarray(A, dtype=float)
    logpath=OUT / (name + '.log')
    if logpath.exists(): logpath.unlink()
    h = highspy.Highs()
    options = dict(presolve='on', threads=1, random_seed=10128,
                   primal_feasibility_tolerance=1e-8, dual_feasibility_tolerance=1e-8,
                   mip_feasibility_tolerance=1e-8, mip_rel_gap=0.0,
                   time_limit=120.0, log_to_console=False, output_flag=True,
                   log_file='results/' + name + '.log')
    if integer is None:
        options['solver'] = 'simplex'
    for k, v in options.items():
        if h.setOptionValue(k, v) != highspy.HighsStatus.kOk:
            raise RuntimeError(f'HiGHS не принял параметр {k}={v}')
    lp = highspy.HighsLp()
    lp.num_col_ = len(c); lp.num_row_ = len(lo)
    lp.col_cost_ = np.asarray(c, dtype=float)
    lp.col_lower_ = np.zeros(len(c)); lp.col_upper_ = np.asarray(upper, dtype=float)
    lp.row_lower_ = np.asarray(lo, dtype=float); lp.row_upper_ = np.asarray(up, dtype=float)
    # Разреженная матрица по строкам, без зависимости от scipy.
    ii, jj = np.nonzero(A)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = np.r_[0, np.cumsum(np.bincount(ii, minlength=len(lo)))].astype(np.int32)
    lp.a_matrix_.index_ = jj.astype(np.int32)
    lp.a_matrix_.value_ = A[ii, jj]
    if integer is not None:
        lp.integrality_ = [highspy.HighsVarType.kInteger if z else highspy.HighsVarType.kContinuous for z in integer]
    if h.passModel(lp) != highspy.HighsStatus.kOk:
        raise RuntimeError(f'{name}: HiGHS не принял модель')
    start = time.perf_counter(); run_status = h.run(); elapsed = time.perf_counter() - start
    info = h.getInfo(); sol = h.getSolution()
    status = h.modelStatusToString(h.getModelStatus())
    valid = bool(sol.value_valid)
    result = dict(status=status, seconds=elapsed, objective=float(info.objective_function_value) if valid else None,
                  x=list(sol.col_value) if valid else None, row_dual=list(sol.row_dual) if sol.dual_valid else None,
                  reduced_cost=list(sol.col_dual) if sol.dual_valid else None,
                  nodes=int(info.mip_node_count) if integer is not None else 0,
                  mip_gap=float(info.mip_gap) if integer is not None and np.isfinite(info.mip_gap) else None,
                  dual_bound=float(info.mip_dual_bound) if integer is not None and np.isfinite(info.mip_dual_bound) else None,
                  iterations=int(info.simplex_iteration_count), columns=len(c), rows=len(lo))
    dump(OUT / (name + '.json'), result)
    if h.writeModel('results/' + name + '.lp') == highspy.HighsStatus.kError:
        raise RuntimeError(f'{name}: не удалось сохранить LP-файл')
    if status != 'Optimal':
        raise RuntimeError(f'{name}: {status}; результаты нельзя считать оптимальными')
    if run_status != highspy.HighsStatus.kOk:
        raise RuntimeError(f'{name}: ошибка выполнения HiGHS {run_status}')
    solution_vector(result['x'],len(c))
    require(np.isfinite(result['objective']),f'{name}: неконечное значение цели')
    return result

def checks(name, rows, objective, solver_objective, extra=None):
    require(bool(rows),'Список проверяемых ограничений пуст')
    finite_array('Значения ограничений',[r['value'] for r in rows])
    finite_array('Нарушения ограничений',[r['violation'] for r in rows],nonnegative=True)
    finite_array('Значения цели',[objective,solver_objective],(2,))
    violation = max([float(r['violation']) for r in rows] + [0.0])
    err = abs(float(objective) - solver_objective)
    report = dict(tolerance=TOL, feasible=violation <= TOL, max_violation=violation,
                  recomputed_objective=float(objective), solver_objective=solver_objective,
                  objective_error=err, objective_ok=err <= TOL)
    if extra:
        report.update(extra)
    csvwrite(OUT / (name + '_constraints.csv'), rows)
    dump(OUT / (name + '_verification.json'), report)
    require(report['feasible'] and report['objective_ok'],f'{name}: независимая проверка не пройдена: {report}')
    return report

def boundrow(label, value, lo=0.0, up=float('inf')):
    require(np.isfinite(value),f'{label}: значение ограничения должно быть конечным')
    require(not np.isnan(lo) and not np.isnan(up) and lo<=up,f'{label}: некорректные границы')
    return dict(constraint=label, value=float(value), lower=lo if np.isfinite(lo) else '',
                upper=up if np.isfinite(up) else '', violation=float(max(0, lo-value, value-up)))

def environment():
    dump(OUT / 'environment.json', dict(python=platform.python_version(), numpy=np.__version__,
        highspy=highspy.Highs().version(), matplotlib=matplotlib.__version__, seed=10128,
        solver_options=dict(threads=1, presolve='on', random_seed=10128, time_limit=120,
        primal_feasibility_tolerance=1e-8, dual_feasibility_tolerance=1e-8,
        mip_feasibility_tolerance=1e-8, mip_rel_gap=0.0)))

def savefig(name):
    plt.tight_layout(); plt.savefig(OUT / name, dpi=180, bbox_inches='tight'); plt.close()
