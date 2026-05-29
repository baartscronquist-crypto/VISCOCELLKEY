import numpy as np
import traceback
import sys
import os
import contextlib
import builtins
import signal

_TRUSTED_STDOUT = sys.stdout
_TRUSTED_STDERR = sys.stderr
_OS_DUP = os.dup
_OS_DUP2 = os.dup2
_OS_CLOSE = os.close
_OS_WRITE = os.write
_OPEN = builtins.open
_DEVNULL = os.devnull
_RESULT_PREFIX = "EVALUATOR_FINAL_SCORE"
_TRUSTED_STDOUT_FD = _OS_DUP(1)
_IMPORT_TIMEOUT_SECONDS = float(os.environ.get("EVALUATOR_IMPORT_TIMEOUT_SECONDS", "10"))
_CALL_TIMEOUT_SECONDS = float(os.environ.get("EVALUATOR_CALL_TIMEOUT_SECONDS", "15"))


class _EvaluatorTimeout(TimeoutError):
    pass


def _emit(message=""):
    """Write only evaluator-owned output to the original stdout fd."""
    _OS_WRITE(_TRUSTED_STDOUT_FD, (str(message) + "\n").encode("utf-8", errors="replace"))


def _emit_final_score(score):
    _emit(f"\n========================================\n测评结束！Agent 最终真实得分: {score}/100\n========================================")
    _emit(f"{_RESULT_PREFIX}={score}/100")


def _flush_streams():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


@contextlib.contextmanager
def _time_limit(seconds, label):
    if seconds <= 0:
        yield
        return

    def _handle_timeout(signum, frame):
        raise _EvaluatorTimeout(f"{label} timed out after {seconds:g}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])


@contextlib.contextmanager
def _suppress_untrusted_output():
    """Hide stdout/stderr produced while importing or calling agent code."""
    _flush_streams()
    saved_stdout_fd = _OS_DUP(1)
    saved_stderr_fd = _OS_DUP(2)
    try:
        with _OPEN(_DEVNULL, "wb") as devnull:
            _OS_DUP2(devnull.fileno(), 1)
            _OS_DUP2(devnull.fileno(), 2)
            yield
    finally:
        _flush_streams()
        _OS_DUP2(saved_stdout_fd, 1)
        _OS_DUP2(saved_stderr_fd, 2)
        _OS_CLOSE(saved_stdout_fd)
        _OS_CLOSE(saved_stderr_fd)
        sys.stdout = _TRUSTED_STDOUT
        sys.stderr = _TRUSTED_STDERR


# 导入 Agent 的代码进行测试
try:
    with _time_limit(_IMPORT_TIMEOUT_SECONDS, "scaffold import"):
        with _suppress_untrusted_output():
            from scaffold import vis_MC_Gillespie_Elastic_SLS as _agent_vis_MC_Gillespie_Elastic_SLS
except BaseException:
    _emit("错误: 找不到 scaffold.py、函数接口被破坏或导入时崩溃。")
    if os.environ.get("EVALUATOR_DEBUG") == "1":
        _OS_WRITE(2, traceback.format_exc().encode("utf-8", errors="replace"))
    _emit_final_score(0)
    sys.exit(0)

# 导入真值基准
try:
    from ground_truth import vis_MC_Gillespie_Elastic_SLS as vis_GT
except ModuleNotFoundError as exc:
    if exc.name != "ground_truth":
        _emit("错误: ground_truth.py 的依赖导入失败，无法执行真实 RMSE 评测。")
        if os.environ.get("EVALUATOR_DEBUG") == "1":
            _OS_WRITE(2, traceback.format_exc().encode("utf-8", errors="replace"))
        _emit_final_score(0)
        sys.exit(0)
    vis_GT = None
except BaseException:
    _emit("错误: ground_truth.py 导入时崩溃，无法执行真实 RMSE 评测。")
    if os.environ.get("EVALUATOR_DEBUG") == "1":
        _OS_WRITE(2, traceback.format_exc().encode("utf-8", errors="replace"))
    _emit_final_score(0)
    sys.exit(0)


def _call_agent(**kwargs):
    with _time_limit(_CALL_TIMEOUT_SECONDS, f"agent call {kwargs}"):
        with _suppress_untrusted_output():
            return _agent_vis_MC_Gillespie_Elastic_SLS(**kwargs)


def _autocorr_peak(values, max_points=4000):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size <= 20:
        return 0.0
    if arr.size > max_points:
        idx = np.linspace(0, arr.size - 1, max_points).astype(int)
        arr = arr[idx]
    centered = arr - np.mean(arr)
    zero_lag = float(np.dot(centered, centered))
    if zero_lag <= 0.0 or not np.isfinite(zero_lag):
        return 0.0
    autocorr = np.correlate(centered, centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:] / zero_lag
    if autocorr.size <= 10:
        return 0.0
    return float(np.max(autocorr[10:]))


def _is_dynamic_timeseries(T, vf, Fsub):
    try:
        T_arr = np.asarray(T, dtype=float).reshape(-1)
        vf_arr = np.asarray(vf, dtype=float).reshape(-1)
        f_arr = np.asarray(Fsub, dtype=float).reshape(-1)
    except Exception:
        return False

    if not (T_arr.size == vf_arr.size == f_arr.size):
        return False
    if T_arr.size < 20:
        return False
    if not (np.all(np.isfinite(T_arr)) and np.all(np.isfinite(vf_arr)) and np.all(np.isfinite(f_arr))):
        return False
    if T_arr[-1] <= T_arr[0] or np.any(np.diff(T_arr) < 0):
        return False
    if np.std(vf_arr) <= 1e-8 or np.std(f_arr) <= 1e-8:
        return False
    return True


def _finite_scalar(value):
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 1 or not np.isfinite(arr[0]):
        raise ValueError("expected a finite scalar")
    return float(arr[0])


def _safe_corrcoef(a, b):
    a_arr = np.asarray(a, dtype=float).reshape(-1)
    b_arr = np.asarray(b, dtype=float).reshape(-1)
    if a_arr.size != b_arr.size or a_arr.size < 3:
        return 0.0
    if not (np.all(np.isfinite(a_arr)) and np.all(np.isfinite(b_arr))):
        return 0.0
    if np.std(a_arr) <= 1e-12 or np.std(b_arr) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a_arr, b_arr)[0, 1])


def _estimate_sawtooth_period(T, vf, grid_dt=0.5):
    """Estimate the mean load-fail wavelength from repeated high-vf bursts."""
    try:
        t_arr = np.asarray(T, dtype=float).reshape(-1)
        vf_arr = np.asarray(vf, dtype=float).reshape(-1)
    except Exception:
        return {
            "period": np.nan,
            "cv": np.inf,
            "count": 0,
            "amplitude": 0.0,
        }

    valid = np.isfinite(t_arr) & np.isfinite(vf_arr)
    t_arr = t_arr[valid]
    vf_arr = vf_arr[valid]
    if t_arr.size < 20 or t_arr.size != vf_arr.size:
        return {
            "period": np.nan,
            "cv": np.inf,
            "count": 0,
            "amplitude": 0.0,
        }

    order = np.argsort(t_arr)
    t_arr = t_arr[order]
    vf_arr = vf_arr[order]
    unique = np.r_[True, np.diff(t_arr) > 1e-9]
    t_arr = t_arr[unique]
    vf_arr = vf_arr[unique]
    if t_arr.size < 20 or t_arr[-1] - t_arr[0] < 100.0:
        return {
            "period": np.nan,
            "cv": np.inf,
            "count": 0,
            "amplitude": 0.0,
        }

    grid = np.arange(t_arr[0], t_arr[-1] + 1e-9, grid_dt)
    if grid.size < 20:
        return {
            "period": np.nan,
            "cv": np.inf,
            "count": 0,
            "amplitude": 0.0,
        }

    values = np.interp(grid, t_arr, vf_arr)
    window = 5
    smoothed = np.convolve(values, np.ones(window) / window, mode="same")
    amplitude = float(np.percentile(smoothed, 90) - np.percentile(smoothed, 10))
    threshold = float(np.percentile(smoothed, 80))
    high = smoothed >= threshold
    starts = np.flatnonzero(high & np.r_[True, ~high[:-1]])

    times = []
    for t_val in grid[starts]:
        if t_val < grid[0] + 5.0:
            continue
        if not times or t_val - times[-1] >= 3.0:
            times.append(float(t_val))

    if len(times) < 3:
        return {
            "period": np.nan,
            "cv": np.inf,
            "count": len(times),
            "amplitude": amplitude,
        }

    intervals = np.diff(times)
    mean_period = float(np.mean(intervals))
    cv = float(np.std(intervals) / mean_period) if mean_period > 0 else np.inf
    return {
        "period": mean_period,
        "cv": cv,
        "count": len(times),
        "amplitude": amplitude,
    }


def _scan_vfmean(ka, neta_scan):
    values = []
    np.random.seed(2026)
    for neta in neta_scan:
        values.append(_finite_scalar(_call_agent(neta=neta, ka=ka)[3]))
    return np.asarray(values, dtype=float)


def evaluate_agent():
    score = 0
    _emit("开始执行 10 项高阶动力学指标评测 (基于 vfmean 物理形态, 满分 100)...\n")

    neta_scan = np.logspace(np.log10(0.01), np.log10(100), 10)

    try:
        # ====================================================================
        # [阶段一] 基础数值稳定性与边界逻辑 (25分，每项5分)
        # ====================================================================
        _emit("--- [阶段一] 基础数值稳定性与基本特征 ---")

        # [指标 i] 数值正常与防 NaN 爆炸
        T_lf, vf_lf, Fsub_lf, vfm_lf, Fadh_lf = _call_agent(neta=1.0, ka=0.1)
        if _is_dynamic_timeseries(T_lf, vf_lf, Fsub_lf):
            _emit(">> [i] 通过: 时序长度、有限性与动态变化均正常 (+5分)")
            score += 5
        else:
            _emit(">> [i] 失败: 输出不是有效的动态时序。")

        # [指标 ii] 宏观受力状态区分
        _, _, _, vfm_sp, Fadh_sp = _call_agent(neta=1.0, ka=10.0)
        Fadh_lf = _finite_scalar(Fadh_lf)
        Fadh_sp = _finite_scalar(Fadh_sp)
        if Fadh_lf > Fadh_sp + 20.0:
            _emit(
                f">> [ii] 通过: 高低刚度黏附力状态区分明确 "
                f"(ka=0.1: {Fadh_lf:.1f}, ka=10: {Fadh_sp:.1f}) (+5分)"
            )
            score += 5
        else:
            _emit(
                f">> [ii] 失败: 高低刚度黏附力差异不足 "
                f"(ka=0.1: {Fadh_lf:.1f}, ka=10: {Fadh_sp:.1f})"
            )

        # 预先跑出扫描数据，固定种子。这里直接考 vfmean，不再把它转成 vu-vf。
        vfm_scan = {
            0.1: _scan_vfmean(0.1, neta_scan),
            0.5: _scan_vfmean(0.5, neta_scan),
            1.0: _scan_vfmean(1.0, neta_scan),
            10.0: _scan_vfmean(10.0, neta_scan),
        }
        vfm_01 = vfm_scan[0.1]
        vfm_10 = vfm_scan[10.0]

        # [指标 iii] 低刚度 ka=0.1 的 vfmean 必须先下降再恢复，极小值在低 neta 段。
        min_idx_01 = int(np.argmin(vfm_01))
        weak_depth_left = float(vfm_01[0] - vfm_01[min_idx_01])
        weak_recovery = float(np.mean(vfm_01[-5:]) - vfm_01[min_idx_01])
        weak_min_neta = float(neta_scan[min_idx_01])
        if 1 <= min_idx_01 <= 6 and weak_depth_left > 3.0 and weak_recovery > 5.0:
            _emit(
                f">> [iii] 通过: ka=0.1 的 vfmean 出现低 neta 谷值 "
                f"(neta={weak_min_neta:.3g}, depth={weak_depth_left:.1f}, recovery={weak_recovery:.1f}) (+5分)"
            )
            score += 5
        else:
            _emit(
                f">> [iii] 失败: ka=0.1 的 vfmean 谷值位置/深度不符 "
                f"(neta={weak_min_neta:.3g}, depth={weak_depth_left:.1f}, recovery={weak_recovery:.1f})"
            )

        # [指标 iv] 强基底 ka=10 的 vfmean 应随 neta 明显上升，而不是高位平线或随机抖动。
        strong_head = float(np.mean(vfm_10[:5]))
        strong_tail = float(np.mean(vfm_10[-5:]))
        strong_rise = strong_tail - strong_head
        strong_corr = _safe_corrcoef(np.log10(neta_scan), vfm_10)
        if strong_head < 65.0 and strong_tail > 90.0 and strong_rise > 35.0 and strong_corr > 0.70:
            _emit(
                f">> [iv] 通过: ka=10 的 vfmean 随 neta 显著上升 "
                f"(head={strong_head:.1f}, tail={strong_tail:.1f}, corr={strong_corr:.2f}) (+5分)"
            )
            score += 5
        else:
            _emit(
                f">> [iv] 失败: ka=10 的 vfmean 不是真值中的上升趋势 "
                f"(head={strong_head:.1f}, tail={strong_tail:.1f}, rise={strong_rise:.1f}, corr={strong_corr:.2f})"
            )

        # [指标 v] 单点绝对精度误差 (直接针对 vfmean)
        test_vfm = float(vfm_01[0])
        if vis_GT is not None:
            np.random.seed(2026)
            gt_point = _finite_scalar(vis_GT(neta=0.01, ka=0.1)[3])
        else:
            gt_point = 38.0
        rel_err = abs(test_vfm - gt_point) / max(abs(gt_point), 1e-9)
        if rel_err <= 0.10:
            _emit(
                f">> [v] 通过: vfmean 单点绝对精度误差 <= 10% "
                f"(目标值 {gt_point:.1f}, 当前值 {test_vfm:.1f}, rel={rel_err:.2%}) (+5分)"
            )
            score += 5
        else:
            _emit(
                f">> [v] 失败: vfmean 单点偏差过大 "
                f"(目标值 {gt_point:.1f}, 当前值 {test_vfm:.1f}, rel={rel_err:.2%})"
            )

        _emit(f"\n目前阶段得分: {score}/25")

        # ====================================================================
        # [阶段二] 高阶形态学拓扑与全维拟合考核 (75分，每项15分)
        # ====================================================================
        _emit("\n--- [阶段二] 高阶形态学拓扑与全维关联 ---")

        # [指标 vi] 低刚度周期锯齿波的平均波长/周期检验
        period_info = _estimate_sawtooth_period(T_lf, vf_lf)
        period = period_info["period"]
        cycle_count = period_info["count"]
        cycle_cv = period_info["cv"]
        amplitude = period_info["amplitude"]
        if (
            np.isfinite(period)
            and 12.0 <= period <= 24.0
            and cycle_count >= 35
            and cycle_cv <= 0.45
            and amplitude >= 45.0
        ):
            _emit(
                f">> [vi] 通过: ka=0.1 周期锯齿波平均波长正确 "
                f"(period={period:.1f}s, cycles={cycle_count}, cv={cycle_cv:.2f}, amp={amplitude:.1f}) (+15分)"
            )
            score += 15
        else:
            _emit(
                f">> [vi] 失败: ka=0.1 周期锯齿波平均波长不符 "
                f"(period={period:.1f}s, cycles={cycle_count}, cv={cycle_cv:.2f}, amp={amplitude:.1f})"
            )

        # [指标 vii] 力-速度交叉相关性检验
        # 无论怎么相减，力与速度的制约关系是不变的
        correlation = _safe_corrcoef(vf_lf, Fsub_lf)
        if correlation < -0.8:
            _emit(f">> [vii] 通过: 速度与离合器牵引力呈现强物理负耦合 (Pearson r={correlation:.2f}) (+15分)")
            score += 15
        else:
            _emit(f">> [vii] 失败: 速度与牵引力失去镜像制约 (Pearson r={correlation:.2f})。")

        # [指标 viii] ka 增大时，低刚度 vfmean 极小值点必须向右移动。
        _emit("\n[指标 viii] 正在分析 ka=0.1/0.5/1 的 vfmean 极小值右移...")
        low_ka_values = [0.1, 0.5, 1.0]
        min_indices = [int(np.argmin(vfm_scan[ka])) for ka in low_ka_values]
        min_netas = [float(neta_scan[idx]) for idx in min_indices]
        valley_depths = [
            float(np.mean(vfm_scan[ka][:3]) - np.min(vfm_scan[ka]))
            for ka in low_ka_values
        ]
        right_shift_ok = min_netas[0] < min_netas[1] < min_netas[2]
        expected_bands_ok = (
            min_netas[0] < 0.20
            and 0.10 <= min_netas[1] <= 1.50
            and 0.50 <= min_netas[2] <= 5.00
        )
        depth_ok = valley_depths[0] > 2.0 and valley_depths[1] > 8.0 and valley_depths[2] > 8.0
        _emit(
            f"极小值 neta: ka=0.1->{min_netas[0]:.3g}, "
            f"ka=0.5->{min_netas[1]:.3g}, ka=1->{min_netas[2]:.3g}"
        )
        _emit(
            f"谷值深度: ka=0.1->{valley_depths[0]:.1f}, "
            f"ka=0.5->{valley_depths[1]:.1f}, ka=1->{valley_depths[2]:.1f}"
        )
        if right_shift_ok and expected_bands_ok and depth_ok:
            _emit(">> [viii] 通过: ka 增大导致 vfmean 极小值点按真值向右移动 (+15分)")
            score += 15
        else:
            _emit(">> [viii] 失败: vfmean 极小值没有随 ka 增大按真值右移。")

        # [指标 ix] 强基底 ka=10 的全局上升趋势和高 neta 平台。
        _emit("\n[指标 ix] 正在分析 ka=10 的 vfmean 全局上升趋势...")
        strong_tail_std = float(np.std(vfm_10[-4:]))
        if (
            strong_head < 65.0
            and strong_tail > 95.0
            and strong_rise > 45.0
            and strong_corr > 0.80
            and strong_tail_std < 2.0
        ):
            _emit(
                f">> [ix] 通过: ka=10 从低速区上升到高 neta 平台 "
                f"(head={strong_head:.1f}, tail={strong_tail:.1f}, rise={strong_rise:.1f}, tail_std={strong_tail_std:.2f}) (+15分)"
            )
            score += 15
        else:
            _emit(
                f">> [ix] 失败: ka=10 全局曲线不像真值的上升-平台形态 "
                f"(head={strong_head:.1f}, tail={strong_tail:.1f}, rise={strong_rise:.1f}, corr={strong_corr:.2f}, tail_std={strong_tail_std:.2f})"
            )

        # [指标 x] 强基底 ka=10 在 neta=1 附近应有大幅阶跃突变。
        _emit("\n[指标 x] 正在分析 ka=10 在 neta=1 附近的 vfmean 阶跃...")
        idx_neta1 = int(np.argmin(np.abs(np.log10(neta_scan))))
        left_slice = vfm_10[max(0, idx_neta1 - 4):idx_neta1]
        right_slice = vfm_10[idx_neta1:min(vfm_10.size, idx_neta1 + 4)]
        plateau_low = float(np.mean(left_slice))
        plateau_high = float(np.mean(right_slice))
        step = plateau_high - plateau_low
        local_jump = float(vfm_10[min(idx_neta1 + 1, vfm_10.size - 1)] - vfm_10[max(idx_neta1 - 1, 0)])
        _emit(
            f"精准定位 neta≈1 位置: 数组索引 {idx_neta1}, neta={neta_scan[idx_neta1]:.3g}"
        )
        _emit(
            f"左平台={plateau_low:.1f} | 右平台={plateau_high:.1f} | "
            f"阶跃幅度={step:.1f} | 局部跳变={local_jump:.1f}"
        )
        if plateau_low < 75.0 and plateau_high > 80.0 and step > 25.0 and local_jump > 20.0:
            _emit(">> [x] 通过: ka=10 在 neta≈1 附近复现大幅阶跃上升 (+15分)")
            score += 15
        else:
            _emit(">> [x] 失败: 未检测到 ka=10 在 neta≈1 附近的有效阶跃。")

    except Exception:
        _emit("\n>> 最终判定: 运行期间发生未知程序崩溃，评分置 0。")
        if os.environ.get("EVALUATOR_DEBUG") == "1":
            _OS_WRITE(2, traceback.format_exc().encode("utf-8", errors="replace"))
        score = 0

    _emit_final_score(score)
    return score

if __name__ == "__main__":
    evaluate_agent()
