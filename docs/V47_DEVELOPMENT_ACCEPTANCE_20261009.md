# V47 完整开发验收 — 2026-10-09

结论：54 项完成且均存活 2048 步，但 final40 未通过性能晋级条件；A 保持当前模型。存活是学习的前提，不等于控制性能达标。失败剩余时间惩罚保持 4000/s。holdout 12701/12702 未使用。

## 统计定义与单位

所有角速度误差表均为 deg/s（度/秒），三元组顺序 roll / pitch / yaw。原始 100 Hz 口径：每 10 ms 日志中的最新发布目标减同一行物理真值，离散样本等权 RMS；这是冻结验收口径。native 口径：控制器实际消费参考减匹配的物理真值，按约 1 ms 实际时间间隔加权 RMS；作为补充诊断，不能替换原门槛。

两种口径均保留首 3 秒。评分全段为仿真时间 (35,55.48] s，first3=(35,38]，after3=(38,55.48]，cruise=(39,54]，tail=(54,55.48]。after3 由全段与首 3 秒平方误差积分相减计算。每工况汇总为两个种子均方误差等权后开方；RP=√((roll_RMSE²+pitch_RMSE²)/2)。ALL9 为九工况等权均方汇总，仅描述，不能代替原 equal7 门槛。

## 冻结门槛及结论

非航点工况每轴 RMSE ≤ PID×1.15+0.02 deg/s；航点每轴绝对误差 P99 ≤ PID×1.10+0.5 deg/s。100 Hz 电机变化指标 ≤ PID×1.15+0.0001。低速/横移还要求全段及巡航 roll、pitch 分别通过前述界限，候选与 PID 速度覆盖率均 ≥0.8。保护工况各相位各轴两种子算术均值不得劣于 A，保护工况及 equal7 电机均值须下降；重复范围重叠不能声称确定改善。equal7 把 lowY、lateral+、lateral− 电机差值先等权合为一组，再与其余六组等权。

| 检查 | A | final40 |
| --- | --- | --- |
| all9_main_pass | 14 | 15 |
| original7_main_pass | 10 | 11 |
| all4_low_pass | 4 | 4 |
| original2_low_pass | 0 | 1 |

| 总条件 | 结果 |
| --- | --- |
| original_PID_guards_pass | False |
| protected_tracking_no_mean_regression | False |
| protected_and_equal7_motor_mean_lower | False |
| motor_repeat_overlap_inconclusive | True |
| equal7_motor_mean_delta | 8.5627802659899e-06 |
| performance_conditions_pass | False |

主门槛失败：lowX/470301 pitch，lowX/470302 roll，lowY/470301 pitch；超界分别 0.117339、0.090736、0.090742 deg/s。低速附加门槛另有 lateral+/470302 覆盖率 0.786 < 0.8。这些失败均保留，不以 RP 合并改善抵消逐轴失败。

## 原始 100 Hz：逐工况全段

| 工况 | PID R/P/Y | A R/P/Y | final40 R/P/Y | RP：PID/A/final40 |
| --- | --- | --- | --- | --- |
| hover_hold | 0.058007 / 0.053728 / 0.022131 | 0.041289 / 0.048722 / 0.021871 | 0.040985 / 0.047478 / 0.020650 | 0.055908 / 0.045158 / 0.044350 |
| hover_transition | 0.258979 / 0.210644 / 0.043880 | 0.144892 / 0.157102 / 0.036818 | 0.198761 / 0.156964 / 0.035527 | 0.236052 / 0.151121 / 0.179086 |
| waypoints | 2.064742 / 1.263436 / 0.731410 | 1.255337 / 0.852806 / 0.792536 | 1.235673 / 0.845014 / 0.766989 | 1.711641 / 1.073114 / 1.058522 |
| combined_native_waypoints | 2.242940 / 1.373943 / 0.834464 | 1.416866 / 0.883242 / 0.766045 | 1.376786 / 0.874009 / 0.683584 | 1.859906 / 1.180598 / 1.153133 |
| noise200_hover_hold | 0.084400 / 0.088701 / 0.044654 | 0.074751 / 0.090236 / 0.044536 | 0.076033 / 0.090846 / 0.039036 | 0.086577 / 0.082856 / 0.083768 |
| low_x_050_goal100 | 0.311346 / 0.144850 / 0.056610 | 0.210552 / 0.169959 / 0.063223 | 0.206294 / 0.200620 / 0.055573 | 0.242814 / 0.191335 / 0.203477 |
| low_y_025_goal100 | 0.236911 / 0.104566 / 0.030185 | 0.195259 / 0.194502 / 0.046525 | 0.174285 / 0.175601 / 0.035505 | 0.183113 / 0.194881 / 0.174944 |
| lateral_plus_goal100 | 0.339996 / 0.068358 / 0.023750 | 0.275273 / 0.057836 / 0.023722 | 0.275392 / 0.056550 / 0.021209 | 0.245224 / 0.198897 / 0.198795 |
| lateral_minus_goal100 | 0.353166 / 0.067153 / 0.024090 | 0.280144 / 0.058812 / 0.022631 | 0.252083 / 0.056487 / 0.022157 | 0.254201 / 0.202410 / 0.182670 |
| ALL9_equal_episode | 1.041627 / 0.630715 / 0.371328 | 0.653910 / 0.423755 / 0.369082 | 0.639593 / 0.420391 / 0.343844 | 0.861043 / 0.550983 / 0.541206 |

### 首 3 秒、后段和巡航 RP

| 工况 | 相位 | PID | A | final40 |
| --- | --- | --- | --- | --- |
| hover_hold | first3 | 0.060199 | 0.051634 | 0.050164 |
| hover_hold | after3 | 0.055138 | 0.043951 | 0.043274 |
| hover_hold | cruise | 0.055844 | 0.043706 | 0.043085 |
| hover_transition | first3 | 0.602641 | 0.380339 | 0.455415 |
| hover_transition | after3 | 0.054346 | 0.043933 | 0.044504 |
| hover_transition | cruise | 0.054281 | 0.044455 | 0.044254 |
| waypoints | first3 | 2.033910 | 1.337249 | 1.330203 |
| waypoints | after3 | 1.650016 | 1.020935 | 1.004534 |
| waypoints | cruise | 1.718481 | 1.023095 | 1.040097 |
| combined_native_waypoints | first3 | 2.293333 | 1.435161 | 1.432394 |
| combined_native_waypoints | after3 | 1.774909 | 1.131164 | 1.098087 |
| combined_native_waypoints | cruise | 1.853767 | 1.175511 | 1.144456 |
| noise200_hover_hold | first3 | 0.094555 | 0.090461 | 0.097834 |
| noise200_hover_hold | after3 | 0.085133 | 0.081479 | 0.081109 |
| noise200_hover_hold | cruise | 0.086669 | 0.082290 | 0.080403 |
| low_x_050_goal100 | first3 | 0.592348 | 0.458234 | 0.494834 |
| low_x_050_goal100 | after3 | 0.094121 | 0.082792 | 0.080526 |
| low_x_050_goal100 | cruise | 0.083454 | 0.084297 | 0.080812 |
| low_y_025_goal100 | first3 | 0.456626 | 0.494844 | 0.442416 |
| low_y_025_goal100 | after3 | 0.059161 | 0.049706 | 0.047600 |
| low_y_025_goal100 | cruise | 0.056409 | 0.046760 | 0.046134 |
| lateral_plus_goal100 | first3 | 0.624143 | 0.505913 | 0.506274 |
| lateral_plus_goal100 | after3 | 0.059987 | 0.049219 | 0.048083 |
| lateral_plus_goal100 | cruise | 0.056154 | 0.048048 | 0.046891 |
| lateral_minus_goal100 | first3 | 0.647855 | 0.515302 | 0.462818 |
| lateral_minus_goal100 | after3 | 0.060615 | 0.049280 | 0.048303 |
| lateral_minus_goal100 | cruise | 0.057137 | 0.047233 | 0.047529 |
| ALL9_equal_episode | first3 | 1.112583 | 0.743846 | 0.741754 |
| ALL9_equal_episode | after3 | 0.810057 | 0.510611 | 0.498744 |
| ALL9_equal_episode | cruise | 0.844575 | 0.522073 | 0.518007 |

## native 实际消费参考：逐工况全段

| 工况 | PID R/P/Y | A R/P/Y | final40 R/P/Y | RP：PID/A/final40 |
| --- | --- | --- | --- | --- |
| hover_hold | 0.057491 / 0.053188 / 0.021917 | 0.040907 / 0.048189 / 0.021706 | 0.040972 / 0.047384 / 0.020524 | 0.055382 / 0.044696 / 0.044294 |
| hover_transition | 0.250460 / 0.192626 / 0.042583 | 0.141644 / 0.145024 / 0.035781 | 0.187139 / 0.149237 / 0.034301 | 0.223422 / 0.143344 / 0.169252 |
| waypoints | 2.019891 / 1.243707 / 0.710175 | 1.224115 / 0.808756 / 0.724910 | 1.211247 / 0.807685 / 0.723407 | 1.677314 / 1.037435 / 1.029435 |
| combined_native_waypoints | 2.197944 / 1.310890 / 0.744562 | 1.373367 / 0.869349 / 0.762947 | 1.373736 / 0.872199 / 0.761272 | 1.809612 / 1.149327 / 1.150626 |
| noise200_hover_hold | 0.084434 / 0.088168 / 0.044302 | 0.074691 / 0.089960 / 0.044310 | 0.075691 / 0.090661 / 0.038802 | 0.086321 / 0.082678 / 0.083512 |
| low_x_050_goal100 | 0.304553 / 0.156316 / 0.057666 | 0.199746 / 0.156268 / 0.061252 | 0.201186 / 0.199662 / 0.055389 | 0.242061 / 0.179329 / 0.200425 |
| low_y_025_goal100 | 0.234252 / 0.100353 / 0.029711 | 0.171717 / 0.164276 / 0.042827 | 0.176276 / 0.182565 / 0.035634 | 0.180201 / 0.168038 / 0.179448 |
| lateral_plus_goal100 | 0.336362 / 0.067646 / 0.023480 | 0.267941 / 0.057250 / 0.023369 | 0.266931 / 0.056182 / 0.020968 | 0.242606 / 0.193739 / 0.192884 |
| lateral_minus_goal100 | 0.338496 / 0.065822 / 0.023865 | 0.268295 / 0.057330 / 0.022347 | 0.270169 / 0.056963 / 0.021912 | 0.243836 / 0.193996 / 0.195239 |
| ALL9_equal_episode | 1.019773 / 0.610679 / 0.344530 | 0.634654 / 0.408149 / 0.352441 | 0.633552 / 0.411693 / 0.351381 | 0.840495 / 0.533559 / 0.534265 |

### 首 3 秒、后段和巡航 RP

| 工况 | 相位 | PID | A | final40 |
| --- | --- | --- | --- | --- |
| hover_hold | first3 | 0.059718 | 0.051472 | 0.050761 |
| hover_hold | after3 | 0.054603 | 0.043427 | 0.043086 |
| hover_hold | cruise | 0.055295 | 0.043199 | 0.042927 |
| hover_transition | first3 | 0.569119 | 0.359350 | 0.429101 |
| hover_transition | after3 | 0.053813 | 0.043722 | 0.044294 |
| hover_transition | cruise | 0.053711 | 0.044275 | 0.044029 |
| waypoints | first3 | 2.016418 | 1.273732 | 1.272213 |
| waypoints | after3 | 1.611960 | 0.991234 | 0.981751 |
| waypoints | cruise | 1.678796 | 0.993372 | 1.018429 |
| combined_native_waypoints | first3 | 2.193044 | 1.420158 | 1.410427 |
| combined_native_waypoints | after3 | 1.735308 | 1.096139 | 1.099885 |
| combined_native_waypoints | cruise | 1.812751 | 1.138164 | 1.143271 |
| noise200_hover_hold | first3 | 0.094501 | 0.090750 | 0.097541 |
| noise200_hover_hold | after3 | 0.084838 | 0.081213 | 0.080860 |
| noise200_hover_hold | cruise | 0.086248 | 0.081991 | 0.080166 |
| low_x_050_goal100 | first3 | 0.590287 | 0.424078 | 0.486605 |
| low_x_050_goal100 | after3 | 0.094068 | 0.082541 | 0.080165 |
| low_x_050_goal100 | cruise | 0.082526 | 0.084662 | 0.080460 |
| low_y_025_goal100 | first3 | 0.449012 | 0.423693 | 0.454661 |
| low_y_025_goal100 | after3 | 0.058684 | 0.047681 | 0.047440 |
| low_y_025_goal100 | cruise | 0.056242 | 0.045850 | 0.045861 |
| lateral_plus_goal100 | first3 | 0.617654 | 0.492254 | 0.490541 |
| lateral_plus_goal100 | after3 | 0.059033 | 0.048885 | 0.047868 |
| lateral_plus_goal100 | cruise | 0.055071 | 0.047962 | 0.047170 |
| lateral_minus_goal100 | first3 | 0.620577 | 0.493281 | 0.496700 |
| lateral_minus_goal100 | after3 | 0.059707 | 0.048298 | 0.048150 |
| lateral_minus_goal100 | cruise | 0.056631 | 0.047020 | 0.047345 |
| ALL9_equal_episode | first3 | 1.081585 | 0.716803 | 0.725327 |
| ALL9_equal_episode | after3 | 0.791772 | 0.495341 | 0.494101 |
| ALL9_equal_episode | cruise | 0.825565 | 0.506243 | 0.512891 |

## 电机变化与覆盖率

motor100Hz = mean(abs(diff(ESC, time)))，对相邻 10 ms 样本及全部电机取平均，是无量纲归一化电机变化，未除以 dt；native torque slew 为归一化力矩/秒，total variation 为累计归一化力矩变化，二者不是电机指标，也不是晋级门槛。

| 工况 | motor PID | motor A | motor final40 | final 对 A |
| --- | --- | --- | --- | --- |
| hover_hold | 0.000896290 | 0.000995862 | 0.000986307 | -0.9595% |
| hover_transition | 0.000934294 | 0.000995144 | 0.001016265 | +2.1224% |
| waypoints | 0.001210092 | 0.001295008 | 0.001324751 | +2.2967% |
| combined_native_waypoints | 0.001261099 | 0.001332007 | 0.001341490 | +0.7119% |
| noise200_hover_hold | 0.001830081 | 0.001938059 | 0.001946751 | +0.4485% |
| low_x_050_goal100 | 0.000920141 | 0.001008793 | 0.001005561 | -0.3205% |
| low_y_025_goal100 | 0.000925386 | 0.000995431 | 0.000993419 | -0.2021% |
| lateral_plus_goal100 | 0.000902756 | 0.000967054 | 0.000980488 | +1.3892% |
| lateral_minus_goal100 | 0.000893129 | 0.000962528 | 0.000962168 | -0.0373% |
| ALL9_equal_episode | 0.001085919 | 0.001165543 | 0.001173022 | +0.6417% |

| 工况 | 种子 | 模型 | 速度覆盖率 |
| --- | --- | --- | --- |
| low_x_050_goal100 | 470301 | candidate | 0.881333 |
| low_x_050_goal100 | 470301 | pid | 0.854667 |
| low_x_050_goal100 | 470301 | A | 0.873333 |
| low_y_025_goal100 | 470301 | pid | 0.850667 |
| low_y_025_goal100 | 470301 | A | 0.850667 |
| low_y_025_goal100 | 470301 | candidate | 0.853333 |
| lateral_plus_goal100 | 470301 | A | 0.842000 |
| lateral_plus_goal100 | 470301 | candidate | 0.830000 |
| lateral_plus_goal100 | 470301 | pid | 0.822000 |
| lateral_minus_goal100 | 470301 | candidate | 0.858667 |
| lateral_minus_goal100 | 470301 | pid | 0.830667 |
| lateral_minus_goal100 | 470301 | A | 0.834000 |
| low_x_050_goal100 | 470302 | pid | 0.880667 |
| low_x_050_goal100 | 470302 | A | 0.874000 |
| low_x_050_goal100 | 470302 | candidate | 0.808000 |
| low_y_025_goal100 | 470302 | A | 0.804000 |
| low_y_025_goal100 | 470302 | candidate | 0.873333 |
| low_y_025_goal100 | 470302 | pid | 0.847333 |
| lateral_plus_goal100 | 470302 | candidate | 0.786000 |
| lateral_plus_goal100 | 470302 | pid | 0.852000 |
| lateral_plus_goal100 | 470302 | A | 0.824000 |
| lateral_minus_goal100 | 470302 | pid | 0.876000 |
| lateral_minus_goal100 | 470302 | A | 0.872667 |
| lateral_minus_goal100 | 470302 | candidate | 0.850667 |

ALL9 电机均值 final40 对 A +0.641715%，对 PID +8.02117%；native RP slew RMS：PID 0.356148、A 0.498600、final40 0.500019（归一化力矩/s）。保护工况电机重复范围有重叠。

## 全部 54 项逐种子全段指标

### original100Hz

| 工况 | 种子 | 模型 | roll | pitch | yaw | 存活步数 |
| --- | --- | --- | --- | --- | --- | --- |
| hover_hold | 470301 | pid | 0.060748 | 0.061325 | 0.023081 | 2048 |
| hover_hold | 470301 | A | 0.045183 | 0.056003 | 0.023204 | 2048 |
| hover_hold | 470301 | candidate | 0.044324 | 0.054518 | 0.019058 | 2048 |
| hover_transition | 470301 | A | 0.146431 | 0.216735 | 0.042019 | 2048 |
| hover_transition | 470301 | candidate | 0.129988 | 0.213084 | 0.022293 | 2048 |
| hover_transition | 470301 | pid | 0.155514 | 0.281633 | 0.027620 | 2048 |
| waypoints | 470301 | candidate | 1.164412 | 0.696002 | 0.668087 | 2048 |
| waypoints | 470301 | pid | 2.088074 | 1.131249 | 0.825718 | 2048 |
| waypoints | 470301 | A | 1.206338 | 0.712603 | 0.720889 | 2048 |
| combined_native_waypoints | 470301 | pid | 2.217443 | 1.155153 | 0.707430 | 2048 |
| combined_native_waypoints | 470301 | A | 1.383384 | 0.799846 | 0.872536 | 2048 |
| combined_native_waypoints | 470301 | candidate | 1.303485 | 0.732388 | 0.571263 | 2048 |
| noise200_hover_hold | 470301 | A | 0.079292 | 0.107620 | 0.039709 | 2048 |
| noise200_hover_hold | 470301 | candidate | 0.082356 | 0.109806 | 0.037660 | 2048 |
| noise200_hover_hold | 470301 | pid | 0.092172 | 0.104737 | 0.043761 | 2048 |
| low_x_050_goal100 | 470301 | candidate | 0.140438 | 0.222619 | 0.059295 | 2048 |
| low_x_050_goal100 | 470301 | pid | 0.421875 | 0.074157 | 0.042304 | 2048 |
| low_x_050_goal100 | 470301 | A | 0.113983 | 0.234409 | 0.064940 | 2048 |
| low_y_025_goal100 | 470301 | pid | 0.221373 | 0.059300 | 0.027965 | 2048 |
| low_y_025_goal100 | 470301 | A | 0.081107 | 0.194254 | 0.049705 | 2048 |
| low_y_025_goal100 | 470301 | candidate | 0.232262 | 0.178938 | 0.043262 | 2048 |
| lateral_plus_goal100 | 470301 | A | 0.266321 | 0.054713 | 0.024261 | 2048 |
| lateral_plus_goal100 | 470301 | candidate | 0.242083 | 0.051755 | 0.022277 | 2048 |
| lateral_plus_goal100 | 470301 | pid | 0.361596 | 0.071398 | 0.023233 | 2048 |
| lateral_minus_goal100 | 470301 | candidate | 0.243943 | 0.052920 | 0.021892 | 2048 |
| lateral_minus_goal100 | 470301 | pid | 0.344807 | 0.069759 | 0.023716 | 2048 |
| lateral_minus_goal100 | 470301 | A | 0.248553 | 0.053111 | 0.022122 | 2048 |
| hover_hold | 470302 | A | 0.036986 | 0.040141 | 0.020452 | 2048 |
| hover_hold | 470302 | candidate | 0.037348 | 0.039193 | 0.022127 | 2048 |
| hover_hold | 470302 | pid | 0.055129 | 0.044861 | 0.021138 | 2048 |
| hover_transition | 470302 | candidate | 0.249228 | 0.062213 | 0.045026 | 2048 |
| hover_transition | 470302 | pid | 0.331596 | 0.097079 | 0.055569 | 2048 |
| hover_transition | 470302 | A | 0.143337 | 0.048871 | 0.030749 | 2048 |
| waypoints | 470302 | pid | 2.041143 | 1.383046 | 0.622986 | 2048 |
| waypoints | 470302 | A | 1.302494 | 0.973012 | 0.858222 | 2048 |
| waypoints | 470302 | candidate | 1.303043 | 0.971431 | 0.854519 | 2048 |
| combined_native_waypoints | 470302 | A | 1.449574 | 0.959415 | 0.642129 | 2048 |
| combined_native_waypoints | 470302 | candidate | 1.446376 | 0.995687 | 0.779893 | 2048 |
| combined_native_waypoints | 470302 | pid | 2.268150 | 1.562389 | 0.944565 | 2048 |
| noise200_hover_hold | 470302 | candidate | 0.069135 | 0.066699 | 0.040365 | 2048 |
| noise200_hover_hold | 470302 | pid | 0.075835 | 0.069036 | 0.045531 | 2048 |
| noise200_hover_hold | 470302 | A | 0.069916 | 0.068577 | 0.048889 | 2048 |
| low_x_050_goal100 | 470302 | pid | 0.126071 | 0.190956 | 0.067969 | 2048 |
| low_x_050_goal100 | 470302 | A | 0.275085 | 0.053143 | 0.061458 | 2048 |
| low_x_050_goal100 | 470302 | candidate | 0.255718 | 0.175891 | 0.051583 | 2048 |
| low_y_025_goal100 | 470302 | A | 0.263958 | 0.194749 | 0.043111 | 2048 |
| low_y_025_goal100 | 470302 | candidate | 0.082490 | 0.172200 | 0.025487 | 2048 |
| low_y_025_goal100 | 470302 | pid | 0.251490 | 0.135468 | 0.032253 | 2048 |
| lateral_plus_goal100 | 470302 | candidate | 0.305086 | 0.060970 | 0.020083 | 2048 |
| lateral_plus_goal100 | 470302 | pid | 0.316927 | 0.065177 | 0.024255 | 2048 |
| lateral_plus_goal100 | 470302 | A | 0.283943 | 0.060798 | 0.023170 | 2048 |
| lateral_minus_goal100 | 470302 | pid | 0.361332 | 0.064442 | 0.024459 | 2048 |
| lateral_minus_goal100 | 470302 | A | 0.308517 | 0.064007 | 0.023128 | 2048 |
| lateral_minus_goal100 | 470302 | candidate | 0.259969 | 0.059842 | 0.022420 | 2048 |

### native_consumed

| 工况 | 种子 | 模型 | roll | pitch | yaw | 存活步数 |
| --- | --- | --- | --- | --- | --- | --- |
| hover_hold | 470301 | pid | 0.060246 | 0.060875 | 0.022818 | 2048 |
| hover_hold | 470301 | A | 0.044809 | 0.055869 | 0.022958 | 2048 |
| hover_hold | 470301 | candidate | 0.044416 | 0.054678 | 0.018892 | 2048 |
| hover_transition | 470301 | A | 0.138346 | 0.199179 | 0.040246 | 2048 |
| hover_transition | 470301 | candidate | 0.125482 | 0.202361 | 0.022031 | 2048 |
| hover_transition | 470301 | pid | 0.147326 | 0.255456 | 0.026643 | 2048 |
| waypoints | 470301 | candidate | 1.149381 | 0.689406 | 0.713217 | 2048 |
| waypoints | 470301 | pid | 2.015219 | 1.074944 | 0.708441 | 2048 |
| waypoints | 470301 | A | 1.176428 | 0.691446 | 0.712960 | 2048 |
| combined_native_waypoints | 470301 | pid | 2.192515 | 1.142742 | 0.744337 | 2048 |
| combined_native_waypoints | 470301 | A | 1.315736 | 0.744517 | 0.748933 | 2048 |
| combined_native_waypoints | 470301 | candidate | 1.320230 | 0.758895 | 0.751452 | 2048 |
| noise200_hover_hold | 470301 | A | 0.079189 | 0.107489 | 0.039459 | 2048 |
| noise200_hover_hold | 470301 | candidate | 0.081967 | 0.109661 | 0.037388 | 2048 |
| noise200_hover_hold | 470301 | pid | 0.091878 | 0.104526 | 0.043289 | 2048 |
| low_x_050_goal100 | 470301 | candidate | 0.140763 | 0.227576 | 0.059287 | 2048 |
| low_x_050_goal100 | 470301 | pid | 0.411117 | 0.072775 | 0.041885 | 2048 |
| low_x_050_goal100 | 470301 | A | 0.114038 | 0.214723 | 0.062687 | 2048 |
| low_y_025_goal100 | 470301 | pid | 0.226354 | 0.059312 | 0.027893 | 2048 |
| low_y_025_goal100 | 470301 | A | 0.078783 | 0.162824 | 0.045567 | 2048 |
| low_y_025_goal100 | 470301 | candidate | 0.234312 | 0.182450 | 0.043457 | 2048 |
| lateral_plus_goal100 | 470301 | A | 0.269942 | 0.054804 | 0.023849 | 2048 |
| lateral_plus_goal100 | 470301 | candidate | 0.267460 | 0.053157 | 0.021989 | 2048 |
| lateral_plus_goal100 | 470301 | pid | 0.335953 | 0.069325 | 0.022982 | 2048 |
| lateral_minus_goal100 | 470301 | candidate | 0.269438 | 0.053839 | 0.021623 | 2048 |
| lateral_minus_goal100 | 470301 | pid | 0.335812 | 0.068868 | 0.023456 | 2048 |
| lateral_minus_goal100 | 470301 | A | 0.266841 | 0.053626 | 0.021865 | 2048 |
| hover_hold | 470302 | A | 0.036591 | 0.039024 | 0.020377 | 2048 |
| hover_hold | 470302 | candidate | 0.037209 | 0.038740 | 0.022036 | 2048 |
| hover_hold | 470302 | pid | 0.054597 | 0.044184 | 0.020978 | 2048 |
| hover_transition | 470302 | candidate | 0.233016 | 0.059944 | 0.043217 | 2048 |
| hover_transition | 470302 | pid | 0.322110 | 0.094616 | 0.054008 | 2048 |
| hover_transition | 470302 | A | 0.144866 | 0.048904 | 0.030672 | 2048 |
| waypoints | 470302 | pid | 2.024552 | 1.392161 | 0.711905 | 2048 |
| waypoints | 470302 | A | 1.270013 | 0.911085 | 0.736666 | 2048 |
| waypoints | 470302 | candidate | 1.270102 | 0.910731 | 0.733456 | 2048 |
| combined_native_waypoints | 470302 | A | 1.428675 | 0.978382 | 0.776709 | 2048 |
| combined_native_waypoints | 470302 | candidate | 1.425234 | 0.972389 | 0.770968 | 2048 |
| combined_native_waypoints | 470302 | pid | 2.203360 | 1.459796 | 0.744787 | 2048 |
| noise200_hover_hold | 470302 | candidate | 0.068845 | 0.066431 | 0.040166 | 2048 |
| noise200_hover_hold | 470302 | pid | 0.076268 | 0.067981 | 0.045292 | 2048 |
| noise200_hover_hold | 470302 | A | 0.069903 | 0.068056 | 0.048680 | 2048 |
| low_x_050_goal100 | 470302 | pid | 0.128404 | 0.208741 | 0.069975 | 2048 |
| low_x_050_goal100 | 470302 | A | 0.258441 | 0.052282 | 0.059783 | 2048 |
| low_x_050_goal100 | 470302 | candidate | 0.247260 | 0.167149 | 0.051196 | 2048 |
| low_y_025_goal100 | 470302 | A | 0.229710 | 0.165715 | 0.039899 | 2048 |
| low_y_025_goal100 | 470302 | candidate | 0.085113 | 0.182681 | 0.025516 | 2048 |
| low_y_025_goal100 | 470302 | pid | 0.241892 | 0.128932 | 0.031423 | 2048 |
| lateral_plus_goal100 | 470302 | candidate | 0.266401 | 0.059053 | 0.019894 | 2048 |
| lateral_plus_goal100 | 470302 | pid | 0.336770 | 0.065925 | 0.023968 | 2048 |
| lateral_plus_goal100 | 470302 | A | 0.265924 | 0.059595 | 0.022879 | 2048 |
| lateral_minus_goal100 | 470302 | pid | 0.341159 | 0.062628 | 0.024267 | 2048 |
| lateral_minus_goal100 | 470302 | A | 0.269741 | 0.060809 | 0.022819 | 2048 |
| lateral_minus_goal100 | 470302 | candidate | 0.270899 | 0.059924 | 0.022197 | 2048 |

## 模型、预算和失败记录

A 二进制 SHA256：`0bdfed33c8a2ab7ae5901d522f8c493184924c633c25e4ad129d5da4f8c24f1e`。final40 模型 ID 1866431061，SHA256 `7364e05c5bd5a55f818515e74fa31be325e23f2795f4fda895e9de302fa27e15`；训练 40 次正式更新，4×7 条采集。仅 154 个 RP 输出头参数可训练；主干、yaw 和 gate 冻结。

第 11 次更新起平滑梯度比例自适应只降不升，末值 alpha=0.2820385593449858；第 21 次起历史混合策略 memory 分位覆盖只作诊断，其余物理/速度/同步门槛保留。两次原始失败记录均保留。训练 ledger 计费 1,671,267 / 2,000,000。

开发验收共 55 次尝试：54 项有效 + 原 dev43 启动失败一次。原 PID lowX/470302 在评分前 SIGSEGV，内核记录指向 libzmq.so.5.2.4，深层根因未确定；其保守计费 56,000 保留，获准一次恢复后完成。验收计费 2,498,802 / 3,000,000 = 已知 NN 1,572,300 + PID 870,502 + 原失败保守 56,000，无待结算预留。最后驱动后处理缺元数据失败已用原 admission 绑定的字节一致元数据离线恢复；原失败/exit1 记录保留。

所有有效项的导出、同步、2048 存活、模型/runtime 哈希已审计。native PX4 SHA256 `6bf92a950c87f04668e5d313100876a83be1b57236df2ee4c007f53313dd025e`；plugin `16faecad2cbe2d587f4a0b663a656a715db0ffd7b91eceb9007eb2801d672ed1`。两种子不支持显著性或因果结论，同种子也不等于相同进入状态。

## 原始证据路径

- `@DRL_ROOT_WINDOWS@\experiments\robustness_v1_20261002\native_fresh_batch_rp_v47\development_recovery1\complete_review.json`
- `@DRL_ROOT_WINDOWS@\experiments\robustness_v1_20261002\native_fresh_batch_rp_v47\development_recovery1\final_evaluation_verification.json`
- `@DRL_ROOT_WINDOWS@\experiments\robustness_v1_20261002\native_fresh_batch_rp_v47\development_recovery1\final_metrics\summary.json`
- `@DRL_ROOT_WINDOWS@\experiments\robustness_v1_20261002\native_fresh_batch_rp_v47\development_recovery1\final_metrics\per_episode_phase_metrics.csv`
- `@DRL_ROOT_WINDOWS@\experiments\robustness_v1_20261002\native_fresh_batch_rp_v47\development_recovery1\final_metrics\per_episode_smoothness_coverage.csv`
- `@DRL_ROOT_WINDOWS@\experiments\robustness_v1_20261002\native_fresh_batch_rp_v47\memory_sampling_revision1\training\batch_3\actor.npz`
- `@DRL_ROOT_WINDOWS@\experiments\robustness_v1_20261002\native_fresh_batch_rp_v47\memory_sampling_revision1\training\batch_3\actor.bin`
- `@DRL_ROOT_WINDOWS@\experiments\robustness_v1_20261002\native_fresh_batch_rp_v47\memory_sampling_revision1\training\batch_3\recovery.pt`
- `@DRL_ROOT_WINDOWS@\experiments\robustness_v1_20261002\native_fresh_batch_rp_v47\ledger.json`
- `@DRL_ROOT_WINDOWS@\experiments\robustness_v1_20261002\native_fresh_batch_rp_v47\development_ledger.json`

## 冻结逐项门槛余量

余量=候选值−允许上限，≤0 为通过；角速度余量单位 deg/s，电机余量无量纲。航点使用 P99，其余使用 RMSE，不能混读。

| 工况 | 种子 | 模型 | 角速度指标 | roll余量 | pitch余量 | yaw余量 | 电机余量 | 通过 |
|---|---|---|---|---|---|---|---|---|
| hover_hold | 470301 | A | axis_rmse | -0.044677453 | -0.034521277 | -0.023339217 | -0.000149089 | True |
| hover_hold | 470301 | candidate | axis_rmse | -0.045536440 | -0.036005964 | -0.027485682 | -0.000152250 | True |
| hover_hold | 470302 | A | axis_rmse | -0.046411596 | -0.031449328 | -0.023857482 | -0.000120654 | True |
| hover_hold | 470302 | candidate | axis_rmse | -0.046050310 | -0.032396842 | -0.022181722 | -0.000136603 | True |
| hover_transition | 470301 | A | axis_rmse | -0.052409618 | -0.127143761 | -0.009744340 | -0.000140396 | True |
| hover_transition | 470301 | candidate | axis_rmse | -0.068853069 | -0.130794192 | -0.029469892 | -0.000136661 | True |
| hover_transition | 470302 | A | axis_rmse | -0.257998461 | -0.082769704 | -0.053155615 | -0.000218193 | True |
| hover_transition | 470302 | candidate | axis_rmse | -0.152106783 | -0.069427788 | -0.038879267 | -0.000179686 | True |
| waypoints | 470301 | A | axis_p99_abs | -4.169350111 | -2.096640842 | -0.934811240 | -0.000209227 | True |
| waypoints | 470301 | candidate | axis_p99_abs | -4.295868509 | -2.239278543 | -1.047688664 | -0.000158507 | True |
| waypoints | 470302 | A | axis_p99_abs | -3.345333937 | -2.203815815 | -0.251864964 | -0.000183968 | True |
| waypoints | 470302 | candidate | axis_p99_abs | -3.412208902 | -2.111497872 | -0.317810000 | -0.000175203 | True |
| combined_native_waypoints | 470301 | A | axis_p99_abs | -4.085303409 | -1.901860731 | -0.066666821 | -0.000260019 | True |
| combined_native_waypoints | 470301 | candidate | axis_p99_abs | -4.503925626 | -2.409613022 | -1.154116334 | -0.000255708 | True |
| combined_native_waypoints | 470302 | A | axis_p99_abs | -4.282029030 | -2.582874476 | -1.368337223 | -0.000176496 | True |
| combined_native_waypoints | 470302 | candidate | axis_p99_abs | -4.080614263 | -2.474227759 | -1.067718666 | -0.000161841 | True |
| noise200_hover_hold | 470301 | A | axis_rmse | -0.046706134 | -0.032827389 | -0.030616420 | -0.000279941 | True |
| noise200_hover_hold | 470301 | candidate | axis_rmse | -0.043641744 | -0.030641600 | -0.032664955 | -0.000231663 | True |
| noise200_hover_hold | 470302 | A | axis_rmse | -0.037294286 | -0.030814885 | -0.023470962 | -0.000253129 | True |
| noise200_hover_hold | 470302 | candidate | axis_rmse | -0.038075778 | -0.032692468 | -0.031994759 | -0.000284021 | True |
| low_x_050_goal100 | 470301 | A | axis_rmse | -0.391173034 | 0.129129135 | -0.003710268 | -0.000130798 | False |
| low_x_050_goal100 | 470301 | candidate | axis_rmse | -0.364717763 | 0.117338931 | -0.009354781 | -0.000135540 | False |
| low_x_050_goal100 | 470302 | A | axis_rmse | 0.110103771 | -0.186456380 | -0.036706257 | -0.000167940 | False |
| low_x_050_goal100 | 470302 | candidate | axis_rmse | 0.090736391 | -0.063708229 | -0.046581305 | -0.000169664 | False |
| low_y_025_goal100 | 470301 | A | axis_rmse | -0.193472614 | 0.106058389 | -0.002454049 | -0.000165368 | False |
| low_y_025_goal100 | 470301 | candidate | axis_rmse | -0.042317619 | 0.090742450 | -0.008897145 | -0.000161633 | False |
| low_y_025_goal100 | 470302 | A | axis_rmse | -0.045255872 | 0.018961424 | -0.013980337 | -0.000172157 | False |
| low_y_025_goal100 | 470302 | candidate | axis_rmse | -0.226723086 | -0.003587479 | -0.031604192 | -0.000179916 | True |
| lateral_plus_goal100 | 470301 | A | axis_rmse | -0.169514385 | -0.047395557 | -0.022456795 | -0.000193422 | True |
| lateral_plus_goal100 | 470301 | candidate | axis_rmse | -0.193752338 | -0.050353575 | -0.024440431 | -0.000149024 | True |
| lateral_plus_goal100 | 470302 | A | axis_rmse | -0.100522735 | -0.034154762 | -0.024722909 | -0.000148809 | True |
| lateral_plus_goal100 | 470302 | candidate | axis_rmse | -0.079380037 | -0.033983553 | -0.027810310 | -0.000166338 | True |
| lateral_minus_goal100 | 470301 | A | axis_rmse | -0.167975542 | -0.047111507 | -0.025150922 | -0.000156948 | True |
| lateral_minus_goal100 | 470301 | candidate | axis_rmse | -0.172585216 | -0.047302601 | -0.025381358 | -0.000156948 | True |
| lateral_minus_goal100 | 470302 | A | axis_rmse | -0.127014113 | -0.030101377 | -0.024999255 | -0.000172193 | True |
| lateral_minus_goal100 | 470302 | candidate | axis_rmse | -0.175562620 | -0.034266591 | -0.025707916 | -0.000172912 | True |

| 低速工况 | 种子 | 模型 | 全段R/P余量 | 巡航R/P余量 | 覆盖通过 | 总通过 |
|---|---|---|---|---|---|---|
| low_x_050_goal100 | 470301 | A | -0.391173034 / 0.129129135 | -0.027291177 / -0.030950528 | True | False |
| low_x_050_goal100 | 470301 | candidate | -0.364717763 / 0.117338931 | -0.035608912 / -0.032670425 | True | False |
| low_x_050_goal100 | 470302 | A | 0.110103771 / -0.186456380 | -0.039270056 / -0.031654739 | True | False |
| low_x_050_goal100 | 470302 | candidate | 0.090736391 / -0.063708229 | -0.040245949 / -0.032354587 | True | False |
| low_y_025_goal100 | 470301 | A | -0.193472614 / 0.106058389 | -0.046057065 / -0.036769416 | True | False |
| low_y_025_goal100 | 470301 | candidate | -0.042317619 / 0.090742450 | -0.044784310 / -0.037285955 | True | False |
| low_y_025_goal100 | 470302 | A | -0.045255872 / 0.018961424 | -0.035861191 / -0.034219183 | True | False |
| low_y_025_goal100 | 470302 | candidate | -0.226723086 / -0.003587479 | -0.039358342 / -0.034256665 | True | True |
| lateral_plus_goal100 | 470301 | A | -0.169514385 / -0.047395557 | -0.033984167 / -0.039775334 | True | True |
| lateral_plus_goal100 | 470301 | candidate | -0.193752338 / -0.050353575 | -0.032677839 / -0.044077226 | True | True |
| lateral_plus_goal100 | 470302 | A | -0.100522735 / -0.034154762 | -0.043367215 / -0.030391463 | True | True |
| lateral_plus_goal100 | 470302 | candidate | -0.079380037 / -0.033983553 | -0.043718959 / -0.031919522 | False | False |
| lateral_minus_goal100 | 470301 | A | -0.167975542 / -0.047111507 | -0.046479971 / -0.043719608 | True | True |
| lateral_minus_goal100 | 470301 | candidate | -0.172585216 / -0.047302601 | -0.042231211 / -0.043635464 | True | True |
| lateral_minus_goal100 | 470302 | A | -0.127014113 / -0.030101377 | -0.038024855 / -0.027221017 | True | True |
| lateral_minus_goal100 | 470302 | candidate | -0.175562620 / -0.034266591 | -0.042246502 / -0.026529456 | True | True |
