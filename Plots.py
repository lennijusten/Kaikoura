import matplotlib.pyplot as plt
import numpy as np

thresh = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                   0.95, 1])

prms = np.array(
    [0.335815505521, 0.334253036624, 0.327030057585, 0.318646800609, 0.309402651934, 0.301508876648, 0.291429046809,
     0.281068505244, 0.263755504156, 0.252463799753, 0.237692799287, 0.219773198008, 0.206043225433, 0.192136982526,
     0.180233072606, 0.158428220688, 0.1438097249, 0.121810298355, 0.10405098488, np.nan])
srms = np.array(
    [0.461920586049, 0.462492284215, 0.45511641201, 0.439711826143, 0.43066610045, 0.402667054445, 0.392794216644,
     0.368258226849, 0.334560926205, 0.310278250092, 0.268663666895, 0.255144201128, 0.225162442816, 0.231791540441,
     0.227326836607, 0.197576045881, 0.107130059334, 0.143542040953, np.nan, np.nan])

p_mean = np.array(
    [-0.00508091719872, -0.00502012347408, -0.0073437369482, -0.0094032638635, -0.0112535156602, -0.0141497373758,
     -0.016132539529, -0.0169547883996, -0.0209655910192, -0.0254273959991, -0.0284973426013, -0.0328554038081,
     -0.0345859874148, -0.0350266933788,  -0.0376216778545, -0.0398589805125, -0.0357520720358, -0.0369142608983,
     -0.0358884659443, np.nan])
s_mean = np.array(
    [-0.174215104225, -0.174244655367, -0.172302508523, -0.162560458213, -0.164705069909, -0.153201485623,
     -0.146948288591, -0.146949931818, -0.17245802521, -0.145484965347, -0.132310659341, -0.134099216783,
     -0.128849522124, -0.135866753086, -0.117773020833, -0.118919578947, -0.0456849, -0.06462875,
     np.nan, np.nan])

P = np.array([5284, 5240, 5176, 5124, 5076, 5001, 4904, 4787, 4666, 4507, 4339, 4131, 3890, 3577, 3236, 2811, 2266,
              1531, 652, 0])
S = np.array([5013, 4741, 4343, 3830, 3251, 2724, 2256, 1817, 1437, 1135, 883, 655, 463, 290, 160, 74, 32,
              8, 1, 0])

P_out = np.array([266, 237, 209, 195, 185, 163, 142, 126, 117, 102, 89, 81, 70, 53, 43, 35, 27, 15, 6, 0])
S_out = np.array([16, 15, 14, 14, 14, 14, 13, 13, 11, 6, 5, 5, 3, 3, 1, 0, 0, 0, 0, 0])

fig, ax1 = plt.subplots()
plt.title('Trade-off between RMSE and available data for PhaseNet P picks')
ax1.plot(thresh, prms, c='#0052cc')
ax1.grid(axis='both', alpha=0.4)
ax1.set(xlabel='PhaseNet probability threshold')
ax1.set_ylabel('RMSE', color='#3366ff')
ax1.tick_params(axis='y', labelcolor='#0052cc')
plt.ylim(0, 1)
ax2 = ax1.twinx()
ax2.set_ylabel('% of data with picks', color='#cc0000')
ax2.tick_params(axis='y', labelcolor='#cc0000')
ax2.plot(thresh, P / 5367, c='#cc0000')
fig.tight_layout()
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()

fig, ax1 = plt.subplots()
plt.title('Trade-off between RMSE and available data for PhaseNet S picks')
ax1.plot(thresh, srms, c='#0052cc')
ax1.grid(axis='both', alpha=0.4)
ax1.set(xlabel='PhaseNet probability threshold')
ax1.set_ylabel('RMSE', color='#3366ff')
ax1.tick_params(axis='y', labelcolor='#0052cc')
plt.ylim(0, 1)
ax2 = ax1.twinx()
ax2.set_ylabel('% of data with picks', color='#cc0000')
ax2.tick_params(axis='y', labelcolor='#cc0000')
ax2.plot(thresh, S / 5367, c='#cc0000')
fig.tight_layout()
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.show()

fig, ax1 = plt.subplots()
ax1.plot(thresh, P_out, label='P')
ax1.plot(thresh, S_out, label='S')
ax1.grid(axis='both', alpha=0.4)
ax1.set(xlabel='PhaseNet probability threshold', ylabel='n outliers (res > 2s)')
plt.title('Number of outliers captured at PhaseNet probability thresholds')
plt.legend()
plt.show()

fig, ax1 = plt.subplots()
ax1.plot(thresh, p_mean, label='P')
ax1.plot(thresh, s_mean, label='S')
ax1.grid(axis='both', alpha=0.4)
ax1.set(xlabel='PhaseNet probability threshold', ylabel='Mean residual (s)')
plt.title('Mean residual by PhaseNet probability threshold')
plt.legend()
plt.show()