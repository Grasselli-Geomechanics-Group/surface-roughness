#%%
import matplotlib.pyplot as plt
import numpy as np
from surface_roughness import Surface

plt.rcParams.update({'font.size': 11})
file = r'tests/example_surface.stl'
s = Surface(file)

s.evaluate_delta_t()

delta_t_az = s.delta_t("az")
delta_t = s.delta_t("delta_t")
# deltastar_t = s.delta_t("delta*_t")
n_tri = s.delta_t("n_tri")

# delta_a_az = s.delta_a("az")
# delta_a = s.delta_a("delta_a")
# deltastar_a = s.delta_a("delta*_a")

# delta_n_az = s.delta_n("az")
# delta_n = s.delta_n("delta_n")
# deltastar_n = s.delta_n("delta*_n")

thetacp1_az = s.thetamax_cp1("az")
thetacp1 = s.thetamax_cp1("thetamax_cp1")
dip_data = s.thetamax_cp1('dip_bin_data')
s._thetamax_cp1.to_csv(f'{file}.csv')
# meandip_az = s.meandip("az")
# meandip = s.meandip("mean_dip")
# stddip = s.meandip("std_dip")
#%%
plt.figure()
plt.polar(delta_t_az,delta_t,'o-b',markersize=4,label=r'$\Delta_T$')
# plt.polar(delta_t_az,deltastar_t,'b--',label=r'$\Delta^*_T$')
# plt.polar(delta_a_az,delta_a,'r',label=r'$\Delta_A$')
# plt.polar(delta_a_az,deltastar_a,'r--',label=r'$\Delta^*_A$')
# plt.polar(delta_n_az,delta_n,label=r'$\Delta_N$')
# plt.polar(delta_n_az,deltastar_n,label=r'$\Delta^*_N$')
plt.polar(thetacp1_az,thetacp1,'x:',color='tab:orange',label=r'$\theta^*_{max}/(C+1)$')
# plt.polar(meandip_az,meandip,'g',label=r'Mean apparent dip')
# plt.polar(meandip_az,stddip,'g--',label=r'Std. dev. apparent dip')
plt.yticks(np.arange(0,21,3))
plt.legend(loc="lower center",bbox_to_anchor=(0.5,-0.33),ncol=2)
plt.gcf().set_figwidth(3.543)
# plt.subplots_adjust(bottom=0.7)
# plt.tight_layout()
# plt.show()
# plt.savefig(f'{file}roughness_comparison.svg')
# %%
